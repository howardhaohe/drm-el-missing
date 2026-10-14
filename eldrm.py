"""Density ratio estimation by dual empirical likelihood.

The full basis Q includes its leading intercept. Parameter order is population
major: [alpha_1, beta_11, ..., alpha_m, beta_m1, ...]; theta_0 is fixed at zero.
The active missing-data formulation is miss4 in the supplied eldrm(1).R.
Formula/API adaptation of drmdel 1.3.2 (Song Cai, GPL-3.0); see LICENSE.
"""
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def basis_matrix(x, basis_func=5):
    """Return Q(x). A callable takes a scalar and returns [1, h(x)...].

    Integer bases 1--11 follow drmdel, with an intercept prepended.
    log/sqrt bases use abs(x), as in the package; log(0) is rejected.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or x.size == 0:
        raise ValueError('x must be a nonempty finite one-dimensional vector')
    if callable(basis_func):
        q = np.asarray([basis_func(float(v)) for v in x], dtype=float)
    else:
        if basis_func not in range(1, 12):
            raise ValueError('basis_func must be 1--11 or a scalar callable')
        need_log = basis_func in (2, 6, 7, 8, 9, 11)
        if need_log and np.any(x == 0):
            raise ValueError('log(abs(x)) requires nonzero x')
        logx = np.log(np.abs(x)) if need_log else np.zeros_like(x)
        sqrtx = np.sqrt(np.abs(x))
        terms = {1:[x], 2:[logx], 3:[sqrtx], 4:[x*x],
                 5:[x,x*x], 6:[x,logx], 7:[logx,sqrtx,x],
                 8:[logx,sqrtx,x*x], 9:[logx,x,x*x],
                 10:[sqrtx,x,x*x], 11:[logx,sqrtx,x,x*x]}
        q = np.column_stack([np.ones(x.size)] + terms[basis_func])
    if q.ndim != 2 or q.shape[0] != x.size or q.shape[1] < 2:
        raise ValueError('Q must be an n by p matrix with p >= 2')
    if not np.all(np.isfinite(q)) or not np.allclose(q[:, 0], 1):
        raise ValueError('Q must be finite and its first column must be one')
    return q


def _counts(n_samples, n):
    c = np.asarray(n_samples)
    if c.ndim != 1 or len(c) < 2 or not np.all(np.isfinite(c)):
        raise ValueError('n_samples must contain at least two positive counts')
    if np.any(c <= 0) or np.any(c != np.floor(c)) or c.sum() != n:
        raise ValueError('positive integer n_samples must sum to len(x)')
    return c.astype(int)


class _DELProblem:
    """Weighted support representation; signed weights implement miss4."""
    def __init__(self, q, labels, weights, counts):
        self.q, self.labels, self.weights = q, labels, weights
        self.counts = counts
        self.n = int(counts.sum())
        self.m, self.p = len(counts)-1, q.shape[1]
        self.logrho = np.log(counts/self.n)

    def _state(self, par, q=None):
        q = self.q if q is None else q
        par = np.asarray(par, dtype=float)
        if par.shape != (self.m*self.p,) or not np.all(np.isfinite(par)):
            raise ValueError('par must be a finite vector of length m*p')
        theta = np.vstack([np.zeros(self.p), par.reshape(self.m,self.p)])
        eta = q @ theta.T
        z = eta + self.logrho
        log_s = logsumexp(z, axis=1)
        probs = np.exp(z-log_s[:,None])
        return eta, log_s, probs

    def value_grad(self, par, q=None):
        q = self.q if q is None else q
        eta, log_s, probs = self._state(par, q)
        value = self.weights @ (log_s - eta[np.arange(len(q)),self.labels])
        residual = probs[:,1:].copy()
        idx = self.labels > 0
        residual[np.flatnonzero(idx), self.labels[idx]-1] -= 1
        grad = (residual.T @ (self.weights[:,None]*q)).ravel()
        return float(value), grad

    def hessian(self, par, q=None):
        q = self.q if q is None else q
        probs = self._state(par, q)[2][:,1:]
        hess = np.empty((self.m*self.p, self.m*self.p))
        for r in range(self.m):
            for s in range(self.m):
                coeff = self.weights * probs[:,r] * ((r == s)-probs[:,s])
                hess[r*self.p:(r+1)*self.p,s*self.p:(s+1)*self.p] = q.T @ (coeff[:,None]*q)
        return (hess+hess.T)/2


def _complete_problem(x, n_samples, basis_func):
    x = np.asarray(x,dtype=float)
    c = _counts(n_samples,len(x))
    return _DELProblem(basis_matrix(x,basis_func), np.repeat(np.arange(len(c)),c),
                       np.ones(len(x)), c)


def _missing_problem(x, x_imp, delta, px_hat, n_samples, basis_func):
    x = np.asarray(x,dtype=float)
    if x.ndim != 1:
        raise ValueError('x must be a vector')
    c = _counts(n_samples,len(x))
    imp = np.asarray(x_imp,dtype=float)
    if imp.ndim == 1:
        imp = imp[:,None]
    delta = np.asarray(delta)
    p = np.asarray(px_hat,dtype=float)
    if delta.shape != x.shape or not np.all(np.isin(delta,[0,1])):
        raise ValueError('delta must be a binary vector matching x')
    if p.shape != x.shape or not np.all(np.isfinite(p)) or np.any((p <= 0)|(p > 1)):
        raise ValueError('px_hat must match x and satisfy 0 < p <= 1')
    if imp.ndim != 2 or imp.shape[0] != len(x) or imp.shape[1] < 1 or not np.all(np.isfinite(imp)):
        raise ValueError('x_imp must be finite with shape (n,) or (n,k)')
    obs = delta.astype(bool)
    if not np.all(np.isfinite(x[obs])):
        raise ValueError('respondents must have finite x values')
    labels = np.repeat(np.arange(len(c)),c)
    a = delta/p
    k = imp.shape[1]
    qobs = basis_matrix(x[obs],basis_func) if obs.any() else None
    qimp = basis_matrix(imp.ravel(),basis_func)
    if qobs is not None and qobs.shape[1] != qimp.shape[1]:
        raise ValueError('basis dimension must be constant')
    q = np.vstack([qobs,qimp]) if qobs is not None else qimp
    lab = np.r_[labels[obs],np.repeat(labels,k)]
    w = np.r_[a[obs],np.repeat((1-a)/k,k)]
    # Drop zero-weight rows; missing x values are never evaluated or multiplied by zero.
    use = w != 0
    return _DELProblem(q[use],lab[use],w[use],c)


def negLDELfn(par, x, n_samples, basis_func=5):
    """Negative DEL, omitting the constant -n log(n)."""
    return _complete_problem(x,n_samples,basis_func).value_grad(par)[0]


def negLDELgr(par, x, n_samples, basis_func=5):
    """Analytic gradient of negLDELfn, in population-major order."""
    return _complete_problem(x,n_samples,basis_func).value_grad(par)[1]


def DEL(par, x, n_samples, basis_func=5):
    """Log DEL (to maximize)."""
    return -negLDELfn(par,x,n_samples,basis_func)


def DEL_gradient(par, x, n_samples, basis_func=5):
    return -negLDELgr(par,x,n_samples,basis_func)


def negLDELfn_miss4(par,x,x_imp,delta,px_hat,n_samples,basis_func=5):
    return _missing_problem(x,x_imp,delta,px_hat,n_samples,basis_func).value_grad(par)[0]


def negLDELgr_miss4(par,x,x_imp,delta,px_hat,n_samples,basis_func=5):
    return _missing_problem(x,x_imp,delta,px_hat,n_samples,basis_func).value_grad(par)[1]


@dataclass
class DRMFit:
    mele: np.ndarray
    theta: np.ndarray
    negldl: float
    success: bool
    message: str
    nit: int
    gradient_norm: float
    scaled_gradient_norm: float
    min_scaled_hessian_eigenvalue: float
    info_mat: np.ndarray
    n_samples: np.ndarray
    basis_func: object
    p_est: object = None
    cdf_est: object = None
    optimizer_success: bool = False

    def __getitem__(self, key):
        return getattr(self,key)


def _fit(problem, basis_func, par_init, maxiter, gtol, warn, compute_distribution, support):
    q, m, p, n = problem.q, problem.m, problem.p, problem.n
    # Q*T is orthonormal up to sqrt(N). This invertible basis change avoids
    # the severe collinearity of [1,y,y^2]. Returned parameters use original Q.
    _, r = np.linalg.qr(q,mode='reduced')
    if r.shape != (p,p) or np.linalg.matrix_rank(r) < p:
        raise ValueError('basis matrix is rank deficient')
    t = np.linalg.solve(r/np.sqrt(len(q)),np.eye(p))
    qs = q @ t
    initial = np.zeros(m*p) if par_init is None else np.asarray(par_init,dtype=float)
    if initial.shape != (m*p,) or not np.all(np.isfinite(initial)):
        raise ValueError('par_init has wrong shape or contains nonfinite values')
    initial = np.linalg.solve(t,initial.reshape(m,p).T).T.ravel()

    def fg(b):
        val,grad = problem.value_grad(b,qs)
        return val/n,grad/n

    opt = minimize(fg,initial,jac=True,method='BFGS',options={'maxiter':maxiter,'gtol':gtol})
    beta = opt.x.reshape(m,p)
    mele = (beta @ t.T).ravel()
    value,grad = problem.value_grad(mele)
    scaled_grad = fg(opt.x)[1]
    hs = problem.hessian(opt.x,qs)/n
    mineig = float(np.linalg.eigvalsh(hs).min()) if np.all(np.isfinite(hs)) else float('nan')
    score = float(np.max(np.abs(scaled_grad)))
    # A tiny score can be reached despite a BFGS precision-loss status. Retain
    # optimizer_success separately; accept only a finite local minimum candidate.
    success = bool(np.isfinite(value) and np.all(np.isfinite(mele)) and
                   score <= max(10*gtol,1e-7) and mineig > 1e-10)
    message = str(opt.message)
    if mineig <= 1e-10:
        message += '; curvature is not strictly positive at the candidate'
    if score > max(10*gtol,1e-7):
        message += '; score tolerance not met'
    fit = DRMFit(mele,np.vstack([np.zeros(p),mele.reshape(m,p)]),value,success,message,
                 int(opt.nit),float(np.max(np.abs(grad))),score,mineig,
                 problem.hessian(mele),problem.counts.copy(),basis_func,
                 optimizer_success=bool(opt.success))
    if not success and warn:
        warnings.warn('DRM fit failed numerical checks: '+message,RuntimeWarning,stacklevel=3)
    if compute_distribution:
        fit.p_est = pEST_func(fit.mele,support,fit.n_samples,basis_func)
        fit.cdf_est = cdfEST_func(fit.p_est)
    return fit


def drmdel(x,n_samples,basis_func=5,*,par_init=None,maxiter=10000,gtol=1e-8,
           warn=True,compute_distribution=True):
    """Compute the ordinary DRM MELE by minimizing negative DEL.

    x is concatenated in population order. basis_func callables include 1,
    unlike the original drmdel package's callable h (which omits the intercept).
    info_mat is the observed DEL Hessian, not a general covariance estimator.
    """
    problem = _complete_problem(x,n_samples,basis_func)
    return _fit(problem,basis_func,par_init,maxiter,gtol,warn,compute_distribution,np.asarray(x))


def eldrm_miss(x,x_imp,delta,px_hat,n_samples,basis_func=5,*,par_init=None,
               maxiter=10000,gtol=1e-8,warn=True,compute_distribution=True):
    """Fit the miss4 augmented objective.

    For k imputations, average *likelihood contributions* across imputations.
    Do not average imputed y before evaluating a nonlinear basis/likelihood.
    success certifies score/curvature checks at a local candidate, not a global
    MELE or a theoretical double-robustness property of the implementation.
    Optional distribution output follows the R completed-support plug-in using
    the first imputation; it is not an augmented distribution estimator.
    """
    problem = _missing_problem(x,x_imp,delta,px_hat,n_samples,basis_func)
    imp = np.asarray(x_imp,dtype=float)
    first = imp if imp.ndim == 1 else imp[:,0]
    support = np.where(np.asarray(delta)==1,np.asarray(x),first)
    return _fit(problem,basis_func,par_init,maxiter,gtol,warn,compute_distribution,support)


def pEST_func(mele,x,n_samples,basis_func=5):
    """Positive normalized completed-support masses, matching active R pEST.

    For ordinary DEL converged fits, the unnormalized masses also sum to one
    within numerical tolerance. For augmented fits this is only a plug-in.
    """
    import pandas as pd
    problem = _complete_problem(x,n_samples,basis_func)
    eta,log_s,_ = problem._state(mele)
    logmass = eta-log_s[:,None]-np.log(problem.n)
    masses = np.exp(logmass-logsumexp(logmass,axis=0))
    return pd.DataFrame({'k':np.repeat(np.arange(len(n_samples)),len(x)),
                         'x':np.tile(x,len(n_samples)), 'p_est':masses.T.ravel()})


def cdfEST_func(p_est):
    """Right-continuous CDF; aggregate duplicate support values."""
    out = p_est.groupby(['k','x'],as_index=False,sort=True)['p_est'].sum()
    out['cdf_est'] = out.groupby('k')['p_est'].cumsum()
    return out[['k','x','cdf_est']]


def quantileEST_func(k,p,cdf_est):
    """Generalized inverse inf{x:F_k(x)>=p}, with endpoints at support bounds."""
    ks,ps = np.broadcast_arrays(np.asarray(k),np.asarray(p,dtype=float))
    if np.any((ps<0)|(ps>1)) or not np.all(np.isfinite(ps)):
        raise ValueError('p must be in [0,1]')
    values = []
    for pop,prob in zip(ks.ravel(),ps.ravel()):
        f = cdf_est[cdf_est.k == pop].sort_values('x')
        if f.empty:
            raise ValueError('population not present in cdf_est')
        pos = min(np.searchsorted(f.cdf_est.to_numpy(),prob,side='left'),len(f)-1)
        values.append(f.x.iloc[pos])
    return np.asarray(values).reshape(ks.shape)
