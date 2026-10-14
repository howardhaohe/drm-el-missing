"""Propensity and imputation methods translated from algorithms.R.

Every imputation routine predicts/samples for ALL rows, as the R routines do.
Use complete_data() to preserve observed outcomes in a completed dataset.
Random routines accept a NumPy Generator or seed and return (n,k) arrays.
Donor IDs are zero-based Python row indices.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logsumexp
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def _xdelta(x,delta):
    x = np.asarray(x,dtype=float)
    if x.ndim == 1:
        x = x[:,None]
    delta = np.asarray(delta)
    if x.ndim != 2 or x.shape[0]<1 or x.shape[1]<1 or not np.all(np.isfinite(x)):
        raise ValueError('x must be a finite vector or n by d matrix')
    if delta.shape != (len(x),) or not np.all(np.isin(delta,[0,1])):
        raise ValueError('delta must be binary and match rows of x')
    return x,delta.astype(bool)


def _xy(x,y,delta):
    x,obs = _xdelta(x,delta)
    y = np.asarray(y,dtype=float)
    if y.shape != (len(x),) or not obs.any() or not np.all(np.isfinite(y[obs])):
        raise ValueError('y must match x, with at least one finite observed donor')
    return x,y,obs


def _positive_int(v,name):
    if not np.isscalar(v) or not np.isfinite(v) or v<1 or int(v)!=v:
        raise ValueError(name+' must be a positive integer')
    return int(v)


def _rng(rng):
    return rng if isinstance(rng,np.random.Generator) else np.random.default_rng(rng)


def _bandwidth(h,d):
    h = np.asarray(h,dtype=float)
    if h.ndim == 0:
        h = np.repeat(h,d)
    if h.shape != (d,) or np.any(h<=0) or not np.all(np.isfinite(h)):
        raise ValueError('h must be positive scalar or a length-d vector')
    return h


def kernel_propensity(x,delta,h=1,*,block_size=256):
    """Gaussian Nadaraya-Watson estimate, including each row's self kernel."""
    x,obs = _xdelta(x,delta)
    z = x/_bandwidth(h,x.shape[1])
    block_size = _positive_int(block_size,'block_size')
    out = np.empty(len(x))
    for start in range(0,len(x),block_size):
        logits = -0.5*cdist(z[start:start+block_size],z,metric='sqeuclidean')
        w = np.exp(logits-logsumexp(logits,axis=1)[:,None])
        out[start:start+block_size] = w @ obs.astype(float)
    return np.clip(out,0,1)  # only remove roundoff outside probability bounds


def logistic_propensity(x,delta):
    """Unpenalized binomial logit MLE; separation raises a clear error.

    This logistic score is used to make bins; the R DGP's floor+logit response
    probability is not exactly a linear-logit GLM.
    """
    x,obs = _xdelta(x,delta)
    if obs.all() or not obs.any():
        return np.full(len(x),float(obs[0]))
    sd = x.std(axis=0,ddof=0)
    sd[sd==0] = 1
    z = np.column_stack([np.ones(len(x)),(x-x.mean(axis=0))/sd])
    target = obs.astype(float)
    def fg(beta):
        eta = z @ beta
        return np.mean(np.logaddexp(0,eta)-target*eta), z.T @ (expit(eta)-target)/len(x)
    opt = minimize(fg,np.zeros(z.shape[1]),jac=True,method='BFGS',
                   options={'maxiter':1000,'gtol':1e-9})
    if np.max(np.abs(fg(opt.x)[1]))>1e-7 or np.max(np.abs(opt.x))>50:
        raise ValueError('logistic MLE did not converge; possible separation')
    return expit(z @ opt.x)


def _groups(scores,c):
    c = _positive_int(c,'c')
    if c>len(scores):
        raise ValueError('c cannot exceed number of rows')
    labels = np.empty(len(scores),dtype=int)
    for g,rows in enumerate(np.array_split(np.argsort(scores,kind='stable'),c)):
        labels[rows] = g
    return labels


def score_propensity(x,delta,c=5):
    """Response fraction in balanced bins of the fitted logistic score.

    Balanced bins work when n/c is not integer. Zero fractions are returned
    explicitly; any probability clipping belongs to the analysis, not here.
    """
    _,obs = _xdelta(x,delta)
    groups = _groups(logistic_propensity(x,delta),c)
    return np.array([obs[groups==g].mean() for g in groups])


def kernel_donor_weights(x,delta,h=1,*,p_hat=None,block_size=256):
    """Return (W, donor_rows), where W is n by n_observed and rows sum to one.

    If p_hat is supplied, multiply each donor kernel by (1-p_j)/p_j, matching
    imp.randKern.dr_func. This changes the donor law; double robustness is not
    asserted by the name alone. Log normalization avoids kernel underflow.
    """
    x,obs = _xdelta(x,delta)
    donors = np.flatnonzero(obs)
    if not len(donors):
        raise ValueError('no observed donors')
    z = x/_bandwidth(h,x.shape[1])
    block_size = _positive_int(block_size,'block_size')
    log_odds = np.zeros(len(donors))
    if p_hat is not None:
        p = np.asarray(p_hat,dtype=float)
        if p.shape != (len(x),) or not np.all(np.isfinite(p)) or np.any((p<=0)|(p>1)):
            raise ValueError('p_hat must match x and satisfy 0<p<=1')
        pd = p[donors]
        with np.errstate(divide='ignore'):
            log_odds = np.log1p(-pd)-np.log(pd)
        if not np.any(np.isfinite(log_odds)):
            raise ValueError('all donor odds are zero')
    w = np.empty((len(x),len(donors)))
    for start in range(0,len(x),block_size):
        logits = -0.5*cdist(z[start:start+block_size],z[donors],metric='sqeuclidean')+log_odds
        w[start:start+block_size] = np.exp(logits-logsumexp(logits,axis=1)[:,None])
    return w,donors


def sample_donors(weights,donors,k=5,*,rng=None):
    """Draw zero-based row IDs from a supplied conditional donor law."""
    k = _positive_int(k,'k')
    w = np.asarray(weights,dtype=float)
    donors = np.asarray(donors,dtype=int)
    if w.ndim!=2 or w.shape[1]!=len(donors) or not len(donors) or np.any(w<0) or not np.all(np.isfinite(w)):
        raise ValueError('invalid donor weights')
    if not np.allclose(w.sum(axis=1),1):
        raise ValueError('donor weights must sum to one in every row')
    cum = w.cumsum(axis=1)
    cum[:,-1] = 1
    uniforms = _rng(rng).random((len(w),k))
    index = np.vstack([np.searchsorted(row,u,side='right') for row,u in zip(cum,uniforms)])
    return donors[index]


def random_kernel_imputation(x,y,delta,h=1,k=5,*,rng=None):
    _,y,_ = _xy(x,y,delta)
    w,ids = kernel_donor_weights(x,delta,h)
    return y[sample_donors(w,ids,k,rng=rng)]


def deterministic_kernel_imputation(x,y,delta,h=1):
    _,y,_ = _xy(x,y,delta)
    w,ids = kernel_donor_weights(x,delta,h)
    return (w @ y[ids])[:,None]


def random_kernel_odds_imputation(x,y,delta,p_hat,h=1,k=5,*,rng=None):
    _,y,_ = _xy(x,y,delta)
    w,ids = kernel_donor_weights(x,delta,h,p_hat=p_hat)
    return y[sample_donors(w,ids,k,rng=rng)]


def _linear_prediction(x,y,obs):
    design = np.column_stack([np.ones(len(x)),x])
    coeff = np.linalg.lstsq(design[obs],y[obs],rcond=None)[0]
    return design @ coeff


def deterministic_linear_imputation(x,y,delta):
    x,y,obs = _xy(x,y,delta)
    return _linear_prediction(x,y,obs)[:,None]


def random_linear_imputation(x,y,delta,k=5,*,rng=None):
    x,y,obs = _xy(x,y,delta)
    pred = _linear_prediction(x,y,obs)
    residual = y[obs]-pred[obs]
    return pred[:,None]+_rng(rng).choice(residual,size=(len(x),_positive_int(k,'k')),replace=True)


def _merge_empty_groups(groups,obs):
    groups = groups.copy()
    levels = sorted(np.unique(groups))
    for g in levels[:-1]:
        if not np.any(obs & (groups==g)):
            groups[groups==g] = g+1
    last = levels[-1]
    if not np.any(obs & (groups==last)):
        occupied = np.unique(groups[obs])
        groups[groups==last] = occupied[-1]
    return groups


def _group_weights(groups,obs):
    ids = np.flatnonzero(obs)
    w = (groups[:,None]==groups[ids][None,:]).astype(float)
    counts = w.sum(axis=1)
    if np.any(counts==0):
        raise ValueError('a group contains no donor')
    return w/counts[:,None],ids


def score_donor_weights(x,y,delta,c=10):
    x,y,obs = _xy(x,y,delta)
    groups = _merge_empty_groups(_groups(_linear_prediction(x,y,obs),c),obs)
    return _group_weights(groups,obs)


def score_imputation(x,y,delta,c=10,k=5,*,rng=None):
    _,y,_ = _xy(x,y,delta)
    w,ids = score_donor_weights(x,y,delta,c)
    return y[sample_donors(w,ids,k,rng=rng)]


def score_donor_ids(x,y,delta,c=10,k=1,*,rng=None):
    w,ids = score_donor_weights(x,y,delta,c)
    return sample_donors(w,ids,k,rng=rng)


def cluster_donor_weights(x,delta,c=10,*,rng=None,n_init=50):
    x,obs = _xdelta(x,delta)
    c = _positive_int(c,'c')
    if len(x)<2 or not obs.any():
        raise ValueError('clustering requires >=2 rows and observed donors')
    sd = x.std(axis=0,ddof=1)
    sd[sd==0] = 1
    z = (x-x.mean(axis=0))/sd
    if c>len(np.unique(z[obs],axis=0)):
        raise ValueError('c cannot exceed distinct observed covariate rows')
    seed = int(_rng(rng).integers(0,2**31-1))
    km = KMeans(n_clusters=c,max_iter=1000,n_init=n_init,random_state=seed).fit(z[obs])
    return _group_weights(km.predict(z),obs)


def cluster_imputation(x,y,delta,c=10,k=5,*,rng=None,n_init=50):
    _,y,_ = _xy(x,y,delta)
    generator = _rng(rng)
    w,ids = cluster_donor_weights(x,delta,c,rng=generator,n_init=n_init)
    return y[sample_donors(w,ids,k,rng=generator)]


def complete_data(y,delta,imputed):
    """Fill only nonrespondents; return (n,k), preserving observed y exactly."""
    y = np.asarray(y,dtype=float)
    d = np.asarray(delta)
    imp = np.asarray(imputed,dtype=float)
    if imp.ndim==1:
        imp = imp[:,None]
    if y.ndim!=1 or d.shape!=y.shape or not np.all(np.isin(d,[0,1])) or imp.ndim!=2 or imp.shape[0]!=len(y):
        raise ValueError('y,delta,imputed dimensions do not match')
    out = np.where(d[:,None]==1,y[:,None],imp)
    if not np.all(np.isfinite(out)):
        raise ValueError('completed outcomes must be finite')
    return out


def getWeight_kern_func(x,delta,h=1):
    w,ids = kernel_donor_weights(x,delta,h)
    full = np.zeros((len(w),len(w)))
    full[:,ids] = w
    return full


def getWeight_score_func(x,y,delta,c=10):
    w,ids = score_donor_weights(x,y,delta,c)
    full = np.zeros((len(w),len(w)))
    full[:,ids] = w
    return full


def getWeight_cluster_func(x,delta,c=10,*,rng=None,n_init=50):
    w,ids = cluster_donor_weights(x,delta,c,rng=rng,n_init=n_init)
    full = np.zeros((len(w),len(w)))
    full[:,ids] = w
    return full


# Familiar R names with dots replaced by underscores. Prefer keyword arguments:
# positional argument order in Python is x,y,delta,h/c,k rather than the R order.
Phat_kern_func = kernel_propensity
Phat_score_func = score_propensity
imp_randKern_func = random_kernel_imputation
imp_detKern_func = deterministic_kernel_imputation
imp_score_func = score_imputation
imp_detLinear_func = deterministic_linear_imputation
imp_randLinear_func = random_linear_imputation
imp_cluster_func = cluster_imputation
imp_randKern_dr_func = random_kernel_odds_imputation
gimp_score_func = score_donor_ids
