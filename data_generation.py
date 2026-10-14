"""Normal-on-normal DRM and MAR response process from nn_5_m.R."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit


def normal_drm_parameters(means,variances):
    """Exact log(f_r/f_0) coefficients for Q(y)=(1,y,y^2)."""
    mu,v = np.asarray(means,dtype=float),np.asarray(variances,dtype=float)
    if mu.ndim != 1 or v.shape != mu.shape or len(mu)<2 or np.any(v<=0):
        raise ValueError('matching means and positive variances for >=2 populations required')
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(v)):
        raise ValueError('means and variances must be finite')
    return np.column_stack([0.5*np.log(v[0]/v[1:])+mu[0]**2/(2*v[0])-mu[1:]**2/(2*v[1:]),
                            mu[1:]/v[1:]-mu[0]/v[0],
                            1/(2*v[0])-1/(2*v[1:])]).ravel()


@dataclass
class GeneratedData:
    x_true: np.ndarray
    x_false: np.ndarray
    y_full: np.ndarray
    y_obs: np.ndarray
    delta: np.ndarray
    propensity_true: np.ndarray
    population: np.ndarray
    n_samples: np.ndarray
    marginal_means: np.ndarray
    marginal_variances: np.ndarray
    theta_true: np.ndarray

    @property
    def slices(self):
        edges = np.r_[0,np.cumsum(self.n_samples)]
        return [slice(a,b) for a,b in zip(edges[:-1],edges[1:])]

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame({'population':self.population,'x_true':self.x_true,
            'x_false':self.x_false,'y_full':self.y_full,'y_obs':self.y_obs,
            'delta':self.delta,'p_true':self.propensity_true})


def generate_missing_data(n_samples=(1000,1000,1000,1000),*,seed=None,rng=None,
                          x_means=(1,1,1.5,1.1),x_sds=(1,1.2,1.4,1.5),
                          intercepts=(10,9,10,11),slopes=(1,1.5,1.2,1.1),
                          error_sd=1.2,response_floor=0.1,response_location=1.0,
                          false_x_mean=5.0,false_x_sd=1.5):
    """Generate independent populations, then make Y missing at random.

    X_r ~ N(x_means[r],x_sds[r]^2), Y|X,r ~ N(a_r+b_r X,error_sd^2).
    P(delta=1|X,Y,r)=floor+(1-floor)*expit(X-location).
    X_false is independent noise, for the R misspecification experiments.
    y_full is retained only for oracle evaluation, never imputation/propensity.
    NumPy and R seeds do not produce identical random datasets.
    """
    if seed is not None and rng is not None:
        raise ValueError('provide seed or rng, not both')
    rng = np.random.default_rng(seed) if rng is None else rng
    ns = np.asarray(n_samples)
    if ns.ndim != 1 or len(ns)<2 or np.any(ns<=0) or np.any(ns!=np.floor(ns)):
        raise ValueError('n_samples must contain positive integer counts')
    ns = ns.astype(int)
    pars = [np.asarray(v,dtype=float) for v in (x_means,x_sds,intercepts,slopes)]
    if any(v.shape != ns.shape or not np.all(np.isfinite(v)) for v in pars):
        raise ValueError('one finite generating parameter per population is required')
    xm,xsd,a,b = pars
    if np.any(xsd<=0) or error_sd<=0 or false_x_sd<=0 or not 0<=response_floor<=1:
        raise ValueError('standard deviations must be positive and floor in [0,1]')
    pop = np.repeat(np.arange(len(ns)),ns)
    x = np.concatenate([rng.normal(m,s,n) for n,m,s in zip(ns,xm,xsd)])
    y = a[pop]+b[pop]*x+rng.normal(0,error_sd,len(x))
    p = response_floor+(1-response_floor)*expit(x-response_location)
    delta = rng.binomial(1,p)
    yobs = np.where(delta==1,y,np.nan)
    xf = rng.normal(false_x_mean,false_x_sd,len(x))
    means = a+b*xm
    variances = b*b*xsd*xsd+error_sd**2
    return GeneratedData(x,xf,y,yobs,delta,p,pop,ns,means,variances,
                          normal_drm_parameters(means,variances))
