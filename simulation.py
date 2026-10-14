"""Importable workers for notebook-safe, reproducible parallel Monte Carlo."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from multiprocessing import get_context
import numpy as np
import pandas as pd
import scipy
from joblib import cpu_count
from threadpoolctl import threadpool_limits
from data_generation import generate_missing_data
from algorithms import (kernel_propensity,score_propensity,kernel_donor_weights,
                         sample_donors,complete_data)
from eldrm import drmdel,eldrm_miss

METHODS = (
    'cc_del','cc_custom','full_del','full_custom',
    'kernel_t_del','kernel_t_aug','kernel_f_del','kernel_f_aug',
    'odds_tt_del','augmented_tt','odds_tf_del','augmented_tf',
    'odds_ft_del','augmented_ft','odds_ff_del','augmented_ff')
PARAMETERS = tuple(f'{term}_{r}' for r in range(1,4) for term in ('alpha','beta_y','beta_y2'))


@dataclass(frozen=True)
class SimulationConfig:
    n_samples: tuple = (1000,1000,1000,1000)
    n_replications: int = 1000
    seed: int = 591
    k: int = 1
    propensity_method: str = 'kernel'
    propensity_classes: int = 20
    propensity_clip: float = 1e-6
    propensity_bandwidth_power: float = -1/5
    imputation_bandwidth_power: float = -1/3
    maxiter: int = 1000
    gtol: float = 1e-8

    def __post_init__(self):
        if len(self.n_samples)!=4 or any(n<2 or int(n)!=n for n in self.n_samples):
            raise ValueError('this comparison uses four populations, each with n>=2')
        if self.n_replications<1 or int(self.n_replications)!=self.n_replications:
            raise ValueError('n_replications must be a positive integer')
        if self.k<1 or int(self.k)!=self.k:
            raise ValueError('k must be a positive integer')
        if self.propensity_method not in ('kernel','score','oracle'):
            raise ValueError('propensity_method must be kernel,score,oracle')
        if not 0 < self.propensity_clip < 0.5:
            raise ValueError('propensity_clip must be between zero and 0.5')


def prepare_analysis(data,config,rng):
    """Estimate nuisance quantities separately within each population.

    Outcomes come only from y_obs. Oracle mode uses true p for t; f remains
    estimated from independent X_false, preserving the misspecification study.
    """
    n = len(data.y_obs)
    pt,pf = np.empty(n),np.empty(n)
    yt,yf = np.empty((n,config.k)),np.empty((n,config.k))
    odds = {key:np.empty((n,config.k)) for key in ('tt','tf','ft','ff')}
    clip_counts = {'t':0,'f':0}
    for sl in data.slices:
        y,d = data.y_obs[sl],data.delta[sl]
        xs = {'t':data.x_true[sl], 'f':data.x_false[sl]}
        prop = {}
        for key,x in xs.items():
            hp = len(x)**config.propensity_bandwidth_power*x.std(ddof=1)
            if config.propensity_method == 'score':
                raw = score_propensity(x,d,c=config.propensity_classes)
            elif config.propensity_method == 'oracle' and key == 't':
                raw = data.propensity_true[sl]
            else:
                raw = kernel_propensity(x,d,h=hp)
            prop[key] = np.clip(raw,config.propensity_clip,1-config.propensity_clip)
            clip_counts[key] += int(np.sum(raw != prop[key]))
        pt[sl],pf[sl] = prop['t'],prop['f']
        for key,x in xs.items():
            hi = len(x)**config.imputation_bandwidth_power*x.std(ddof=1)
            w,ids = kernel_donor_weights(x,d,h=hi)
            imp = y[sample_donors(w,ids,config.k,rng=rng)]
            (yt if key=='t' else yf)[sl] = imp
            # Reweight the existing Gaussian law by donor odds, in logs.
            # Recompute logits if roundoff has underflowed an entry to zero;
            # kernel_donor_weights already implements stable log normalization.
            for pk in ('t','f'):
                wd,jd = kernel_donor_weights(x,d,h=hi,p_hat=prop[pk])
                odds[key+pk][sl] = y[sample_donors(wd,jd,config.k,rng=rng)]
    return {'p_t':pt,'p_f':pf,'y_t':yt,'y_f':yf,'odds':odds,'clip_counts':clip_counts}


def analyze_dataset(data,config=None,*,rng=None):
    """Return fits and nuisance estimates for the R's sixteen output rows.

    Ordinary multiple-imputation DEL averages the k likelihood contributions
    (eldrm_miss with p=1); it is not Rubin pooling. At k=1 it is completed DEL.
    Duplicate R implementation comparisons share one verified Python engine.
    """
    config = SimulationConfig(n_samples=tuple(data.n_samples)) if config is None else config
    rng = np.random.default_rng(config.seed) if rng is None else rng
    nuisance = prepare_analysis(data,config,rng)
    common = dict(basis_func=5,maxiter=config.maxiter,gtol=config.gtol,
                  warn=False,compute_distribution=False)
    fits,errors = {},{}

    def safe(name,fn):
        try:
            fits[name] = fn()
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
            fits[name] = None
            errors[name] = str(error)

    safe('full_del',lambda:drmdel(data.y_full,data.n_samples,**common))
    fits['full_custom'] = fits['full_del']
    counts_obs = np.array([data.delta[sl].sum() for sl in data.slices])
    safe('cc_del',lambda:drmdel(data.y_obs[data.delta==1],counts_obs,**common))
    fits['cc_custom'] = fits['cc_del']

    def completed_fit(imp):
        if config.k==1:
            return drmdel(complete_data(data.y_obs,data.delta,imp)[:,0],data.n_samples,**common)
        return eldrm_miss(data.y_obs,imp,data.delta,np.ones(len(data.delta)),data.n_samples,**common)

    for key in ('t','f'):
        safe('kernel_'+key+'_del',lambda key=key:completed_fit(nuisance['y_'+key]))
        fits['kernel_'+key+'_aug'] = fits['kernel_'+key+'_del']
    for key in ('tt','tf','ft','ff'):
        safe('odds_'+key+'_del',lambda key=key:completed_fit(nuisance['odds'][key]))
        safe('augmented_'+key,lambda key=key:eldrm_miss(data.y_obs,nuisance['y_'+key[0]],
              data.delta,nuisance['p_'+key[1]],data.n_samples,**common))
    return fits,nuisance,errors


def _replicate(index,seed,config):
    rng = np.random.default_rng(seed)
    estimates = np.full((len(METHODS),9),np.nan)
    ok = np.zeros(len(METHODS),dtype=bool)
    scores = np.full(len(METHODS),np.nan)
    mineigs = scores.copy()
    nit = np.zeros(len(METHODS),dtype=int)
    messages = np.full(len(METHODS),'',dtype='<U512')
    missing = np.full(4,np.nan)
    clipped = np.zeros(2,dtype=int)
    try:
        with threadpool_limits(limits=1):
            data = generate_missing_data(config.n_samples,rng=rng)
            missing = np.array([1-data.delta[sl].mean() for sl in data.slices])
            fits,nuisance,errors = analyze_dataset(data,config,rng=rng)
        clipped = np.array([nuisance['clip_counts'][key] for key in ('t','f')])
        for j,name in enumerate(METHODS):
            fit = fits[name]
            if fit is None:
                messages[j] = errors.get(name,'shared fit failed')
                continue
            estimates[j],ok[j],scores[j],mineigs[j],nit[j] = (fit.mele,fit.success,
                fit.scaled_gradient_norm,fit.min_scaled_hessian_eigenvalue,fit.nit)
            messages[j] = fit.message
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
        messages[:] = 'replication error: '+str(error)
    return index,estimates,ok,scores,mineigs,nit,messages,missing,clipped


def run_monte_carlo(config=None,*,n_jobs=None,output_dir='results',batch_size=100,resume=True,verbose=True):
    """Run all requested replications using independent SeedSequence streams.

    Worker functions are module-level for Windows/macOS/notebook compatibility.
    Atomic batch checkpoints permit resume only with exactly matching config.
    Return raw candidate estimates, validity masks, diagnostics and metadata.
    """
    config = SimulationConfig() if config is None else config
    if int(batch_size)!=batch_size or batch_size<1:
        raise ValueError('batch_size must be a positive integer')
    batch_size = int(batch_size)
    n_jobs = min(8,cpu_count()) if n_jobs is None else n_jobs
    if not isinstance(n_jobs,int) or n_jobs == 0:
        raise ValueError('n_jobs must be a nonzero integer')
    if n_jobs < 0:
        n_jobs = max(1,cpu_count()+1+n_jobs)
    directory = Path(output_dir)
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/'monte_carlo.npz'
    config_json = json.dumps(asdict(config),sort_keys=True)
    shape = (config.n_replications,len(METHODS))
    arrays = dict(estimates=np.full(shape+(9,),np.nan),success=np.zeros(shape,dtype=bool),
                  scaled_gradient=np.full(shape,np.nan),min_hessian_eigenvalue=np.full(shape,np.nan),
                  nit=np.zeros(shape,dtype=int),messages=np.full(shape,'',dtype='<U512'),
                  missing_fraction=np.full((shape[0],4),np.nan),clipped_probabilities=np.zeros((shape[0],2),int),
                  completed=np.zeros(shape[0],dtype=bool))
    if path.exists() and resume:
        with np.load(path,allow_pickle=False) as saved:
            if str(saved['config_json']) != config_json:
                raise ValueError('checkpoint config differs; use a new output_dir or resume=False')
            for name in arrays:
                arrays[name] = saved[name].copy()
    seeds = np.random.SeedSequence(config.seed).spawn(config.n_replications)
    pending = np.flatnonzero(~arrays['completed'])
    start_time = time.perf_counter()
    stored_names = ['estimates','success','scaled_gradient','min_hessian_eigenvalue','nit',
                    'messages','missing_fraction','clipped_probabilities']
    # Standard spawn-based processes avoid platform-specific process monitoring
    # and work with importable workers from notebooks and Windows.
    pool_context = (ProcessPoolExecutor(max_workers=n_jobs,mp_context=get_context('spawn'))
                    if n_jobs>1 and len(pending) else nullcontext(None))
    with pool_context as executor:
        for start in range(0,len(pending),batch_size):
            chunk = pending[start:start+batch_size]
            indices = [int(i) for i in chunk]
            if executor is None:
                rows = [_replicate(i,seeds[i],config) for i in indices]
            else:
                rows = executor.map(_replicate,indices,[seeds[i] for i in indices],[config]*len(indices))
            for row in rows:
                idx = row[0]
                for name,value in zip(stored_names,row[1:]):
                    arrays[name][idx] = value
                arrays['completed'][idx] = True
            temp = directory/'monte_carlo.tmp.npz'
            np.savez_compressed(temp,config_json=config_json,methods=np.array(METHODS),
                                parameters=np.array(PARAMETERS),**arrays)
            temp.replace(path)
            if verbose:
                print(f"Completed {arrays['completed'].sum()}/{config.n_replications} "
                      f"({time.perf_counter()-start_time:.1f} s this session)",flush=True)
    metadata = dict(config=asdict(config),methods=list(METHODS),parameters=list(PARAMETERS),
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        n_jobs=n_jobs,elapsed_seconds_this_session=time.perf_counter()-start_time,
        RE_definition='paired MSE(full_del) / MSE(method)',RB_definition='100*(mean-truth)/truth',
        note='Successful local candidates only; failure counts are reported; augmented global optima are not certified.')
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return dict(arrays,config=config,methods=METHODS,parameters=PARAMETERS,metadata=metadata)


def load_monte_carlo(output_dir='results'):
    with np.load(Path(output_dir)/'monte_carlo.npz',allow_pickle=False) as saved:
        result = {name:saved[name].copy() for name in saved.files if name not in ('config_json','methods','parameters')}
        config = json.loads(str(saved['config_json']))
        config['n_samples'] = tuple(config['n_samples'])
    result.update(config=SimulationConfig(**config),methods=METHODS,parameters=PARAMETERS)
    return result


def summarize_monte_carlo(result,theta_true=None,*,reference='full_del'):
    """RB(%), empirical variance, MSE, and paired-reference relative efficiency.

    Mean/bias/MSE are conditional on each method passing numerical checks.
    RE uses only the same successful replications for method AND reference.
    RE>1 means smaller MSE than reference; RE<1 means larger MSE.
    RB is NaN for zero true parameters. Variance uses denominator B_success,
    consistent with the R empirical moment definition; no 5000 divisor.
    """
    if theta_true is None:
        theta_true = generate_missing_data(result['config'].n_samples,seed=0).theta_true
    truth = np.asarray(theta_true,dtype=float)
    estimates = result['estimates']
    if truth.shape != (estimates.shape[2],):
        raise ValueError('theta_true has wrong length')
    valid = result['success'] & result['completed'][:,None] & np.all(np.isfinite(estimates),axis=2)
    r = METHODS.index(reference)
    completed = int(result['completed'].sum())
    rows = []
    for j,name in enumerate(METHODS):
        use = valid[:,j]
        b = int(use.sum())
        sample = estimates[use,j]
        paired = use & valid[:,r]
        n_pair = int(paired.sum())
        for p,label in enumerate(PARAMETERS):
            values = sample[:,p]
            mean = float(values.mean()) if b else np.nan
            bias = mean-truth[p]
            mse = float(np.mean((values-truth[p])**2)) if b else np.nan
            variance = float(np.var(values,ddof=0)) if b else np.nan
            paired_mse = float(np.mean((estimates[paired,j,p]-truth[p])**2)) if n_pair else np.nan
            ref_mse = float(np.mean((estimates[paired,r,p]-truth[p])**2)) if n_pair else np.nan
            re = ref_mse/paired_mse if paired_mse>0 else np.nan
            rows.append(dict(method=name,parameter=label,truth=truth[p],mean=mean,bias=bias,
                RB_percent=100*bias/truth[p] if truth[p]!=0 else np.nan,variance=variance,MSE=mse,
                RE=re,paired_MSE=paired_mse,reference_paired_MSE=ref_mse,n_paired_reference=n_pair,
                n_success=b,n_failed=completed-b,n_completed=completed,
                failure_percent=100*(completed-b)/completed if completed else np.nan,
                MCSE_mean=np.std(values,ddof=1)/np.sqrt(b) if b>1 else np.nan))
    return pd.DataFrame(rows)


def save_summary(result,output_dir='results'):
    directory = Path(output_dir)
    directory.mkdir(parents=True,exist_ok=True)
    summary = summarize_monte_carlo(result)
    summary.to_csv(directory/'summary.csv',index=False)
    for name,column in [('RB','RB_percent'),('RE','RE'),('MSE','MSE')]:
        summary.pivot(index='method',columns='parameter',values=column).reindex(METHODS).to_csv(directory/(name+'.csv'))
    failures = []
    for i,j in np.argwhere(result['completed'][:,None] & ~result['success']):
        failures.append(dict(replication=int(i)+1,method=METHODS[j],message=str(result['messages'][i,j]),
            scaled_gradient=result['scaled_gradient'][i,j],min_hessian_eigenvalue=result['min_hessian_eigenvalue'][i,j]))
    pd.DataFrame(failures,columns=['replication','method','message','scaled_gradient','min_hessian_eigenvalue']).to_csv(directory/'failures.csv',index=False)
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replications',type=int,default=1000)
    parser.add_argument('--n',type=int,default=1000,help='sample size PER population')
    parser.add_argument('--jobs',type=int,default=None)
    parser.add_argument('--seed',type=int,default=591)
    parser.add_argument('--k',type=int,default=1)
    parser.add_argument('--output',default='results/n1000')
    args = parser.parse_args()
    cfg = SimulationConfig(n_samples=(args.n,)*4,n_replications=args.replications,seed=args.seed,k=args.k)
    result = run_monte_carlo(cfg,n_jobs=args.jobs,output_dir=args.output)
    summary = save_summary(result,args.output)
    print(summary.groupby('method',sort=False)[['n_success','n_failed']].first().to_string())
