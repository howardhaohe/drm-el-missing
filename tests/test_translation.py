"""Numerical checks beyond syntax: formulas, derivatives, donor laws and seeds."""
import sys
from pathlib import Path
import unittest
import tempfile
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import norm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eldrm import *
from eldrm import _complete_problem,_missing_problem
from algorithms import *
from data_generation import *
from simulation import *


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.y = self.rng.normal(1,0.8,48)
        self.ns = [12]*4
        self.par = self.rng.normal(0,0.1,9)
        self.delta = self.rng.binomial(1,0.7,48)
        self.yobs = np.where(self.delta,self.y,np.nan)
        self.imp = self.rng.normal(1,1,(48,3))
        self.pi = np.full(48,0.7)

    def test_complete_scalar_formula(self):
        theta = np.vstack([np.zeros(3),self.par.reshape(3,3)])
        total = 0
        for j,y in enumerate(self.y):
            q = np.array([1,y,y*y])
            total += np.log(np.sum(np.exp(theta @ q)/4))-theta[j//12] @ q
        self.assertAlmostEqual(total,negLDELfn(self.par,self.y,self.ns),places=12)

    def test_complete_gradient(self):
        finite = approx_derivative(lambda p:negLDELfn(p,self.y,self.ns),self.par).ravel()
        np.testing.assert_allclose(negLDELgr(self.par,self.y,self.ns),finite,rtol=2e-7,atol=2e-7)

    def test_augmented_scalar_formula_and_gradient(self):
        theta = np.vstack([np.zeros(3),self.par.reshape(3,3)])
        def contribution(y,g):
            q = np.array([1,y,y*y])
            return np.log(np.sum(np.exp(theta @ q)/4))-theta[g] @ q
        scalar = 0
        for i in range(48):
            pred = np.mean([contribution(y,i//12) for y in self.imp[i]])
            a = self.delta[i]/self.pi[i]
            scalar += (a*contribution(self.y[i],i//12) if self.delta[i] else 0)+(1-a)*pred
        args = (self.yobs,self.imp,self.delta,self.pi,self.ns)
        self.assertAlmostEqual(scalar,negLDELfn_miss4(self.par,*args),places=12)
        finite = approx_derivative(lambda p:negLDELfn_miss4(p,*args),self.par).ravel()
        np.testing.assert_allclose(negLDELgr_miss4(self.par,*args),finite,rtol=2e-7,atol=2e-7)

    def test_hessians(self):
        for problem in [_complete_problem(self.y,self.ns,5),
                        _missing_problem(self.yobs,self.imp,self.delta,self.pi,self.ns,5)]:
            finite = approx_derivative(lambda p:problem.value_grad(p)[1],self.par)
            np.testing.assert_allclose(problem.hessian(self.par),finite,rtol=2e-7,atol=2e-7)

    def test_no_missing_reduces_to_del(self):
        args = (self.y,self.imp,np.ones(48),self.pi,self.ns)
        # With p<1 augmentation remains active even if every realized delta=1.
        # p=1 is the appropriate algebraic ordinary-DEL reduction.
        args = (self.y,self.imp,np.ones(48),np.ones(48),self.ns)
        self.assertAlmostEqual(negLDELfn(self.par,self.y,self.ns),negLDELfn_miss4(self.par,*args))

    def test_unit_propensity_matches_completed_del(self):
        args = (self.yobs,self.imp[:,0],self.delta,np.ones(48),self.ns)
        completed = complete_data(self.yobs,self.delta,self.imp[:,0])[:,0]
        self.assertAlmostEqual(negLDELfn_miss4(self.par,*args),negLDELfn(self.par,completed,self.ns))
        fit = eldrm_miss(*args,compute_distribution=False)
        ref = drmdel(completed,self.ns,compute_distribution=False)
        self.assertTrue(fit.success and ref.success)
        np.testing.assert_allclose(fit.mele,ref.mele,atol=2e-5,rtol=2e-5)

    def test_generic_population_basis_dimension(self):
        fit = drmdel(self.y,[24,24],basis_func=5)
        self.assertTrue(fit.success)
        self.assertEqual(fit.mele.shape,(3,))
        self.assertEqual(fit.theta.shape,(2,3))
        np.testing.assert_allclose(fit.p_est.groupby('k').p_est.sum(),1)

    def test_basis_callable_and_all_package_codes(self):
        for code in range(1,12):
            q = basis_matrix(np.array([0.2,1,2]),code)
            np.testing.assert_equal(q[:,0],1)
        np.testing.assert_equal(basis_matrix(self.y,5),basis_matrix(self.y,lambda x:[1,x,x*x]))

    def test_kernel_propensity_and_weights(self):
        x = np.array([0,1,2,3.])
        d = np.array([1,0,1,0])
        raw = np.exp(-0.5*(x[:,None]-x[None,:])**2)
        np.testing.assert_allclose(kernel_propensity(x,d),raw @ d/raw.sum(axis=1))
        full = getWeight_kern_func(x,d)
        np.testing.assert_equal(full[:,d==0],0)
        np.testing.assert_allclose(full.sum(axis=1),1)
        np.testing.assert_allclose(full[:,d==1],raw[:,d==1]/raw[:,d==1].sum(axis=1)[:,None])
        w,_ = kernel_donor_weights(x,d,p_hat=np.array([.2,.3,.8,.9]))
        odds = np.array([4,.25])
        ref = raw[:,d==1]*odds
        np.testing.assert_allclose(w,ref/ref.sum(axis=1)[:,None])

    def test_tiny_bandwidth_and_multivariate(self):
        x = np.array([[0,3],[100,4],[1000,5.]])
        w,ids = kernel_donor_weights(x,[1,1,0],h=[.001,.001])
        np.testing.assert_allclose(w.sum(axis=1),1)
        self.assertTrue(np.all(np.isfinite(w)))

    def test_imputation_no_leakage_and_shapes(self):
        x = np.arange(8.)
        d = np.array([1,0,1,0,1,0,1,0])
        y = np.where(d,2+3*x,np.nan)
        donors = y[d==1]
        for fn,kw in [(random_kernel_imputation,{}),(score_imputation,{'c':3}),
                      (cluster_imputation,{'c':2,'n_init':2}),
                      (random_kernel_odds_imputation,{'p_hat':np.full(8,.6)})]:
            result = fn(x,y,d,k=3,rng=23,**kw)
            self.assertEqual(result.shape,(8,3))
            self.assertTrue(np.all(np.isin(result,donors)))
            np.testing.assert_equal(complete_data(y,d,result)[d==1],np.repeat(donors[:,None],3,axis=1))
        np.testing.assert_allclose(deterministic_linear_imputation(x,y,d)[:,0],2+3*x)
        np.testing.assert_allclose(random_linear_imputation(x,y,d,k=2,rng=5),np.repeat((2+3*x)[:,None],2,axis=1))
        self.assertEqual(deterministic_kernel_imputation(x,y,d).shape,(8,1))

    def test_balanced_groups_and_empty_donor_bins(self):
        x = np.arange(11.)
        d = np.zeros(11,dtype=int)
        d[2] = 1
        y = np.where(d,7,np.nan)
        result = score_imputation(x,y,d,c=8,k=2,rng=1)
        np.testing.assert_equal(result,7)
        np.testing.assert_equal(score_donor_ids(x,y,d,c=8,k=2,rng=1),2)
        np.testing.assert_allclose(getWeight_score_func(x,y,d,c=8).sum(axis=1),1)
        p = score_propensity(np.sin(x),np.array([0,1,1,0,1,0,0,1,0,1,0]),c=3)
        self.assertEqual(len(p),11)
        self.assertTrue(np.all((p>=0)&(p<=1)))

    def test_data_truth_and_missingness(self):
        data = generate_missing_data([100]*4,seed=15)
        np.testing.assert_allclose(data.marginal_means,[11,10.5,11.8,12.21])
        np.testing.assert_allclose(data.marginal_variances,[2.44,4.68,4.2624,4.1625])
        self.assertTrue(np.all(np.isnan(data.y_obs[data.delta==0])))
        np.testing.assert_equal(data.y_obs[data.delta==1],data.y_full[data.delta==1])
        for r in range(1,4):
            values = np.linspace(3,20,50)
            ratio = norm.logpdf(values,data.marginal_means[r],np.sqrt(data.marginal_variances[r]))-norm.logpdf(values,data.marginal_means[0],np.sqrt(data.marginal_variances[0]))
            np.testing.assert_allclose(basis_matrix(values) @ data.theta_true.reshape(3,3)[r-1],ratio,atol=1e-12)

    def test_quantiles_at_ties_and_endpoints(self):
        import pandas as pd
        mass = pd.DataFrame({'k':[0]*3,'x':[1.,1.,2.],'p_est':[.25,.25,.5]})
        cdf = cdfEST_func(mass)
        np.testing.assert_equal(quantileEST_func(0,[0,.5,.51,1],cdf),[1,1,2,2])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            kernel_donor_weights([1,2],[0,0])
        with self.assertRaises(ValueError):
            kernel_propensity([1,2],[1,0],h=0)
        with self.assertRaises(ValueError):
            drmdel([1,2],[1,0])
        with self.assertRaises(ValueError):
            eldrm_miss(self.yobs,self.imp,self.delta,np.zeros(48),self.ns)

    def test_parallel_reproducibility_and_checkpoint(self):
        cfg = SimulationConfig(n_samples=(40,)*4,n_replications=3,k=1)
        with tempfile.TemporaryDirectory() as tmp:
            a = run_monte_carlo(cfg,n_jobs=1,output_dir=Path(tmp)/'a',verbose=False)
            b = run_monte_carlo(cfg,n_jobs=2,output_dir=Path(tmp)/'b',verbose=False)
            np.testing.assert_equal(a['estimates'],b['estimates'])
            np.testing.assert_equal(a['success'],b['success'])
            again = run_monte_carlo(cfg,n_jobs=2,output_dir=Path(tmp)/'a',verbose=False)
            np.testing.assert_equal(a['estimates'],again['estimates'])

    def test_summary_rb_re_and_failed_fit_handling(self):
        truth = np.ones(9)
        estimates = np.ones((3,16,9))*2
        estimates[:,2] = 1.5 # full reference
        success = np.ones((3,16),dtype=bool)
        success[2,0] = False
        estimates[2,0] = 1000 # must be excluded and counted
        result = dict(estimates=estimates,success=success,completed=np.ones(3,bool))
        summary = summarize_monte_carlo(result,truth)
        row = summary.iloc[0]
        self.assertEqual(row.RB_percent,100)
        self.assertEqual(row.RE,.25)
        self.assertEqual(row.n_success,2)
        self.assertEqual(row.n_failed,1)
        self.assertEqual(row.n_paired_reference,2)
        row = summarize_monte_carlo(result,np.zeros(9)).iloc[0]
        self.assertTrue(np.isnan(row.RB_percent))


if __name__=='__main__':
    unittest.main()
