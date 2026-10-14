# Density ratio estimation with missing data (Python)

This project translates `algorithms.R`, the active ordinary-DEL and `miss4`
formulations in `eldrm(1).R`, and the normal-on-normal experiment in `nn_5_m.R`.
It implements the parameter-estimation part of `drmdel` 1.3.2, not every
hypothesis-testing or covariance routine in that package. The supplied papers
are background; the supplied executable R formulas determine this translation.

## Files and getting started

| File | Purpose |
|---|---|
| `eldrm.py` | Log DEL, negative DEL, analytic gradients, MELE wrapper, augmented missing-data wrapper, completed-support masses/CDF/quantiles |
| `algorithms.py` | All 13 active propensity, imputation, weight, and donor-index functions from `algorithms.R` |
| `data_generation.py` | Configurable normal-on-normal data and MAR missingness generator; exact DRM parameters |
| `simulation.py` | Importable parallel workers, 16 estimators, checkpoints, RB/RE/MSE and failure reporting |
| `analysis.ipynb` | Executed analysis, method demonstrations, single-data fits, 1,000-replication results and plots |
| `results/n1000/` | Actual simulation output; `summary.csv`, `RB.csv`, `RE.csv`, `MSE.csv`, raw `.npz`, metadata and failure log |
| `tests/test_translation.py` | Formula, derivative, imputation, missingness, metric and parallel reproducibility tests |
| `validation/` | Actual package C-core parity script and numerical validation report |

Clone this repository, open a terminal in its root directory, and run:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab analysis.ipynb
```

If your notebook uses a different Python environment, install dependencies in a
notebook cell using `%pip install -r requirements.txt`, then restart the kernel.

The notebook loads the included 1,000-replication results by default. Set
`FORCE_RERUN = True` in its Monte Carlo cell to perform the simulation again.
Each replication contains **1,000 observations per population, 4,000 total**,
with `k=1` imputation draw per unit, matching the first R sample-size scenario.
The root seed is 591. Larger scenarios can be run explicitly:

```bash
python simulation.py --replications 1000 --n 2000 --jobs 8 --output results/n2000
python simulation.py --replications 1000 --n 5000 --jobs 8 --output results/n5000
```

These two larger scenarios are configurable; their results are not bundled.
The Gaussian kernels take quadratic time in population sample size. The
implementation blocks distance calculations to limit temporary memory, but
conditional donor weights still require an n by n_observed matrix per group.
Reducing worker count can help when using larger groups or many imputations.

Run the numerical tests using:

```bash
python -m unittest discover -s tests -v
```

Parallelism uses `ProcessPoolExecutor` with the `spawn` start method and
importable module-level workers. When calling from a standalone script, place
the call under `if __name__ == '__main__':`; the provided CLI already does this.
Workers limit numerical-library threads to one. Independent NumPy
`SeedSequence` streams are assigned by replication index, so changing the
worker count does not change the simulation draws. Python and R do not
produce the same random numbers from the same integer seed.

## 1. DRM and DEL

For populations r=0,...,m,

\[
dF_r(y)=\exp\{\theta_r^TQ(y)\}\,dF_0(y),\qquad\theta_0=0.
\]

The normal-family basis is Q(y)=(1,y,y²)ᵀ. Concatenate data in population
order, with sample sizes n₀,...,nₘ and ρᵣ=nᵣ/n. Write

\[
s(y;\theta)=\sum_{r=0}^m\rho_r\exp\{\theta_r^TQ(y)\}.
\]

The negative DEL, up to the parameter-independent constant, is

\[
L(\theta)=\sum_{i=1}^n\left[\log s(y_i;\theta)-\theta_{g_i}^TQ(y_i)\right].
\]

Thus `DEL = -negLDELfn`, and `drmdel` minimizes `negLDELfn`. With
πᵢᵣ(θ)=ρᵣ exp{θᵣᵀQ(yᵢ)}/s(yᵢ;θ), the gradient for r>=1 is

\[
\nabla_{\theta_r}L=\sum_i[\pi_{ir}(\theta)-1\{g_i=r\}]Q(y_i).
\]

The (r,s) Hessian block is

\[
\sum_i\pi_{ir}(\theta)[1\{r=s\}-\pi_{is}(\theta)]Q(y_i)Q(y_i)^T.
\]

`eldrm.py` uses log-sum-exp to avoid exponent overflow in the likelihood.
The optimizer uses an invertible QR basis transformation to improve
conditioning, then transforms estimates back to the original basis.
There is no ridge penalty, parameter bound, or silently changed statistical
objective. BFGS starts at zero. The simulation allows 1,000 optimizer
iterations; the public fitting wrappers default to 10,000, as in R.

```python
from eldrm import drmdel, DEL, DEL_gradient
fit = drmdel(y, n_samples=[1000]*4, basis_func=5)
print(fit.mele)       # flattened nonbaseline parameters
print(fit.theta)      # population rows, including theta_0=0
print(fit.success)    # finite values, small score, positive local curvature
```

Both `fit.mele` and `fit['mele']` work. The order is
`alpha_1,beta_y_1,beta_y2_1,alpha_2,...,alpha_3,beta_y_3,beta_y2_3`.
Population count and basis dimension are independent. Integer bases 1--11
match the original package's definitions. A Python callable returns the
**full** Q including 1, e.g. `lambda y: [1,y,y*y]`, matching the supplied
custom R scripts. The package's own R callable interface returns h without
the intercept; prepend 1 when converting that interface.

## 2. Missing-data objective: the active miss4 formulation

Let δᵢ=1 for a respondent, p̂ᵢ its estimated response probability, and
ỹᵢℓ the ℓth imputation, computed for every unit, including respondents.
Define the unit negative DEL contribution

\[
\ell_i(\theta;y)=\log s(y;\theta)-\theta_{g_i}^TQ(y).
\]

The exact active R formula, extended to k imputations, is

\[
L_{aug}(\theta)=\sum_i\left[
\frac{\delta_i}{\hat p_i}\ell_i(\theta;y_i)
+\left(1-\frac{\delta_i}{\hat p_i}\right)
\frac1k\sum_{\ell=1}^k\ell_i(\theta;\tilde y_{i\ell})\right].
\]

Its gradient applies the same weights to each contribution's analytic
gradient. Missing outcomes are never evaluated; there is no `0 * NaN`.

```python
from eldrm import eldrm_miss
fit_aug = eldrm_miss(y_obs, y_tilde, delta, p_hat, [1000]*4)
```

When every p̂ᵢ=1, this is ordinary DEL on observed outcomes plus imputed
nonrespondent outcomes. With k>1, nonlinear likelihood contributions are
averaged, not outcomes or fitted coefficients. This extension is **not**
Rubin multiple-imputation pooling or an uncertainty estimator.

Some coefficients 1-δᵢ/p̂ᵢ are negative. Consequently, the augmented
objective can be nonconvex or unbounded in finite samples. A successful fit
means that the score and strictly positive local curvature checks passed;
it does not prove a global maximum of an empirical likelihood or a
double-robustness theorem. `optimizer_success` retains the original BFGS
status separately; a precision-loss status can pass the final numerical
checks if the score and curvature are satisfactory.

`info_mat` is the observed objective Hessian. Its inverse is not provided as
a missing-data sampling covariance; valid uncertainty estimation would need
the appropriate sandwich or bootstrap analysis, including nuisance fitting.
Optional `p_est` and `cdf_est` outputs for augmented fits follow the active
R **completed-support, positive-normalized plug-in** and use the first
imputation if k>1. They are not an augmented distribution estimator and have
no claimed missing-data robustness property.

Unused exploratory `miss2`/`miss3` formulas are not exported. In particular,
the supplied `miss3` code has malformed indexing and a mismatch between its
objective and derivative. The actual R analysis calls `miss4`.

## 3. Propensity and imputation methods

For Gaussian Kₕ, the kernel response estimate is

\[
\hat p_i=\frac{\sum_j\delta_jK_h(X_j-X_i)}{\sum_jK_h(X_j-X_i)}.
\]

The self kernel is included, as in R. Score propensity estimation fits an
unpenalized logistic model, sorts its fitted probabilities, and assigns each
balanced group its response fraction. The logistic score is a working
model: the generating probability 0.1+0.9 expit(X-1) is not exactly a linear
logit probability. Score estimates can equal zero or one. The simulation
explicitly clips probabilities to [10⁻⁶,1-10⁻⁶] and records clipping counts;
the public estimators themselves return their raw estimates.

Ordinary kernel donor probabilities over responding donors are

\[
W_{ij}=\frac{\delta_jK_h(X_j-X_i)}{\sum_t\delta_tK_h(X_t-X_i)}.
\]

The R `imp.randKern.dr_func` instead multiplies donor kernels by
(1-p̂ⱼ)/p̂ⱼ before normalizing. The Python name explicitly calls this
**odds-weighted kernel imputation**. It is distinct from minimizing the
augmented DEL objective with ordinary imputations.

| R function | Python function or alias |
|---|---|
| `Phat.kern_func` | `kernel_propensity` / `Phat_kern_func` |
| `Phat.score_func` | `score_propensity` / `Phat_score_func` |
| `imp.randKern_func` | `random_kernel_imputation` / `imp_randKern_func` |
| `imp.detKern_func` | `deterministic_kernel_imputation` / `imp_detKern_func` |
| `imp.score_func` | `score_imputation` / `imp_score_func` |
| `imp.detLinear_func` | `deterministic_linear_imputation` / `imp_detLinear_func` |
| `imp.randLinear_func` | `random_linear_imputation` / `imp_randLinear_func` |
| `imp.cluster_func` | `cluster_imputation` / `imp_cluster_func` |
| `imp.randKern.dr_func` | `random_kernel_odds_imputation` / `imp_randKern_dr_func` |
| `getWeight.kern_func` | `getWeight_kern_func` |
| `getWeight.score_func` | `getWeight_score_func` |
| `getWeight.cluster_func` | `getWeight_cluster_func` |
| `gimp.score_func` | `score_donor_ids` / `gimp_score_func` |

Use keywords when moving an R call: Python arguments begin with x,y,delta,
not h/c/k. All imputation methods return n by k arrays (deterministic methods
return n by 1), including for observed units. `complete_data()` only replaces
missing outcomes. Donor index functions return zero-based row indices.
The `getWeight_*` compatibility routines return full n by n matrices with
zero columns for nonrespondents; efficient `*_donor_weights` routines return
n by n_observed matrices plus donor row indices.

Score imputation uses respondent OLS predictions to define groups and uniform
hot-deck draws within groups. Empty donor groups merge forward and the final
empty group merges backward. This logic also applies to score donor weights
and donor-index draws. Cluster imputation standardizes covariates using the
full-sample mean and sample SD (ddof=1), fits k-means to respondent covariates
with 50 starts, and assigns everyone to the nearest center. NumPy/sklearn
random draws and k-means solutions need not equal R's.
Residual imputation adds sampled respondent OLS residuals to predicted
outcomes. It does not estimate model-parameter uncertainty.

## 4. Generating model and exact parameters

Within each population, independently,

\[
X_r\sim N(\mu_{Xr},\sigma_{Xr}^2),\quad
Y_r=a_r+b_rX_r+\varepsilon_r,\quad\varepsilon_r\sim N(0,1.2^2).
\]

| Population | E[X] | SD(X) | a | b | E[Y] | Var(Y) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1.0 | 1.0 | 10 | 1.0 | 11.00 | 2.4400 |
| 1 | 1.0 | 1.2 | 9 | 1.5 | 10.50 | 4.6800 |
| 2 | 1.5 | 1.4 | 10 | 1.2 | 11.80 | 4.2624 |
| 3 | 1.1 | 1.5 | 11 | 1.1 | 12.21 | 4.1625 |

For marginal mean μᵣ and variance vᵣ,

\[
\alpha_r=\tfrac12\log(v_0/v_r)+\mu_0^2/(2v_0)-\mu_r^2/(2v_r),\quad
\beta_{r1}=\mu_r/v_r-\mu_0/v_0,\quad
\beta_{r2}=1/(2v_0)-1/(2v_r).
\]

The generator calculates these rather than hardcoding them. Response obeys

\[
P(\delta=1\mid X,Y,r)=0.1+0.9\operatorname{expit}(X-1).
\]

Thus Y is missing at random conditional on the fully observed X. The false
covariate X_false~N(5,1.5²) is independent noise. The t/f comparisons estimate
conditional imputation and propensity functions using X_true or X_false.
`y_full` is retained only for the oracle full-data benchmark and evaluation;
all nuisance methods receive `y_obs` and responding donors only.

## 5. Sixteen estimator rows

The first character of tt/tf/ft/ff selects the imputation covariate; the second
selects the propensity covariate. t uses X_true and f uses X_false. These are
the same misspecification combinations as the R script, not assertions that
every working estimator is perfectly specified in finite samples.

| R output rows | Python labels | Interpretation |
|---|---|---|
| 1,2 | `cc_del`, `cc_custom` | Complete cases; recompute responding group counts |
| 3,4 | `full_del`, `full_custom` | Full-data oracle |
| 5,6 | `kernel_t_del`, `kernel_t_aug` | Ordinary kernel completion using X_true; augmented p=1 is identical |
| 7,8 | `kernel_f_del`, `kernel_f_aug` | Ordinary kernel completion using X_false; augmented p=1 is identical |
| 9,10 | `odds_tt_del`, `augmented_tt` | Odds-weighted completion versus augmented DEL, t/t |
| 11,12 | `odds_tf_del`, `augmented_tf` | Odds-weighted completion versus augmented DEL, t/f |
| 13,14 | `odds_ft_del`, `augmented_ft` | Odds-weighted completion versus augmented DEL, f/t |
| 15,16 | `odds_ff_del`, `augmented_ff` | Odds-weighted completion versus augmented DEL, f/f |

The R package/custom ordinary-DEL pairs and the p=1 completed/augmented pairs
are mathematically identical. They share one Python fit to avoid duplicate
optimization and retain all 16 row labels for comparison with R.
The final four pairs are different estimators and are optimized separately.

Kernel propensity bandwidths are nᵣ^(-1/5) SD(Xᵣ); imputation bandwidths are
nᵣ^(-1/3) SD(Xᵣ), with sample SD. Everything is fitted separately within
populations. Unequal group sizes use each group's own nᵣ.

## 6. Monte Carlo metrics and failure policy

For successful replications b in a method's validity set Sⱼ,

\[
\overline\theta_j=|S_j|^{-1}\sum_{b\in S_j}\hat\theta_j^{(b)},\qquad
RB_j(\%)=100(\overline\theta_j-\theta)/\theta,
\]

\[
MSE_j=|S_j|^{-1}\sum_{b\in S_j}(\hat\theta_j^{(b)}-\theta)^2.
\]

Metrics are componentwise. RB is undefined (NaN) when a true parameter is
zero. Empirical variance uses denominator |Sⱼ|, matching the R moment
definition. `MCSE_mean` uses the sample SD divided by sqrt(|Sⱼ|).

**RE is defined here as an MSE ratio relative to the full-data estimator**:

\[
RE_j=\frac{MSE_{full,S_j\cap S_{full}}}{MSE_{j,S_j\cap S_{full}}}.
\]

This convention was not specified in the supplied R script (which computes
RB and MSE but no RE). RE is a ratio, not a percentage. RE=1 is equal MSE;
RE<1 is larger MSE than full data; RE>1 is smaller MSE in this experiment.
The paired subset ensures numerator and denominator use the same datasets.
For biased methods, this MSE-based definition is more informative than a
variance-only ratio. `paired_MSE` and `reference_paired_MSE` are exported.

The numerical acceptance policy requires finite estimates/objective, a
normalized score below max(10*gtol,10⁻⁷), and the smallest normalized Hessian
eigenvalue above 10⁻¹⁰ in the transformed basis. Every replication stores
candidate estimates, validity masks, score, curvature, BFGS iterations and
messages. Failures are counted and exported, never silently replaced or
repeated until success. If failures occur, reported RB/MSE are conditional on
successful fits and must be read together with the failure rate.

Checkpoints contain the exact configuration; resume rejects a changed
configuration. Use a new output directory for a different experiment.

## 7. Corrections and intentional differences from R

* The R loop performs 2,000 replications but divides sums by 5,000. The Python
  summaries use the actual number of valid completed replications.
* The incorrect check `ncol(Q)==m` is replaced by checking basis and parameter
  dimensions separately; Q's dimension does not depend on population count.
* Undefined global `n` and `n_sample` in weight helpers are replaced by local
  array dimensions. The `dr`/`df.r` typo in `gimp.score_func` is corrected.
* Score bins use balanced integer partitions when n is not divisible by c.
  Empty donor bins are handled consistently for all score donor routines.
* Missing response values never enter a basis evaluation. Matrix indexing and
  scalar/vector bandwidth validation are explicit.
* Sampling uses donor row indices, avoiding R's special `sample()` behavior
  when a donor pool contains a single positive integer-valued outcome.
* All deterministic imputations return n by 1, rather than mixing R vectors
  and matrices. Python donor IDs are zero-based.
* Kernel calculations use stable log normalization, and the response model
  uses `expit`, avoiding direct exponential overflow/underflow.
* Step CDFs combine duplicate support values. Quantiles use the conventional
  generalized inverse, fixing the supplied R off-by-one/endpoint behavior.
* The unused experimental miss2/miss3 objectives and the package's composite
  tests/asymptotic covariance functions are outside this implementation.
* Random results are reproducible within Python, not bitwise identical to R.

## 8. Validation and provenance

Seventeen tests cover independent scalar likelihood formulas, finite
differences of ordinary/augmented gradients and Hessians, equivalence at
p=1, generic basis/population dimensions, Gaussian donor/propensity formulas,
no missing-outcome leakage, extreme bandwidths, empty donor bins, all
imputation methods, normal density-ratio truth, CDF quantiles, metrics,
and identical parallel/serial draws plus checkpoint resume.

The actual `logDualL` and `logDualLGr` C functions from the supplied
`drmdel_1.3.2.tar.gz` were extracted and compiled unchanged outside R to
validate Python on identical data. Across 20 parameter vectors, maximum
absolute differences were approximately 5.7e-13 for the objective and
2.5e-11 for its gradient. Separately optimized parameter vectors differed
by at most 7.4e-7. See `validation/package_core_validation.json`.

An R runtime is not installed in the execution environment, so an end-to-end
R `optim`/R RNG comparison was not run. The C parity check is a check of the
actual package core; missing-data formulas are checked against independent
scalar expressions and numerical derivatives of the supplied miss4 formula.

All notebook code cells were also executed with IPython's in-process shell.
The hosted environment forbids socket binding, so a separate Jupyter kernel
could not be launched here. Actual computed cell outputs are stored in the
notebook; this limitation affects the local validation transport, not the
notebook's standard Python kernel specification or the Monte Carlo run.
See `validation/notebook_validation.json`.

The tested environment is recorded in `requirements-tested.txt`. General
minimum dependencies appear in `requirements.txt`.

`drmdel` is by Song Cai and distributed under GPL-3.0. The Python DEL/basis
adaptation is distributed under GPL-3.0; the complete license is included in
`LICENSE`. The custom simulation/imputation logic comes from the supplied
R scripts. Attribution does not imply that the original package author has
reviewed or endorsed this translation.
