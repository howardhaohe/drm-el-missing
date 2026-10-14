"""Validate against the actual C DEL/gradient core in supplied drmdel archive.

R is not required. A C compiler (gcc by default) is required for this optional
check. Extracted Song Cai GPL-3.0 functions are compiled unchanged, with only
an allocation-error shim and a minimal ctypes wrapper around them.
Usage: python validation/compare_package_core.py /path/drmdel_1.3.2.tar.gz
"""
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eldrm import negLDELfn,negLDELgr,drmdel
from data_generation import generate_missing_data


def validate(archive):
    with tarfile.open(archive) as package:
        source = package.extractfile('drmdel/src/drmdel.c').read().decode()
    prefix = '#include <stdlib.h>\n#include <math.h>\nvoid errMsg(char *s) {abort();}\n'
    functions = source[source.index('void lp_val('):source.index('void logDualLWrapper(')]
    gradient = source[source.index('void logDualLGr('):source.index('void logDualLGrWrapper(')]
    wrapper = r'''
void h_normal(double x,double *h) {h[0]=x; h[1]=x*x;}
double reference(double *par,double *x,unsigned long *counts,unsigned long m,double *grad) {
  double **pm=malloc(m*sizeof(double*));
  double **xm=malloc((m+1)*sizeof(double*));
  double **gm=malloc(m*sizeof(double*));
  unsigned long n=0,offset=0;
  for(unsigned long r=0;r<m;r++) {pm[r]=par+3*r; gm[r]=grad+3*r;}
  for(unsigned long r=0;r<m+1;r++) {xm[r]=x+offset; offset+=counts[r]; n+=counts[r];}
  double value=logDualL(n,counts,m,2,pm,h_normal,xm);
  logDualLGr(n,counts,m,2,pm,h_normal,xm,gm);
  free(pm); free(xm); free(gm);
  return value;
}
'''
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        (path/'core.c').write_text(prefix+functions+gradient+wrapper)
        subprocess.run(['gcc','-std=c99','-O2','-shared','-fPIC',str(path/'core.c'),'-lm','-o',str(path/'core.so')],check=True)
        lib = ctypes.CDLL(str(path/'core.so'))
        arr = np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        nsarr = np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS')
        lib.reference.argtypes = [arr,arr,nsarr,ctypes.c_ulong,arr]
        lib.reference.restype = ctypes.c_double
        data = generate_missing_data((80,90,100,110),seed=319)
        x = np.ascontiguousarray(data.y_full)
        counts = np.ascontiguousarray(data.n_samples,dtype=np.uint64)
        def core(p):
            p = np.ascontiguousarray(p,dtype=np.float64)
            grad = np.empty(9)
            val = lib.reference(p,x,counts,3,grad)
            return -val,-grad
        rng = np.random.default_rng(923)
        value_errors,gradient_errors = [],[]
        for _ in range(20):
            p = rng.normal(0,.01,9)
            val,grad = core(p)
            pyval = negLDELfn(p,x,counts)
            pygrad = negLDELgr(p,x,counts)
            value_errors.append(abs(val-pyval))
            gradient_errors.append(np.max(np.abs(grad-pygrad)))
            np.testing.assert_allclose(val,pyval,atol=2e-10,rtol=2e-12)
            np.testing.assert_allclose(grad,pygrad,atol=2e-9,rtol=2e-12)
        pyfit = drmdel(x,counts,compute_distribution=False)
        cfit = minimize(core,np.zeros(9),method='BFGS',jac=True,
                        options={'maxiter':10000,'gtol':1e-7})
        np.testing.assert_allclose(cfit.x,pyfit.mele,atol=3e-5,rtol=3e-5)
        np.testing.assert_allclose(cfit.fun,pyfit.negldl,atol=1e-8)
        return dict(reference='Actual drmdel 1.3.2 C logDualL/logDualLGr, compiled without R',
                    random_parameter_checks=20,max_absolute_objective_error=max(value_errors),
                    max_absolute_gradient_error=float(max(gradient_errors)),
                    max_absolute_fitted_parameter_difference=float(np.max(np.abs(cfit.x-pyfit.mele))),
                    fitted_objective_difference=float(abs(cfit.fun-pyfit.negldl)),
                    python_fit_success=pyfit.success,
                    R_runtime_comparison='Not run: R runtime is not installed',status='passed')


if __name__=='__main__':
    report = validate(sys.argv[1])
    target = Path(__file__).with_name('package_core_validation.json')
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
