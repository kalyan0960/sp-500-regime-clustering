"""GARCH(1,1) estimation and leakage-safe conditional-volatility utilities."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import kurtosis, skew
from statsmodels.stats.diagnostic import acorr_ljungbox

class GarchValidationError(ValueError): pass

def prepare_percent_returns(returns: pd.Series) -> pd.Series:
    """Validate decimal log returns and convert them to percentage points."""
    values=pd.to_numeric(returns,errors='raise').astype(float)
    if values.isna().any() or not np.isfinite(values).all(): raise GarchValidationError('Log_Return must be finite.')
    return (100*values).rename('Log_Return_Pct')

def fit_garch(returns_pct: pd.Series, distribution: str):
    """Fit constant-mean GARCH(1,1) with the requested innovation distribution."""
    if distribution not in {'t','normal'}: raise ValueError('distribution must be t or normal.')
    return arch_model(returns_pct, mean='Constant', vol='GARCH', p=1, q=1, dist=distribution, rescale=False).fit(disp='off')

def parameters(result) -> dict[str, Any]:
    """Extract reproducible GARCH parameters and fit statistics."""
    p=result.params
    alpha=float(p['alpha[1]']); beta=float(p['beta[1]'])
    return {'converged': bool(result.convergence_flag==0),'convergence_flag':int(result.convergence_flag),
            'optimizer_message':str(result.optimization_result.message),'nobs':int(result.nobs),
            'loglikelihood':float(result.loglikelihood),'aic':float(result.aic),'bic':float(result.bic),
            'mu':float(p['mu']),'omega':float(p['omega']),'alpha[1]':alpha,'beta[1]':beta,
            'alpha_plus_beta':alpha+beta,'shock_half_life_days':float(np.log(0.5)/np.log(alpha+beta)) if 0 < alpha+beta < 1 else None,'nu':float(p['nu']) if 'nu' in p.index else None}

def recursive_variance(returns_pct: pd.Series, *, mu: float, omega: float, alpha: float, beta: float, initial_variance: float) -> pd.Series:
    """Generate one-step-ahead variance: date t uses only the date t-1 shock."""
    values=prepare_percent_returns(returns_pct/100).to_numpy() if returns_pct.name!='Log_Return_Pct' else returns_pct.to_numpy(dtype=float)
    if not all(np.isfinite([mu,omega,alpha,beta,initial_variance])) or omega<=0 or alpha<0 or beta<0 or initial_variance<=0: raise GarchValidationError('Invalid GARCH parameters or initialization.')
    variance=np.full(len(values),np.nan); previous_variance=initial_variance; previous_shock=None
    for i,value in enumerate(values):
        if previous_shock is None: variance[i]=previous_variance
        else:
            previous_variance=omega+alpha*previous_shock**2+beta*previous_variance; variance[i]=previous_variance
        previous_shock=value-mu
    return pd.Series(variance,index=returns_pct.index,name='variance')

def residual_diagnostics(result, label: str) -> dict[str, Any]:
    """Summarize standardized-residual distribution and Ljung-Box diagnostics."""
    r=pd.Series(result.std_resid).dropna()
    out={'model':label,'resid_mean':float(r.mean()),'resid_std':float(r.std()),'skewness':float(skew(r)),'excess_kurtosis':float(kurtosis(r))}
    for lag in (10,20):
        lb=acorr_ljungbox(r,lags=[lag],return_df=True).iloc[0]; sq=acorr_ljungbox(r**2,lags=[lag],return_df=True).iloc[0]
        out[f'lb_resid_stat_{lag}']=float(lb['lb_stat']); out[f'lb_resid_p_{lag}']=float(lb['lb_pvalue'])
        out[f'lb_sq_resid_stat_{lag}']=float(sq['lb_stat']); out[f'lb_sq_resid_p_{lag}']=float(sq['lb_pvalue'])
    arch_lm=result.arch_lm_test(lags=10,standardized=True)
    out['arch_lm_lags']=10; out['arch_lm_stat']=float(arch_lm.stat); out['arch_lm_pvalue']=float(arch_lm.pval)
    return out

def build_garch_features(frame: pd.DataFrame, training_end: str) -> tuple[pd.DataFrame, dict[str,Any]]:
    """Add descriptive full-sample and training-fitted leakage-safe GARCH volatility."""
    if 'Log_Return' not in frame: raise GarchValidationError('Log_Return is required.')
    output=frame.copy(); valid=output['Log_Return'].dropna(); pct=prepare_percent_returns(valid)
    train_pct=pct.loc[pct.index<=pd.Timestamp(training_end)]
    student=fit_garch(train_pct,'t'); normal=fit_garch(train_pct,'normal'); full=fit_garch(pct,'t')
    sp=parameters(student); np_=parameters(normal); fp=parameters(full)
    if sp['alpha_plus_beta']>=1: raise GarchValidationError('Training Student-t persistence is at least one; methodological decision required.')
    train_vol=pd.Series(student.conditional_volatility,index=train_pct.index)
    test_pct=pct.loc[pct.index>pd.Timestamp(training_end)]
    test_var=recursive_variance(test_pct,mu=sp['mu'],omega=sp['omega'],alpha=sp['alpha[1]'],beta=sp['beta[1]'],initial_variance=float(train_vol.iloc[-1]**2))
    predictive=pd.concat([train_vol.rename('GARCH_Volatility_TrainFit_Pct'),np.sqrt(test_var).rename('GARCH_Volatility_TrainFit_Pct')]).reindex(output.index)
    descriptive=pd.Series(full.conditional_volatility,index=pct.index).reindex(output.index).rename('GARCH_Volatility_FullSample_Pct')
    for series in (predictive,descriptive):
        if not series.dropna().gt(0).all() or not np.isfinite(series.dropna()).all(): raise GarchValidationError('Volatility must be finite and positive.')
    output['GARCH_Volatility_TrainFit_Pct']=predictive; output['GARCH_Volatility_TrainFit']=predictive/100
    output['GARCH_Volatility_FullSample_Pct']=descriptive; output['GARCH_Volatility_FullSample']=descriptive/100
    return output, {'student':sp,'normal':np_,'full':fp,'diagnostics':[residual_diagnostics(student,'training_student_t'),residual_diagnostics(normal,'training_normal'),residual_diagnostics(full,'full_student_t')]}

def save_outputs(results: dict[str,Any], directory: str|Path) -> None:
    """Save parameter JSON files and diagnostic/model-comparison CSV tables."""
    root=Path(directory); (root/'models').mkdir(parents=True,exist_ok=True); (root/'tables').mkdir(parents=True,exist_ok=True)
    for key,name in [('student','garch_training_student_t_parameters.json'),('normal','garch_training_normal_parameters.json'),('full','garch_full_sample_student_t_parameters.json')]: (root/'models'/name).write_text(json.dumps(results[key],indent=2),encoding='utf-8')
    pd.DataFrame([{'model':k,**results[k]} for k in ('student','normal','full')]).to_csv(root/'tables'/'garch_model_comparison.csv',index=False)
    pd.DataFrame(results['diagnostics']).to_csv(root/'tables'/'garch_residual_diagnostics.csv',index=False)
