"""Posterior reports and publication-oriented diagnostics."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .model import Fit, make_designs
from .diagnostics import diagnostics


def quantiles(x):
    return np.quantile(x,[.025,.5,.975]).tolist()


def posterior_reports(fit: Fit, panel, outdir: str | Path, ppc_draws: int = 200) -> dict:
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    theta=fit.theta
    states=fit.states
    C=len(panel.cities)
    Z=states.reshape(-1,C)
    Th=theta.reshape(-1,C,3)
    cpyears=fit.years_cp
    X=make_designs(panel.t,cpyears,int(panel.years[0]))
    n_chain, n_draw=states.shape[:2]
    N=len(Z)
    global_rows=[]
    vars_global={
        'mu_alpha':fit.mu[:,:,0], 'mu_beta':fit.mu[:,:,1], 'mu_delta':fit.mu[:,:,2],
        'lambda_spatial':fit.lam, 'sigma_observation':np.sqrt(fit.sigma2),
        'pi_cp':fit.pi,
        'Sigma_alpha_sd':np.sqrt(fit.Sigma[:,:,0,0]),
        'Sigma_beta_sd':np.sqrt(fit.Sigma[:,:,1,1]),
        'Sigma_delta_sd':np.sqrt(fit.Sigma[:,:,2,2]),
    }
    diagnostic_rows=[]
    for name,a in vars_global.items():
        lo, med, hi=quantiles(a)
        global_rows.append(dict(parameter=name,mean=float(a.mean()),median=med,ci95_low=lo,ci95_high=hi))
        diagnostic_rows.append(dict(parameter=name, **diagnostics(a)))
    pd.DataFrame(global_rows).to_csv(outdir/'global_posterior_summary.csv',index=False,encoding='utf-8-sig')
    years_rows=[];city_rows=[]
    for i,city in enumerate(panel.cities):
        zi=Z[:,i]>0
        pcp=float(zi.mean())
        pstate=np.bincount(Z[:,i],minlength=len(cpyears)+1)/N
        for ind,p in enumerate(pstate):
            years_rows.append(dict(city=city,state='no_change' if ind==0 else str(cpyears[ind-1]),probability=float(p)))
        yc=pstate[1:]
        if zi.sum():
            cond=yc/pcp
            cum=np.cumsum(cond)
            tau_lo=int(cpyears[np.searchsorted(cum,.025)])
            tau_mid=int(cpyears[np.searchsorted(cum,.50)])
            tau_hi=int(cpyears[min(np.searchsorted(cum,.975),len(cpyears)-1)])
            tau_map=int(cpyears[np.argmax(cond)])
            d=Th[zi,i,2]; pre=Th[zi,i,1]; post=pre+d
            ppre=float((pre>0).mean());ppost=float((post>0).mean())
            pdelta=float((d>0).mean())
            # Evaluate rules only after P(CP) evidence gate. The labels are
            # descriptive decision rules, not an extra causal/CP test.
            if pcp<.80:
                category='无明确变点证据'
            elif ppre>=.80 and ppost<=.20:
                category='转折后退化风险'
            elif ppost>=.80 and pdelta<=.10:
                category='转折后改善减速'
            elif ppost>=.80:
                category='转折后改善'
            elif .35<=ppost<=.65:
                category='后期趋势不确定'
            else:
                category='其他转折轨迹'
            post_ci=quantiles(post)
        else:
            tau_lo=tau_mid=tau_hi=tau_map=None
            ppre=ppost=pdelta=float('nan')
            category='无明确变点证据'
            post_ci=[float('nan')]*3
        city_rows.append(dict(city=city,p_changepoint=pcp,p_no_changepoint=1-pcp,
                              cp_year_map_cond_cp=tau_map,cp_year_median_cond_cp=tau_mid,
                              cp_year_ci95_low_cond_cp=tau_lo,cp_year_ci95_high_cond_cp=tau_hi,
                              n_cp_draws=int(zi.sum()),low_cp_draw_count=bool(zi.sum()<50),
                              p_pre_positive_cond_cp=ppre,p_post_positive_cond_cp=ppost,
                              p_delta_positive_cond_cp=pdelta,
                              beta_pre_mean=float(Th[:,i,1].mean()),
                              delta_mean_cond_cp=float(Th[zi,i,2].mean()) if zi.any() else np.nan,
                              beta_post_ci95_low_cond_cp=post_ci[0],
                              beta_post_ci95_high_cond_cp=post_ci[2],
                              trajectory_joint=category))
        # R-hat and ESS of city-level slope/intercept and z-indicator.
        for k,n in enumerate(['alpha','beta','delta']):
            diagnostic_rows.append(dict(parameter=f'{n}[{city}]',**diagnostics(theta[:,:,i,k])))
        diagnostic_rows.append(dict(parameter=f'has_cp[{city}]',**diagnostics((states[:,:,i]>0).astype(float))))
    pd.DataFrame(years_rows).to_csv(outdir/'city_changepoint_year_posterior.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(city_rows).to_csv(outdir/'city_posterior_summary.csv',index=False,encoding='utf-8-sig')
    diag=pd.DataFrame(diagnostic_rows)
    diag.to_csv(outdir/'mcmc_diagnostics_rhat_ess.csv',index=False,encoding='utf-8-sig')

    # Representative publication-ready trace plot for global parameters.
    fig,axes=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
    for ax,(name,a) in zip(axes.ravel(),list(vars_global.items())[:6]):
        for cc in range(n_chain):
            ax.plot(np.arange(n_draw),a[cc],alpha=.6,lw=.65,label=f'chain {cc+1}')
        ax.set_title(name); ax.set_xlabel('Saved draw')
    axes[0,0].legend(fontsize=7)
    fig.savefig(outdir/'trace_global.png',dpi=160)
    plt.close(fig)

    # Posterior predictive checks sample across unthinned retained draws
    # from *all* chains, respecting the state of each sampled city.
    total=n_chain*n_draw
    select=np.unique(np.linspace(0,total-1,min(ppc_draws,total),dtype=int))
    rng=np.random.default_rng(fit.seed+54321)
    samples=np.empty((len(select),*panel.Ymodel.shape),float)
    latent=np.empty_like(samples)
    for row,sample_idx in enumerate(select):
        ch,dv=divmod(int(sample_idx),n_draw)
        mu=np.einsum('ctp,cp->ct',X[states[ch,dv]],theta[ch,dv]) + fit.lam[ch,dv]*panel.S
        latent[row]=mu
        samples[row] = mu + rng.normal(size=mu.shape)*np.sqrt(fit.sigma2[ch,dv])
    # Posterior mean of the latent expectation; PPC intervals include observation noise.
    predicted=latent.mean(axis=0)
    lower=np.quantile(samples,.025,axis=0)
    upper=np.quantile(samples,.975,axis=0)
    observed=panel.Ymodel
    residual=observed-predicted
    rmse=float(np.sqrt(np.mean(residual**2)))
    mae=float(np.mean(np.abs(residual)))
    cover=float(np.mean((observed>=lower)&(observed<=upper)))
    ppc_rows=[]
    for i,city in enumerate(panel.cities):
        for k,yr in enumerate(panel.years[1:]):
            ppc_rows.append(dict(city=city,year=int(yr),observed_rsei=float(observed[i,k]),
                                 predicted_rsei=float(predicted[i,k]),
                                 predictive_ci95_low=float(lower[i,k]),
                                 predictive_ci95_high=float(upper[i,k]),
                                 residual=float(residual[i,k])))
    pd.DataFrame(ppc_rows).to_csv(outdir/'posterior_predictive_city_year.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([dict(rmse_in_sample=rmse,mae_in_sample=mae,
                       predictive_ci95_coverage=cover,ppc_draws_used=len(select))]).to_csv(outdir/'ppc_metrics.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.3,5.5))
    ax.scatter(observed.ravel(),predicted.ravel(),s=7,alpha=.45)
    ax.plot([0,1],[0,1],'k--',lw=1)
    ax.set(xlabel='Observed RSEI',ylabel='Posterior predictive mean RSEI',xlim=(0,1),ylim=(0,1))
    fig.tight_layout();fig.savefig(outdir/'ppc_observed_vs_predicted.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.hist([observed.ravel(),samples[0].ravel()],bins=30,label=['Observed','One posterior predictive replicate'],alpha=.7,density=True)
    ax.legend();ax.set(xlabel='RSEI',ylabel='Density');fig.tight_layout()
    fig.savefig(outdir/'ppc_distribution.png',dpi=160);plt.close(fig)
    # Posterior source arrays with all chains/draws; no fictitious CSV samples.
    np.savez_compressed(outdir/'posterior_samples.npz',theta=theta,states=states,
                        mu=fit.mu,Sigma=fit.Sigma,lambda_spatial=fit.lam,
                        sigma2=fit.sigma2,pi=fit.pi,cp_years=fit.years_cp,
                        cities=np.array(panel.cities,dtype=str))
    summary=dict(n_cities=C,model_years=f'{int(panel.years[1])}-{int(panel.years[-1])}',
                 n_obs=int(observed.size),chains=n_chain,retained_draws_per_chain=n_draw,
                 n_iter=fit.iterations,warmup=fit.warmup,thin=fit.thinning,seed=fit.seed,
                 cp_years=fit.years_cp.tolist(),k=panel.k,distance=panel.distance_method,
                 spatial_lag_center=panel.spatial_mean,spatial_lag_sd=panel.spatial_sd,
                 classification_probability_threshold=0.8,
                 rhat_max_finite=float(diag['rhat'].max()) if diag['rhat'].notna().any() else None,
                 n_rhat_above_1_01=int((diag['rhat']>1.01).sum()),
                 n_diagnostics_unavailable=int(diag['rhat'].isna().sum()),
                 rmse_in_sample=rmse,mae_in_sample=mae,ppc_coverage=cover,
                 warning='Posterior predictive in-sample fit is not out-of-sample predictive validation; convergence must be checked.' )
    (outdir/'run_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(panel.W,index=panel.cities,columns=panel.cities).to_csv(outdir/'W_knn_row_standardized.csv',encoding='utf-8-sig')
    return summary
