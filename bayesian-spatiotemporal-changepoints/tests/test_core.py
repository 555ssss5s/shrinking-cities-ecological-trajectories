import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from stcp.model import make_designs, collapsed_log_likelihood, run_sampler
from stcp.data import distance_matrix, knn_weights, load_data
from stcp.diagnostics import diagnostics


def test_collapsed_likelihood_matches_multivariate_normal():
    t=np.arange(1,10,dtype=float)
    X=make_designs(t,np.array([2004,2005]),2000)
    rng=np.random.default_rng(5)
    y=rng.normal(.42,.07,size=(2,len(t)))
    S=rng.normal(size=y.shape)
    mu=np.array([.4,.007,-.004])
    Sigma=np.diag([.03,.003,.002])
    sig2=.005
    lm,V,pr,Xty=collapsed_log_likelihood(y,X,mu,Sigma,.2,S,sig2)
    for city in range(2):
        for state in range(len(X)):
            cov=sig2*np.eye(len(t))+X[state]@Sigma@X[state].T
            expected=multivariate_normal.logpdf(y[city]-.2*S[city],mean=X[state]@mu,cov=cov)
            np.testing.assert_allclose(lm[city,state],expected,rtol=0,atol=1e-10)


def test_weights():
    xy=np.array([[0.,0.],[10.,0.],[0.,15.],[22.,20.]])
    D=distance_matrix(xy,'euclidean')
    W=knn_weights(D,k=2)
    assert W.shape==(4,4)
    np.testing.assert_allclose(W.sum(axis=1),np.ones(4))
    assert (np.diag(W)==0).all()
    assert np.count_nonzero(W)==8


def test_sampler_small_shape():
    rng=np.random.default_rng(1)
    Y=np.clip(.42 + .003*np.arange(15)[None,:] + rng.normal(0,.02,(4,15)),0,1)
    S=rng.normal(size=Y.shape)
    fit=run_sampler(Y,S,np.arange(1,16),start_year=2000,
                    cp_years=np.array([2004,2007]),chains=2,
                    n_iter=24,warmup=8,thin=2,seed=32,progress=False)
    assert fit.theta.shape==(2,8,4,3)
    assert fit.states.shape==(2,8,4)
    assert set(np.unique(fit.states)).issubset({0,1,2})
    assert np.isfinite(fit.mu).all()
    assert np.all(fit.sigma2>0)
    assert np.all((fit.pi>0)&(fit.pi<1))


def test_dataset_inputs(tmp_path):
    d=pd.DataFrame([{'city':c,'year':yr,'rsei':.4+.002*(yr-2000)+j*.01}
                    for j,c in enumerate('ABCD') for yr in range(2000,2022)])
    p=tmp_path/'panel.csv'; d.to_csv(p,index=False)
    co=tmp_path/'coords.csv';pd.DataFrame(dict(city=list('ABCD'),x=[0,2,0,3],y=[0,0,3,3])).to_csv(co,index=False)
    panel=load_data(p,co,k=2)
    assert panel.Ymodel.shape==(4,21)
    assert panel.S.shape==(4,21)
    assert panel.years[0]==2000
    assert panel.t[0]==1


def test_diagnostics():
    rng=np.random.default_rng(4)
    x=rng.normal(size=(4,80))
    dd=diagnostics(x)
    assert .8<dd['rhat']<1.3
    assert 1<=dd['ess_bulk']<=320
