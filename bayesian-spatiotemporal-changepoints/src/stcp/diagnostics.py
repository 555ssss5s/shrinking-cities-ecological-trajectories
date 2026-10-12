"""Split rank-normalized R-hat and approximate bulk/tail ESS diagnostics.

Formulas follow modern rank-normalized split-chain diagnostics. Designed to
run without ArviZ; report parameters with degenerate chains as NA.
"""
import numpy as np
from scipy.stats import norm, rankdata


def split_chains(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x,float)
    if x.ndim != 2:
        raise ValueError("chains x draws required")
    half=x.shape[1]//2
    if half<4 or x.shape[0]<2:
        return np.empty((0,0))
    return np.concatenate([x[:,:half], x[:,-half:]],axis=0)


def _rhat_core(x: np.ndarray) -> float:
    m,n=x.shape
    W=float(np.mean(x.var(axis=1, ddof=1)))
    if W <= 1e-30:
        return float('nan')
    B=n*np.var(x.mean(axis=1),ddof=1)
    return float(np.sqrt(((n-1)/n*W+B/n)/W))


def _ranknorm(x: np.ndarray) -> np.ndarray:
    flat=x.ravel()
    r=rankdata(flat,method='average')
    z=norm.ppf((r-3/8)/(len(r)+1/4))
    return z.reshape(x.shape)


def rank_rhat(x: np.ndarray) -> float:
    x=split_chains(x)
    if x.size == 0:
        return float('nan')
    r1=_rhat_core(_ranknorm(x))
    r2=_rhat_core(_ranknorm(np.abs(x-np.median(x))))
    if np.isnan(r1) or np.isnan(r2):
        return float('nan')
    return max(r1,r2)


def _ess_core(x: np.ndarray) -> float:
    """FFT autocovariance, initial positive sequence truncation."""
    x=np.asarray(x,float)
    m,n=x.shape
    if m<2 or n<8:
        return float('nan')
    w=float(np.mean(np.var(x,axis=1,ddof=1)))
    if w<=1e-30:
        return float('nan')
    B=n*np.var(x.mean(axis=1),ddof=1)
    varplus=(n-1)/n*w+B/n
    centered=x-x.mean(axis=1,keepdims=True)
    nfft=1 << ((2*n-1).bit_length())
    f=np.fft.rfft(centered,n=nfft,axis=1)
    acov=np.fft.irfft(f*np.conjugate(f),n=nfft,axis=1)[:,:n]/n
    rho=np.empty(n)
    rho[0]=1
    rho[1:]=1-(w-np.mean(acov[:,1:],axis=0))/varplus
    pair_sums=[]
    for lag in range(1,n-1,2):
        v=float(rho[lag]+rho[lag+1])
        if v<=0:
            break
        pair_sums.append(v)
    # initial monotone sequence
    for j in range(1,len(pair_sums)):
        pair_sums[j]=min(pair_sums[j],pair_sums[j-1])
    tau=max(1.,1+2*sum(pair_sums))
    return float(np.clip(m*n/tau,1,m*n))


def diagnostics(x: np.ndarray) -> dict:
    """x is chains x draws; ESS includes split chains, bulk rank and tails."""
    sx=split_chains(x)
    if sx.size==0:
        return dict(rhat=np.nan,ess_bulk=np.nan,ess_tail=np.nan)
    taillo=(sx<=np.quantile(sx,.05)).astype(float)
    tailhi=(sx<=np.quantile(sx,.95)).astype(float)
    return dict(rhat=rank_rhat(x), ess_bulk=_ess_core(_ranknorm(sx)),
                ess_tail=min(_ess_core(tailhi),_ess_core(taillo)))
