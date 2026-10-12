"""Partially collapsed Gibbs sampler for a zero-or-one changepoint model.

Only continuous parameters enter the Gibbs random effects. For each city the
finite no-CP/CP-year state is sampled from its *integrated* conditional
likelihood, analytically integrating theta_i. Theta_i is then drawn given
that state. This is a coherent blocked Gibbs transition, not BIC selection.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from scipy.stats import invwishart


@dataclass
class Fit:
    theta: np.ndarray  # chains, draws, cities, 3
    states: np.ndarray  # 0=no changepoint; 1..S correspond to years
    mu: np.ndarray  # chains, draws, 3
    Sigma: np.ndarray  # chains, draws, 3, 3
    lam: np.ndarray  # chains, draws
    sigma2: np.ndarray  # chains, draws
    pi: np.ndarray  # chains, draws
    years_cp: np.ndarray
    seed: int
    warmup: int
    thinning: int
    iterations: int


def make_designs(t: np.ndarray, cp_years: np.ndarray, start_year: int) -> np.ndarray:
    """Design: state=0 has zero hinge, states 1+ have t-tau hinge."""
    nstates = len(cp_years) + 1
    designs = np.zeros((nstates, len(t), 3), float)
    designs[:, :, 0] = 1.
    designs[:, :, 1] = t
    for s, year in enumerate(cp_years, 1):
        designs[s, :, 2] = np.maximum(0., t - (year - start_year))
    return designs


def collapsed_log_likelihood(Y: np.ndarray, X: np.ndarray, mu: np.ndarray,
                             Sigma: np.ndarray, lam: float, S: np.ndarray,
                             sigma2: float) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Exact Normal marginal likelihood per city/state; integrate theta_i.

    Args: Y [city,time], X [state,time,3], S [city,time].
    Returns: log-likelihood [city,state], V_s [state,3,3],
             h_s=Sigma^-1 mu [3], and X^T(Y-lambda*S) [city,state,3].
    """
    C,T = Y.shape
    G = np.einsum("stp,stq->spq", X, X)
    sign, ldSigma = np.linalg.slogdet(Sigma)
    if sign != 1 or sigma2 <= 0:
        raise ValueError("Invalid covariance or variance")
    Sinv = np.linalg.inv(Sigma)
    A = Sinv[None,:,:] + G/sigma2
    V = np.linalg.inv(A)
    signs, ldA = np.linalg.slogdet(A)
    if not np.all(signs == 1):
        raise ValueError("Conditional precision not positive definite")
    R = Y - lam*S
    Xty = np.einsum("stp,ct->csp", X, R)
    r2 = (R*R).sum(axis=1)[:,None] - 2 * np.einsum("csp,p->cs",Xty,mu) + np.einsum("p,spq,q->s",mu,G,mu)[None,:]
    q = (Xty - np.einsum("spq,q->sp",G,mu)[None,:,:])/sigma2
    corrected = np.einsum("csp,spq,csq->cs", q, V, q)
    logml = -0.5 * (T*np.log(2*np.pi*sigma2) + ldSigma + ldA[None,:] + r2/sigma2 - corrected)
    return logml, V, Sinv@mu, Xty


def one_chain(Y: np.ndarray, S: np.ndarray, X: np.ndarray, cp_years: np.ndarray,
              n_iter: int, warmup: int, thin: int, seed: int,
              m0: np.ndarray, kappa0: float = .01, nu0: float = 6,
              Lambda0: np.ndarray | None = None,
              a0: float = 2., b0: float = .005) -> dict:
    if thin < 1 or not (0 <= warmup < n_iter):
        raise ValueError("Invalid iterations/warmup/thinning")
    rng = np.random.default_rng(seed)
    C,T = Y.shape
    nstates = len(cp_years)+1
    n_kept = len(range(warmup, n_iter, thin))
    if Lambda0 is None:
        Lambda0 = np.diag([.05, .002, .002])
    store = dict(theta=np.empty((n_kept,C,3)), states=np.empty((n_kept,C),dtype=np.int16),
                 mu=np.empty((n_kept,3)), Sigma=np.empty((n_kept,3,3)),
                 lam=np.empty(n_kept), sigma2=np.empty(n_kept), pi=np.empty(n_kept))
    state = np.zeros(C, dtype=np.int16)
    theta = np.zeros((C,3),float)
    theta[:,0] = np.mean(Y,axis=1)
    mu = m0.copy()
    Sigma = np.diag([.02,.0002,.0002])
    lam = 0.0
    sigma2 = .004
    pi = .5
    current = 0
    GG = np.einsum("stp,stq->spq",X,X)
    for it in range(n_iter):
        # Block 1: integrate random coefficients -> categorical state draw.
        logml,V,priorv,Xty = collapsed_log_likelihood(Y,X,mu,Sigma,lam,S,sigma2)
        logprior = np.full(nstates, np.log(np.clip(pi,1e-12,1-1e-12))-np.log(nstates-1))
        logprior[0] = np.log1p(-np.clip(pi,1e-12,1-1e-12))
        logits = logml + logprior[None,:]
        probs = np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
        u = rng.random(C)
        state = (u[:,None]>np.cumsum(probs,axis=1)).sum(axis=1).astype(np.int16)
        state=np.minimum(state,nstates-1)
        # Block 2: sample theta_i | s_i and hyperparameters.
        Vs = V[state]
        means = np.einsum("cpq,cq->cp",Vs,priorv[None,:] + Xty[np.arange(C),state,:]/sigma2)
        L = np.linalg.cholesky(Vs)
        theta = means + np.einsum("cpq,cq->cp",L,rng.standard_normal((C,3)))
        # Block 3: sample lambda and sigma2 (Gaussian inverse-Gamma conjugacy).
        Xsel=X[state]
        eta_no_sp=np.einsum("ctp,cp->ct",Xsel,theta)
        resid_no_sp=Y-eta_no_sp
        lvar=1/(1+np.sum(S*S)/sigma2)
        lmean=lvar*np.sum(S*resid_no_sp)/sigma2
        lam=lmean+np.sqrt(lvar)*rng.normal()
        err=resid_no_sp-lam*S
        sigma2=1/rng.gamma(a0+C*T/2,1/(b0+np.sum(err*err)/2))
        # Block 4: update NIW hyperparameters and class frequency.
        kpost=kappa0+C
        ave=theta.mean(axis=0)
        mp=(kappa0*m0+C*ave)/kpost
        centered=theta-ave
        sep=(ave-m0)[:,None]
        Lam=Lambda0+centered.T@centered+(kappa0*C/kpost)*(sep@sep.T)
        Sigma=invwishart.rvs(df=nu0+C,scale=Lam,random_state=rng)
        mu=mp+np.linalg.cholesky(Sigma/kpost)@rng.standard_normal(3)
        pi=rng.beta(1+np.count_nonzero(state),1+C-np.count_nonzero(state))
        if it>=warmup and (it-warmup)%thin==0:
            store['theta'][current]=theta
            store['states'][current]=state
            store['mu'][current]=mu
            store['Sigma'][current]=Sigma
            store['lam'][current]=lam
            store['sigma2'][current]=sigma2
            store['pi'][current]=pi
            current+=1
    assert current == n_kept
    return store


def run_sampler(Y: np.ndarray, S: np.ndarray, t: np.ndarray, start_year: int,
                cp_years: np.ndarray, chains: int = 4, n_iter: int = 6000,
                warmup: int = 2000, thin: int = 4, seed: int = 20260610,
                progress: bool = True) -> Fit:
    Y=np.asarray(Y,float); S=np.asarray(S,float)
    if Y.ndim != 2 or Y.shape != S.shape:
        raise ValueError("Y and S must match city x time")
    if not np.isfinite(Y).all() or not np.isfinite(S).all():
        raise ValueError("Missing/nonfinite observation or spatial lag")
    if chains<1 or len(cp_years)<1:
        raise ValueError("At least 1 chain and 1 CP candidate required")
    X=make_designs(np.asarray(t,float),np.asarray(cp_years,int),start_year)
    # Fixed seeds for platform-level reproducibility (exact equality across library
    # versions / CPUs is not guaranteed).
    seeds=np.random.SeedSequence(seed).generate_state(chains, dtype=np.uint32)
    m0=np.array([Y.mean(),0.,0.])
    out=[]
    for c in range(chains):
        if progress:
            print(f"Sampling chain {c+1}/{chains}: n_iter={n_iter}, warmup={warmup}, thin={thin}, seed={int(seeds[c])}",flush=True)
        r=one_chain(Y,S,X,np.asarray(cp_years),n_iter,warmup,thin,int(seeds[c]),m0)
        out.append(r)
    stacked={key:np.stack([o[key] for o in out]) for key in out[0]}
    return Fit(**stacked,years_cp=np.asarray(cp_years),seed=seed,warmup=warmup,thinning=thin,iterations=n_iter)
