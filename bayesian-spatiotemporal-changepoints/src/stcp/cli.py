"""stcp executable, public CLI."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import load_data
from .model import run_sampler
from .report import posterior_reports


def make_parser():
    p=argparse.ArgumentParser(description='Bayesian city RSEI change/no-change hierarchical model')
    p.add_argument('--panel',required=True,help='city/year/rsei long or city/2000/... wide CSV; xlsx also supported')
    p.add_argument('--coords',required=True,help='city/x/y (projected) or city/lon/lat (degrees) CSV/XLSX')
    p.add_argument('--out',default='outputs/joint_cp')
    p.add_argument('--distance',choices=['euclidean','haversine'],default='euclidean')
    p.add_argument('--k',type=int,default=5)
    p.add_argument('--year-start',type=int,default=2000)
    p.add_argument('--year-end',type=int,default=2021)
    p.add_argument('--cp-first',type=int,default=2004)
    p.add_argument('--cp-last',type=int,default=2017)
    p.add_argument('--chains',type=int,default=4)
    p.add_argument('--n-iter',type=int,default=6000)
    p.add_argument('--burn',type=int,default=2000)
    p.add_argument('--thin',type=int,default=4)
    p.add_argument('--seed',type=int,default=20260610)
    p.add_argument('--ppc-draws',type=int,default=200)
    return p


def main(argv=None):
    args=make_parser().parse_args(argv)
    if not (args.year_start+2<=args.cp_first<=args.cp_last<=args.year_end-2):
        raise ValueError('CP candidates must have at least two observations on each side')
    panel=load_data(args.panel,args.coords,k=args.k,method=args.distance,
                    first=args.year_start,last=args.year_end)
    yrs=np.arange(args.cp_first,args.cp_last+1,dtype=int)
    print(f'Loaded {len(panel.cities)} cities x {len(panel.t)} model years; {len(yrs)+1} CP/no-CP states',flush=True)
    fit=run_sampler(panel.Ymodel,panel.S,panel.t,start_year=args.year_start,cp_years=yrs,
                    chains=args.chains,n_iter=args.n_iter,warmup=args.burn,thin=args.thin,seed=args.seed)
    report=posterior_reports(fit,panel,args.out,args.ppc_draws)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('Results:',str(Path(args.out).resolve()))


if __name__=='__main__':
    main()
