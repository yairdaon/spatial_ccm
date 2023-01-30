import pdb
import warnings
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
from contextlib import redirect_stdout

from pyEDM.CoreEDM import CCM

from teaspoon.parameter_selection.autocorrelation import autoCorrelation_tau
from teaspoon.parameter_selection.FNN_n import FNN_n

from EDM_tools.tools import optimal_tau, find_embedding_dimension, ssa

def ccm(df, xmapper, target, n_sample=200):
    assert df.columns[0] == 'time'
    assert target != 'time'
    assert xmapper != 'time'
    srs = df[xmapper].values
    tau = autoCorrelation_tau(srs)
    E = FNN_n(srs, tau=tau)[1]
    libs = (E+2, srs.size - (E-1)*tau)
    
    dd = CCM(tau=-tau,
             E = E,
             knn = E+1,
             Tp = 0,
             dataFrame=df,
             target=target,
             columns=xmapper,
             libSizes = libs,
             sample = n_sample,
             includeData = True)
    dd = dd['PredictStats1'] ## The direction we are testing!
    dd = dd.assign(target=target, xmapper=xmapper, err=err).drop("N",axis=1)
    dd.to_pickle(f"outputs/{xmapper}:{target}.pickle")
    print(f"xmapper={xmapper} tau={tau} E={E}", flush=True)

        
def main():
    df = pd.read_csv("all_states.csv", parse_dates=True, index_col=0)
    cols = df.columns
    df = ssa(df, M=int(365/7*3)).reset_index()
    n_sample = 200
    it = product(cols, cols)
    Parallel(n_jobs=-1, verbose=1)(delayed(ccm)(df, xmapper=xmapper, target=target, n_sample=n_sample) for xmapper, target in it)
    
    

if __name__ == '__main__':
    try:
        main()
    except:
        import pdb, traceback, sys
        traceback.print_exc()
        tb = sys.exc_info()[2]        
        pdb.post_mortem(tb)



