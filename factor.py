from mpi4py import MPI
from pyDNMFk.runner import pyDNMFk_Runner
import numpy as np
import os
from scipy.io import loadmat, savemat
from pathlib import Path
import re
import sys
from icecream import ic


def NMFk_mpi_wrapper(fpath, fname, ftype, p_r, p_c, result_path):
    
    runner = pyDNMFk_Runner(itr=100, init='nnsvd', verbose=False,
                            norm='fro', method='mu', precision=np.float32,
                            checkpoint=False, sill_thr=0.8, process="pyDNMFk")
    results = runner.run(grid=[p_r, p_c], fpath=fpath,
                         fname=fname, ftype=ftype, results_path='results/',
                         k_range=[10,30], step_k=5)#, k=20)
    k_opt = results["nopt"]
    
    runner = pyDNMFk_Runner(itr=500, init='nnsvd', verbose=False,
                            norm='fro', method='mu', precision=np.float64,
                            checkpoint=False, sill_thr=0.8, process="pyDNMF")
    results = runner.run(grid=[p_r, p_c], fpath=fpath,
                         fname=fname, ftype=ftype, results_path='results/',
                         k=k_opt)
    
    W = results["W"]
    H = results["H"]
    err = results["err"]
    rank = results["rank"]
    
    W_name = f"""W_{rank}.mat"""
    H_name = f"""H_{rank}.mat"""

    if p_r > 1:
        savemat(os.path.join(result_path, W_name), {'W': W})
        if rank == 0:
            savemat(os.path.join(result_path, H_name), {'H': H})
    elif p_c > 1:
        savemat(os.path.join(result_path, H_name), {'H': H})
        if rank == 0:
            savemat(os.path.join(result_path, W_name), {'W': W})            


def setup_mpi(filepath, nrof_proc):
    X = loadmat(filepath)['X']
    r, c = X.shape
    if r > c:
        p_r = nrof_proc
        p_c = 1
    elif r < c:
        p_r = 1
        p_c = nrof_proc
    return p_r, p_c


def main(filepath='/home/ec2-user/data/malware/X.mat', result_path='/home/ec2-user/data/malware'):
    p = Path(filepath)
    fpath = str(p.parent) + '/'
    fname = p.stem
    ftype = re.sub(r'^\.', '', p.suffix)
    
    nrof_proc = MPI.COMM_WORLD.Get_size()
    p_r, p_c = setup_mpi(filepath, nrof_proc)

    NMFk_mpi_wrapper(fpath=fpath, fname=fname, ftype=ftype, p_r=p_r, p_c=p_c, result_path=result_path)


if __name__ == "__main__":
    main()
