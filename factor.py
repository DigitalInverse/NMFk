from mpi4py import MPI
from pyDNMFk.runner import pyDNMFk_Runner
import numpy as np
import os
from scipy.io import savemat
from icecream import ic


def NMFk_mpi_wrapper():

    runner = pyDNMFk_Runner(itr=100, init='nnsvd', verbose=False,
                            norm='fro', method='mu', precision=np.float32,
                            checkpoint=False, sill_thr=0.6, process="pyDNMFk")
    results = runner.run(grid=[4,1], fpath='/home/ec2-user/data/malware/',
                         fname='X', ftype='mat', results_path='results/',
                         k_range=[5,50], step_k=5, k=5)
    k_opt = results["nopt"]
    #
    runner = pyDNMFk_Runner(itr=100, init='nnsvd', verbose=False,
                            norm='fro', method='mu', precision=np.float32,
                            checkpoint=False, sill_thr=0.6, process="pyDNMF")
    results = runner.run(grid=[4,1], fpath='/home/ec2-user/data/malware/',
                         fname='X', ftype='mat', results_path='results/',
                         k=k_opt)
    #
    W = results["W"]
    H = results["H"]
    err = results["err"]
    rank = results["rank"]
    
    #return (W, H, err)
    #ic(rank)
    
    #comm = MPI.Comm.Get_parent()
    #rank = comm.Get_rank()
    data_path='/home/ec2-user/data/malware'
    W_name = f"""W_{rank}.mat"""
    H_name = f"""H_{rank}.mat"""
    savemat(os.path.join(data_path, W_name), {'W': W})
    if rank == 0:
        savemat(os.path.join(data_path, H_name), {'H': H})
    
    #return k_opt
    #ic(W.shape)
    #ic(H.shape)


if __name__ == "__main__":
    
    #W, H, err = NMFk_mpi_wrapper()
    NMFk_mpi_wrapper()
    
    #print("W.shape:", W.shape)
    #print("H.shape:", H.shape)
    #comm = MPI.Comm.Get_parent()
    #rank = comm.Get_rank()
    #data_path='/home/ec2-user/data/malware'
    #W_name = f"""W_{rank}.mat"""
    #H_name = f"""H_{rank}.mat"""
    #savemat(os.path.join(data_path, W_name), {'W': W})
    #savemat(os.path.join(data_path, H_name), {'H': H})
    #ic(W.shape)
    #ic(H.shape)
    #ic(err)
