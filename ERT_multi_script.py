import os
# These MUST be set before numpy/scipy are imported
threadset = "1"
os.environ["OMP_NUM_THREADS"] = threadset
os.environ["MKL_NUM_THREADS"] = threadset
os.environ["OPENBLAS_NUM_THREADS"] = threadset
os.environ["VECLIB_MAXIMUM_THREADS"] = threadset
os.environ["NUMEXPR_NUM_THREADS"] = threadset
import h5py
import numpy as np
import multiprocessing as mp
from functools import partial
#import matplotlib.pyplot as plt
from scipy.sparse import diags, bmat, issparse, block_diag, csr_matrix
from scipy.sparse.linalg import norm, expm_multiply
from scipy.linalg import expm
from scipy.interpolate import interp1d

from config import config
from ERT_h4 import compBasis, HMunichS, HMunichO, sand1, runConfigSetup, κ, compBasisDUMB, block4, ObsEnsH
from ERT_h4 import dV, Bjk3050 , Bjk50100, Bjk010, getC0, getC1, fullHT, dag, sandH, sandX
from ERT_h4 import VsR, C0_pref, C0_struct, C1_pref, C1_struct, Bjork, Lmom, VoR
from ERT_h4 import TAMU_VsR, TAMU_VoR, HTAMUS, HTAMUO, WLC_VsR, WLC_VoR, HWLCS, HWLCO, V_WLC, HWLC, ERTens_1T

# --- Configuration Setup ---
conf = config("params.txt")
conf['NPts'] = 100
conf['delta'] = 1
conf['a0fb'] = 1/1.334
conf['N_Workers'] = 200  # Set this to 64 if running on the full quark node
conf['TMin'] = 0.15

conf.echoParams()
runConfigSetup(conf)

hbarc = conf['hbarc']
N = conf['NPts']
rv = conf['rv']
z = np.linspace(0,0,N)

# --- Physics Functions ---
def TEvo2(t):
    if t > 0.6/hbarc:
        return Bjk50100(t)
    return 0

def TEvo1(t):
    if t > 0.6/hbarc:
        return Bjk3050(t)
    return 0

def TEvo_i(t, T0i):
    if t > 0.6/hbarc:
        return Bjork(t, 0.6/hbarc, T0i)
    return 0

TEvoC = TEvo2

def sandHr(c,L,R): 
    return dag(np.power(rv,2)*L)@R

def sandHB(c,L,R):
    return np.sum([sandHr(c,L[i*c['NPts']:(i+1)*c['NPts']],R[i*c['NPts']:(i+1)*c['NPts']]) for i in range(4)])

def ensTRr(c,psiL):
    return np.sum(np.array([sandHB(c,psi,psi) for psi in psiL]))

def get_Eigs(c, HF):
    Hs0 = HF(c, 0, 0)
    eigenvalues, eigenvectors = np.linalg.eig(Hs0)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    f0 = np.asarray(eigenvectors[:,0]).flatten()/np.power(c['rv'],0)
    f0 = f0/np.sqrt(sandHr(conf,f0,f0))
    f1 = np.asarray(eigenvectors[:,1]).flatten()/np.power(c['rv'],0)
    f1 = f1/np.sqrt(sandHr(conf,f1,f1))
    return f0, f1

def get_adaptive_dt(T_array, dt_ref, T_ref, power=3.0):
    T = np.array(T_array)
    dt = dt_ref * (T_ref / np.maximum(T, 0.01))**power
    dt_min = 0.001 * dt_ref
    dt_max = 20.0 * dt_ref
    return np.clip(dt, dt_min, dt_max)

def absorbing_boundary(rv, RMax, strength=10.0, width=5.0):
    return -1j * strength * np.exp((rv - RMax) / (width/5))

#tC = 0.005
tC = 0.00025

def runERT_1T(c, psiLi, tev, t0=0.0/hbarc, tF=10.0/hbarc):
    f0, f1 = get_Eigs(c, HMunichS)
    TevF = tev
    TK = c['Tkin']
    V_abs = absorbing_boundary(c['rv'], c['RMax'])
    Hs0 = TK + dV(VsR(c, c['rv'], 0) + Lmom(c,0) + V_abs)
    Hs1 = TK + dV(VsR(c, c['rv'], 0) + Lmom(c,1) + V_abs)
    Ho0 = TK + dV(VoR(c, c['rv'], 0) + Lmom(c,0) + V_abs)
    Ho1 = TK + dV(VoR(c, c['rv'], 0) + Lmom(c,1) + V_abs)
    H_all = block_diag((csr_matrix(Hs0), csr_matrix(Hs1), csr_matrix(Ho0), csr_matrix(Ho1)))
    HMunich = lambda t: H_all

    C0T, C1T = getC0(c), getC1(c)
    botAkL = [lambda t: C0T(TevF(t)), lambda t: C1T(TevF(t))]
    P1S = np.concatenate([f0,z,z,z])
    P2S = np.concatenate([f1,z,z,z])

    Brv = np.concatenate([c['rv'],c['rv'],c['rv'],c['rv']])
    botERT = ERTens_1T(c,HMunich, botAkL, get_adaptive_dt(0, tC/hbarc, 0.35) , psiLi, R=40, init_t=t0)
    proj1S_s = [ObsEnsH(c, botERT.psiL*Brv ,P1S*Brv)]
    proj2S_s = [ObsEnsH(c, botERT.psiL*Brv ,P2S*Brv)]
    psiLR = [botERT.psiL]
    tx = [0.0/hbarc]
    nit = 0
    Trdef = [ensTRr(c,botERT.psiL)]
    
    for ti in range(200000):
        if botERT.t > tF:
            break
        nit+=1
        botERT.dt = get_adaptive_dt(TevF(botERT.t), tC/hbarc, 0.35) 
        botERT.step()
        
        if TevF(botERT.t)<c['TMin'] and botERT.t>(0.6/hbarc):
            break
            
        psiLR.append(botERT.psiL)
        Trdef.append(ensTRr(c,botERT.psiL))
        tx.append(botERT.t)

    for i in range(len(psiLR)-1):
        proj1S_s.append(ObsEnsH(c, psiLR[i+1]*Brv ,P1S*Brv))
        proj2S_s.append(ObsEnsH(c, psiLR[i+1]*Brv ,P2S*Brv))
        
    return tx, None, Trdef, proj1S_s, proj2S_s

# --- Parallel Worker Functions ---
def worker_task(task_tuple, config_obj, initial_psi, states):
    idx, tev_func = task_tuple
    f0, f1 = states
    pid = os.getpid()
    print(f"Task {idx} started by PID: {pid}")
    tx, _, _, proj1S, proj2S = runERT_1T(config_obj, initial_psi, tev_func)
    return idx, np.array(tx), np.array(proj1S), np.array(proj2S)

# GLOBAL WRAPPER: Placed here so Multiprocessing can successfully pickle it
def global_worker_wrapper(task_dict, config_obj, initial_psi, states):
    idx_label = (task_dict['class_name'], task_dict['traj_key'])
    return worker_task((idx_label, task_dict['t_func']), config_obj, initial_psi, states)



def get_trajectory_interp(h5_path, class_name, traj_key):
    """Navigates through the centrality class folder to find the trajectory."""
    with h5py.File(h5_path, 'r') as h5f:
        data = h5f[class_name][traj_key][()]
        tau_vals = data[:, 0]
        temp_vals = data[:, 3]
        
    return interp1d(tau_vals/conf['hbarc'], temp_vals, kind='linear', 
                    bounds_error=False, fill_value=(0, 0))

def worker_unpack(args):
    # This just mimics what starmap does automatically
    return global_worker_wrapper(*args)

# --- Main Execution ---
def main():
    #Update this to whatever you named the output from the Trajectory Sampler
    h5_filename = "/scratch/jboyd29/ipglasma/OO_coll_run_5K/upsilon_trajectories_multi_class.h5" 
    
    # 1. Prepare Initial Wavefunctions and Eigs BEFORE packaging tasks
    f0, f1 = get_Eigs(conf, HMunichS)
    
    def get_init_WF():
        sqI = np.zeros(N)
        sqI[0] = 1
        return sqI/np.sqrt(sandHr(conf, sqI, sqI))
    
    init_WF = get_init_WF()
    z_arr = np.zeros_like(init_WF)
    s_part = np.concatenate([init_WF, z_arr, z_arr, z_arr])
    o_part = np.concatenate([z_arr, z_arr, init_WF, z_arr])
    
    a1 = 1 / (1 + (conf['delta'] / 0.3))
    b1 = 1 - a1
    psiLi = np.array([s_part * np.sqrt(a1), o_part * np.sqrt(b1)]) 

    # 2. Package tasks by iterating over all centrality classes
    tasks = []
    with h5py.File(h5_filename, 'r') as h5f:
        class_groups = list(h5f.keys())
        print(f"Found centrality classes: {class_groups}")

        for class_name in class_groups:
            # Sort traj_0, traj_1, etc.
            traj_keys = sorted(h5f[class_name].keys(), 
                               key=lambda x: int(x.split('_')[1]))
            
            for key in traj_keys:
                t_func = get_trajectory_interp(h5_filename, class_name, key)
                
                # Bundle everything into a tuple for starmap
                tasks.append((
                    {'class_name': class_name, 'traj_key': key, 't_func': t_func},
                    conf, 
                    psiLi, 
                    (f0, f1)
                ))

    # 3. Parallel Execution using imap_unordered
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
        
    print(f"Starting parallel pool with {len(tasks)} trajectories...")
    
    # Optional: Clear the file at the start of a new run
    with open("part_compl.out", "w") as f:
        f.write(f"Starting run: {len(tasks)} total jobs\n")

    results = []
    with mp.Pool(processes=conf['N_Workers']) as pool:
        # imap_unordered yields results as soon as each individual worker finishes
        for res in pool.imap_unordered(worker_unpack, tasks):
            results.append(res)
            
            # This is your progress tracker
            with open("part_compl.out", "a") as f:
                f.write(f"Job {len(results)} of {len(tasks)} completed\n")
            
            # Keep a console counter too so you don't get bored
            print(f"Progress: {len(results)}/{len(tasks)} tasks finished", end='\r')

    print("\nAll tasks completed. Moving to HDF5 update...")
        
    # 4. Resample and Update HDF5 File
    print("Evolution complete. Resampling and writing back to HDF5...")
    with h5py.File(h5_filename, 'a') as h5f:
        for idx_tuple, tx, p1s, p2s in results:
            class_name, key = idx_tuple
            traj_group = h5f[class_name]
            
            # Load original [tau, x, y, T]
            original_data = traj_group[key][()]
            hydro_tau = original_data[:, 0]
            
            # Convert solver time back to fm/c
            solver_tau_fm = tx * conf['hbarc']
            
            f_1s = interp1d(solver_tau_fm, p1s, kind='linear', 
                            bounds_error=False, fill_value=(p1s[0], p1s[-1]))
            f_2s = interp1d(solver_tau_fm, p2s, kind='linear', 
                            bounds_error=False, fill_value=(p2s[0], p2s[-1]))
            
            resampled_1s = f_1s(hydro_tau)
            resampled_2s = f_2s(hydro_tau)
            
            updated_data = np.column_stack([original_data, resampled_1s, resampled_2s])
            
            # Replace the old dataset with the new 6-column one
            del traj_group[key]
            traj_group.create_dataset(
                key, 
                data=updated_data, 
                compression='gzip', 
                compression_opts=4
            )
            
    print(f"Processing complete. {h5_filename} is now fully updated.")

if __name__ == "__main__":
    main()
