import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.linalg import expm, block_diag
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, bmat, issparse, csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.interpolate import interp1d
from scipy.integrate import quad, solve_ivp
from scipy.sparse import lil_matrix
from scipy.special import eval_genlaguerre, gammaln
import scipy.integrate as spi
import scipy.special as sp
from math import factorial
import time
from config import config

#a0fb = 1/1.334 # [GeV]  fireball paper a0 for comparison
hbarc = 0.1973

NF = 2
NC = 3
CF = 4/3

# Utility functions
def dag(array):
    return array.T.conj()
def dV(vec): # quick vector to diagonal array
    return diags(vec, offsets=0, format="csr")
def block4(B):
    A = B.toarray()
    return block_diag(A,A,A,A)
### Inner products
#def sandH(c,L,R): # inner produc spherical coords other method
#    return dag(L)@(np.power(c['rv'],2)*R)*c['dr']
#def sandH(c,L,R):
#    return np.trapz(dag(L)*R * np.power(c['rv'],2), c['rv'], axis=0)
def sandH(c,L,R): # inner product flat
    return dag(L)@R#*c['dr']
def sandHB(c,L,R):
    return np.sum([sandH(c,L[i*c['NPts']:(i+1)*c['NPts']],R[i*c['NPts']:(i+1)*c['NPts']]) for i in range(4)])
def sand1(c,L,R): # inner product flat
    return dag(L)@R#*c['dr']
#def sandX(c,L, R): # inner produc spherical coords in exponential space
#    return np.trapz(L*R * np.exp(3*c['xv']), c['xv'], axis=0)*4*np.pi # np.power(xv,3) -> 2 from r^2 metric + 1 from dr=e^x*dx
def sandX(c,L,R):
    return L.conj()@(np.exp(3*c['xv'])*R)


### Config setting

def runConfigSetup(c):
    print('CONFIG SETUP RUNNING')
    m = c['M']/2 # = M*M/(M+M) = reduced mass
    NPts = c['NPts']
    RMax = c['RMax']
    ### Operators ###
    #rv = np.linspace(0, RMax, NPts) # r - vector
    rv = np.linspace(0,c['RMax'],c['NPts']+1)[1:]
    NPts = c['NPts']
    dr = rv[1]-rv[0]
    ##-first derivative

    ddr = diags([np.ones(NPts-1), -np.ones(NPts-1)], [1,-1], shape=(NPts, NPts))
    ddr = ddr.toarray()  # Convert to full array for modification
    ddr[0, 0] = -2  # Forward difference at first point
    ddr[0, 1] = 2
    ddr[0,0:3] = np.array([-3,4,-1])
    ddr = ddr/(2*dr)


    ##-second derivative
    d2dr2 = diags([np.ones(NPts-1), -2*np.ones(NPts), np.ones(NPts-1)], offsets=[1,0,-1],shape=(NPts, NPts),dtype=complex)
    d2dr2 = d2dr2.toarray()
    d2dr2[0, :5] = np.array([1,-2,1,0,0])
    #d2dr2[0, :5] = np.array([0,0,0,0,0])
    #d2dr2[-1, -5:] = np.array([0,0,0,0,0])
    d2dr2 = d2dr2/(dr**2)


    #m = conf['M']/2
    Tkin = (-1/c['M'])*(d2dr2 + (dV(2/rv)@ddr))
    c['m'] = m
    c['dr'] = dr
    c['rv'] = rv
    c['ddr'] = ddr
    c['d2dr2'] = d2dr2
    
    c['Tkin'] = Tkin#csr_matrix(Tkin)
    
def runConfigSetupTK(c, TK):
    print('CONFIG SETUP RUNNING')
    m = c['M']/2 # = M*M/(M+M) = reduced mass
    NPts = c['NPts']
    RMax = c['RMax']
    ### Operators ###
    #rv = np.linspace(0, RMax, NPts) # r - vector
    rv = np.linspace(0,c['RMax'],c['NPts']+1)[1:]
    NPts = c['NPts']
    dr = rv[1]-rv[0]
    ##-first derivative

    ddr = diags([np.ones(NPts-1), -np.ones(NPts-1)], [1,-1], shape=(NPts, NPts))
    ddr = ddr.toarray()  # Convert to full array for modification
    ddr[0, 0] = -2  # Forward difference at first point
    ddr[0, 1] = 2
    ddr[0,0:3] = np.array([-3,4,-1])
    ddr = ddr/(2*dr)


    ##-second derivative
    d2dr2 = diags([np.ones(NPts-1), -2*np.ones(NPts), np.ones(NPts-1)], offsets=[1,0,-1],shape=(NPts, NPts),dtype=complex)
    d2dr2 = d2dr2.toarray()
    d2dr2[0, :5] = np.array([1,-2,1,0,0])
    #d2dr2[0, :5] = np.array([0,0,0,0,0])
    #d2dr2[-1, -5:] = np.array([0,0,0,0,0])
    d2dr2 = d2dr2/(dr**2)


    #m = conf['M']/2
    Tkin = (-1/c['M'])*(d2dr2 + (dV(2/rv)@ddr))
    c['m'] = m
    c['dr'] = dr
    c['rv'] = rv
    c['ddr'] = ddr
    c['d2dr2'] = d2dr2
    
    c['Tkin'] = TK#csr_matrix(Tkin)
    
def runConfigSetupR(c, r):
    print('CONFIG SETUP RUNNING')
    m = c['M']/2 # = M*M/(M+M) = reduced mass
    NPts = c['NPts']
    RMax = c['RMax']
    ### Operators ###
    #rv = np.linspace(0, RMax, NPts) # r - vector
    rv = r
    NPts = c['NPts']
    dr = rv[1]-rv[0]
    ##-first derivative

    ddr = diags([np.ones(NPts-1), -np.ones(NPts-1)], [1,-1], shape=(NPts, NPts))
    ddr = ddr.toarray()  # Convert to full array for modification
    ddr[0, 0] = -2  # Forward difference at first point
    ddr[0, 1] = 2
    ddr[0,0:3] = np.array([-3,4,-1])
    ddr = ddr/(2*dr)


    ##-second derivative
    d2dr2 = diags([np.ones(NPts-1), -2*np.ones(NPts), np.ones(NPts-1)], offsets=[1,0,-1],shape=(NPts, NPts),dtype=complex)
    d2dr2 = d2dr2.toarray()
    d2dr2[0, :5] = np.array([1,-2,1,0,0])
    #d2dr2[0, :5] = np.array([0,0,0,0,0])
    #d2dr2[-1, -5:] = np.array([0,0,0,0,0])
    d2dr2 = d2dr2/(dr**2)


    #m = conf['M']/2
    Tkin = (-1/c['M'])*(d2dr2 + (dV(2/rv)@ddr))
    c['m'] = m
    c['dr'] = dr
    c['rv'] = rv
    c['ddr'] = ddr
    c['d2dr2'] = d2dr2
    
    c['Tkin'] = Tkin#csr_matrix(Tkin)
    
    c['wv'] = np.diff(rv, prepend=0)
    
### Coupling 
CF = 4/3
nf = 3
TF = 1/2
CA = 3
PI=3.1415926
pi=3.1415926
fmGeV = 5.0676896
gammaE = 0.5772156649

def alpsmu(mu, Nf = 3):
    beta0 = (33 - 2*Nf)/(12 * np.pi)
    beta1 = (153 - 19*Nf)/(24 * np.pi**2)
    beta2 = (77139 - 15099*Nf + 325*Nf**2)/(3456 * np.pi**3)
    beta3 = (29242.964136194125 - 6946.289617003554*Nf + 405.0890404598629*Nf**2 + 1.4993141289437584*Nf**3)/(256 * np.pi**4)
    beta4 = (524.5582754592147 - 181.79877882258594*Nf + 17.156013333434416*Nf**2 - 0.22585710219837543*Nf**3 - 0.0017992914141834987*Nf**4)/(np.pi**5)
    L = np.log(mu**2/0.332**2)
    alpha_s = 1/(beta0*L) - 1/(beta0**2*L**2)*beta1/beta0*np.log(L) + 1/(beta0**3*L**3)*((beta1/beta0)**2*(np.log(L)**2-np.log(L)-1) + beta2/beta0) + \
        1/(beta0**4*L**4)*((beta1/beta0)**3*(-np.log(L)**3 + 5/2*np.log(L)**2 + 2*np.log(L) - 1/2) - (3*beta1*beta2)/beta0**2*np.log(L) + beta3/(2*beta0)) + \
            1/(beta0**5*L**5)*((3*beta1**2*beta2)/beta0**3*(2*np.log(L)**2 - np.log(L) - 1) + (beta1/beta0)**4*(6*np.log(L)**4 - 26*np.log(L)**3 - 9*np.log(L)**2 + 24*np.log(L) + 7)/6 \
                - (beta1*beta3)/(6*beta0**2)*(12*np.log(L) + 1) + 5/3*(beta2/beta0)**2 + beta4/(3*beta0))
    return alpha_s
Ti = np.linspace(0.155,0.5,100)
alphaS = interp1d(Ti, alpsmu(2*np.pi*Ti),kind='cubic',bounds_error=False, fill_value=0.3)   
print('alphaS(0) = ',alphaS(0))
#print('const = ',alphaS(1/a0fb)*CF)
print('alphaS(M) = ',alphaS(4.8))
print('alphaS(0.9,1,1.1) = ', alphaS(0.9/1.334), alphaS(1.0/1.334),  alphaS(1.1/1.334))
# Kappa function
def κ(c,T):
    return c['kSet']    

def TEvo_i(t, T0i):
    if t > 0.6/hbarc:
        return Bjork(t, 0.6/hbarc, T0i)
    else:
        return 0

### Temperature evolution
#vs velocity of sound in plasma = 1/3
def Bjork(t, t0, T0):
    return T0*np.power(t0/t, 1/3)
def Bjk3050(t): #preloaded 30-50 centrality
    return Bjork(t,0.6/hbarc,.425)
def Bjk50100(t): #preloaded 50-100 centrality
    return Bjork(t,0.6/hbarc,.304) 
def Bjk010(t): #preloaded 30-50 centrality
    return Bjork(t,0.6/hbarc,.471) 
    
##### Effective potential

def Lmom(c,l): #Angular Momentum 
    #return l*(l+1)/((2*c['m'])*np.power(c['rv'],2))
    return l*(l+1)/(c['M']*np.power(c['rv'],2))
    
### Munich Potential
## Real
def VsR(c,r,T): 
    #return -(alphaS(1/c['a0fb'])*CF/r) #- ((1/2)*c['gam']*np.power(T,3)*np.power(r,2))
    const = 2/(c['M']*CF*c['a0fb'])
    return -(const*CF/r)

def VoR(c,r,T):
    #return ((1/8)*(alphaS(1/c['a0fb'])*CF/r)) #- ((7/32)*c['gam']*np.power(T,3)*np.power(r,2))
    const = 2/(c['M']*CF*c['a0fb'])
    return ((1/8)*const*CF/r)
    
### Hamiltonian
def HMunichS(c, T, l):
    return c['Tkin'] + dV(VsR(c, c['rv'], 0) + Lmom(c,l))
def HMunichO(c, T, l):
    return c['Tkin'] + dV(VoR(c, c['rv'], 0) + Lmom(c,l))
    
def fullHT(c, T, Hs, Ho, lVals): #Hs-singlet hamiltonan function  Ho-octet  lVals- angular momentum l values 
    #returns a lambda for tempertature of the full hamiltonian
    hF = []
    for l in lVals:
        hF.append(Hs(c,T,l))#.toarray())
    for l in lVals:
        hF.append(Ho(c,T,l))#.toarray())
    return block_diag(*hF)
    
### Jump operators

def C0_pref(c,T):
    return np.sqrt(κ(c,T)*np.power(T,3)/(np.power(NC,2)-1))
def C0_struct(c):
    r = dV(c['rv'])
    Z = np.zeros((c['NPts'], c['NPts']))
    C0_1 = [Z,Z,Z,r*1/np.sqrt(3)]
    C0_2 = [Z,Z,r*1,Z]
    C0_3 = [Z,r*np.sqrt(np.power(NC,2)-1)/np.sqrt(3),Z,Z]
    C0_4 = [r*np.sqrt(np.power(NC,2)-1),Z,Z,Z]
    C0struct = bmat([C0_1, C0_2, C0_3, C0_4])
    return C0struct
def C1_pref(c,T):
    return np.sqrt((np.power(NC,2)-4)*κ(c,T)*np.power(T,3)/(2*(np.power(NC,2)-1)))
def C1_struct(c):
    r = dV(c['rv'])
    Z = np.zeros((c['NPts'], c['NPts']))
    C1_1 = [Z,Z,Z,Z]
    C1_2 = [Z,Z,Z,Z]
    C1_3 = [Z,Z,Z,r*1/np.sqrt(3)]
    C1_4 = [Z,Z,r*1,Z]
    C1struct = bmat([C1_1, C1_2, C1_3, C1_4])
    return C1struct

def getC0(c): # returns lambda of T for C0 operator !!!!!!!! lVals must= [0,1] !!!!!!!!
    #Structure
    r = dV(c['rv'])
    Z = np.zeros((c['NPts'], c['NPts']))
    C0_1 = [Z,Z,Z,r*1/np.sqrt(3)]
    C0_2 = [Z,Z,r*1,Z]
    C0_3 = [Z,r*np.sqrt(np.power(NC,2)-1)/np.sqrt(3),Z,Z]
    C0_4 = [r*np.sqrt(np.power(NC,2)-1),Z,Z,Z]
    C0struct = bmat([C0_1, C0_2, C0_3, C0_4])
    return lambda T: np.sqrt(κ(c,T)*np.power(T,3)/(np.power(NC,2)-1))*C0struct

def getC1(c): # returns lambda of T for C1 operator !!!!!!!! lVals must= [0,1] !!!!!!!!
    #Structure
    r = dV(c['rv'])
    Z = np.zeros((c['NPts'], c['NPts']))
    C1_1 = [Z,Z,Z,Z]
    C1_2 = [Z,Z,Z,Z]
    C1_3 = [Z,Z,Z,r*1/np.sqrt(3)]
    C1_4 = [Z,Z,r*1,Z]
    C1struct = bmat([C1_1, C1_2, C1_3, C1_4])
    return lambda T: np.sqrt((np.power(NC,2)-4)*κ(c,T)*np.power(T,3)/(2*(np.power(NC,2)-1)))*C1struct   
    
def badBoiFilter(psi): # numerical solutions where all the wf sits at large r
    mabs = np.mean(np.abs(psi))
    #if (psi[-1] > mabs) or (psi[-2] > mabs):
        #print('reject largeR')
    return (psi[-1] > mabs) or (psi[-2] > mabs) #or (psi[-3] > mabs)
def badBoiFilter2(psi): # numerical solution where all the wf sits at small r
    mabs = np.mean(np.abs(psi)[:10])
    #if (np.abs(psi[0]) > mabs): 
        #print('reject smallR')
    return (np.abs(psi[0]) > mabs) #or (np.abs(psi[1]) > mabs) or (np.abs(psi[2]) > mabs)

def compBasis(c, H_L, L_list, IPF, inv=False): #H_L-H function of only l,  L_list = [N-s-states, N-p-states, N-d-states, ...]
    res = {}                     #IPF - inner product function - sand(), sandX()
    stLet = ['S','P','D']           #inv invert eigensolver, switch ordering
    for i, Ln in enumerate(L_list):
        Hi = H_L(i)
        if issparse(Hi):
            Hi = Hi.toarray()
        
        eigenvalues, eigenvectors = np.linalg.eig(Hi)
        idx = np.argsort(eigenvalues)
        if inv:
            idx = idx[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        j=0
        chk=0
        while j < Ln+chk:
            eig = eigenvectors[:,j]
            if badBoiFilter(eig):# or badBoiFilter2(eig):
                chk += 1
                j+=1
                continue
            #print('shapes: ', eig.shape, Hi.shape)
            #print(type(Hi))
            res[str(j+1-chk)+stLet[i]] = {'E':eigenvalues[j],'wf':eig/np.sqrt(IPF(c,eig,eig))}
            j += 1
        
        #for j in range(Ln):
        #    eig = eigenvectors[:,j]
        #    #print('eigshape',eig)
        #    res[str(j+1)+stLet[i]] = {'E':eigenvalues[j],'wf':eig/np.sqrt(IPF(c,eig,eig)*np.pi*4)}
    return res      
    
def compBasisDUMB(c,H_L, L_list): #H_L-H function of only l,  L_list = [N-s-states, N-p-states, N-d-states, ...]
    res = {}
    stLet = ['S','P','D']
    for i, Ln in enumerate(L_list):
        Hi = H_L(i)
        if issparse(Hi):
            Hi = Hi.toarray()
        print('Hi.shape',Hi.shape)
        eigenvalues, eigenvectors = np.linalg.eig(Hi)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        for j in range(Ln):
            eig = np.array(eigenvectors[:,j]).T[0]
            res[str(j+1)+stLet[i]] = {'E':eigenvalues[j],'wf':eig/np.sqrt(sandH(c,eig.conj(),eig))}
    return res  

def compBasisEXP(c,H_L, L_list): #H_L-H function of only l,  L_list = [N-s-states, N-p-states, N-d-states, ...]
    res = {}
    stLet = ['S','P','D']
    for i, Ln in enumerate(L_list):
        Hi = H_L(i)
        if issparse(Hi):
            Hi = Hi.toarray()
        print('Hi.shape',Hi.shape)
        eigenvalues, eigenvectors = np.linalg.eig(Hi)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        for j in range(Ln):
            eig = np.array(eigenvectors[:,j]).T[0]
            res[str(j+1)+stLet[i]] = {'E':eigenvalues[j],'wf':eig/np.sqrt(sandX(c,eig.conj(),eig))}
    return res


### ERT Solver

import numpy as np
from multiprocessing import shared_memory
from scipy.sparse import csr_matrix


class Tbug:
    def __init__(self):
        self.dat = []
        self.Ltime = 0
    def peep(self, n):
        cT = time.time()
        if n == 0:
            self.dat.append([])
        else: 
            self.dat[-1].append(cT - self.Ltime)
        self.Ltime = cT
            

from scipy.sparse.linalg import norm

def evoWF(A, H, dt, K, psi):
    
    def Jk():
        return (-1j*H)+((K/2)*((A@A)-(dag(A)@A)))
    def Uka():
        return (dt*Jk())-(1j*np.sqrt(K*dt)*A)
    def Vka():
        return (dt*Jk())+(1j*np.sqrt(K*dt)*A)
    
    Uop = expm_multiply(Uka().tocsr(), psi)/2
    Vop = expm_multiply(Vka().tocsr(), psi)/2
    #Uop = expm(Uka().toarray())@ psi
    #Vop = expm(Vka().toarray())@ psi
    return Uop, Vop
def evoWF_CN(A, H, dt, K, psi):
    
    def Jk():
        return (-1j*H)+((K/2)*((A@A)-(dag(A)@A)))
    def Uka():
        return (dt*Jk())-(1j*np.sqrt(K*dt)*A)
    def Vka():
        return (dt*Jk())+(1j*np.sqrt(K*dt)*A)
    
    Uop = expm_multiply(Uka().tocsr(), psi)/2
    Vop = expm_multiply(Vka().tocsr(), psi)/2
    #Uop = expm(Uka().toarray())@ psi
    #Vop = expm(Vka().toarray())@ psi
    return Uop, Vop
def evoWFunp(a):
    return evoWF(a[0],a[1],a[2],a[3],a[4])

import numpy as np
from multiprocessing import shared_memory
from scipy.sparse import csr_matrix

def apply_op_worker(task_chunk, psi_shape, op_metadata, psi_shm_name):
    """
    task_chunk: List of (psi_idx, op_idx)
    op_metadata: List of (shm_names, shapes) for each operator
    """
    # 1. Access Shared Psi
    shm_psi = shared_memory.SharedMemory(name=psi_shm_name)
    psi_all = np.ndarray(psi_shape, dtype=np.complex128, buffer=shm_psi.buf)
    
    # 2. Local Cache for Reconstructed Operators
    # We reconstruct the CSR object once per worker per operator to save time
    reconstructed_ops = {}
    results = []

    for p_idx, o_idx in task_chunk:
        if o_idx not in reconstructed_ops:
            meta = op_metadata[o_idx] # ( (names), (shapes), mat_shape )
            
            # Attach to the 3 CSR arrays
            shm_d = shared_memory.SharedMemory(name=meta[0][0])
            shm_i = shared_memory.SharedMemory(name=meta[0][1])
            shm_p = shared_memory.SharedMemory(name=meta[0][2])
            
            d_arr = np.ndarray(meta[1][0], dtype=np.complex128, buffer=shm_d.buf)
            i_arr = np.ndarray(meta[1][1], dtype=np.int32, buffer=shm_i.buf)
            p_arr = np.ndarray(meta[1][2], dtype=np.int32, buffer=shm_p.buf)
            
            reconstructed_ops[o_idx] = csr_matrix((d_arr, i_arr, p_arr), shape=meta[2])

        # 3. Apply Operator
        results.append(reconstructed_ops[o_idx] @ psi_all[p_idx])

    shm_psi.close()
    return results
    
from itertools import product
from concurrent.futures import ProcessPoolExecutor

def parallel_apply(ops_list, psi_list):
    # Ensure psi_list is a proper numpy array with the correct complex type
    psi_array = np.array(psi_list, dtype=np.complex128) 
    shm_track = []
    
    # --- STEP 1: Share Psi ---
    # Use psi_array.nbytes to get the exact byte count required
    shm_psi = shared_memory.SharedMemory(create=True, size=psi_array.nbytes)
    shm_track.append(shm_psi)
    
    # Map the buffer and copy data
    # IMPORTANT: The dtype here MUST match what the worker uses
    psi_shared = np.ndarray(psi_array.shape, dtype=psi_array.dtype, buffer=shm_psi.buf)
    psi_shared[:] = psi_array[:]

    # --- STEP 2: Share Operators ---
    op_metadata = []
    for op in ops_list:
        op = op.tocsr()
        # Explicitly check dtypes for sparse components (usually complex128 and int32)
        d_shm = shared_memory.SharedMemory(create=True, size=op.data.nbytes)
        i_shm = shared_memory.SharedMemory(create=True, size=op.indices.nbytes)
        p_shm = shared_memory.SharedMemory(create=True, size=op.indptr.nbytes)
        shm_track.extend([d_shm, i_shm, p_shm])
        
        # Copy data using the exact same dtype as the original sparse matrix
        np.ndarray(op.data.shape, dtype=op.data.dtype, buffer=d_shm.buf)[:] = op.data
        np.ndarray(op.indices.shape, dtype=op.indices.dtype, buffer=i_shm.buf)[:] = op.indices
        np.ndarray(op.indptr.shape, dtype=op.indptr.dtype, buffer=p_shm.buf)[:] = op.indptr
        
        op_metadata.append((
            (d_shm.name, i_shm.name, p_shm.name),
            (op.data.shape, op.indices.shape, op.indptr.shape),
            op.shape,
            op.data.dtype,    # Pass the dtypes to the worker
            op.indices.dtype  # Pass the dtypes to the worker
        ))

    # --- STEP 3: Dispatch ---
    all_tasks = list(product(range(len(psi_list)), range(len(ops_list))))
    num_workers = 8 # Adjust based on CPU cores
    chunks = [all_tasks[i::num_workers] for i in range(num_workers)]
    
    final_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(apply_op_worker, c, psi_array.shape, op_metadata, shm_psi.name) 
                   for c in chunks]
        for f in futures:
            final_results.extend(f.result())

    # --- STEP 4: Cleanup ---
    for shm in shm_track:
        shm.close()
        shm.unlink()
        
    return final_results

import numpy as np
from scipy.sparse.linalg import expm_multiply

def worker_expm_apply(task_chunk):
    """
    task_chunk: A list of tuples (generator_matrix, psi_vector)
    """
    results = []
    for M, psi in task_chunk:
        # M is the sparse matrix (the 'generator')
        # expm_multiply computes exp(M) @ psi without forming exp(M)
        # We divide by 2 to match your original normalization logic
        res = expm_multiply(M, psi) / 2.0
        results.append(res)
    return results

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product

class ERTensMT: #AkL list of dissapators w/ sqrt(Γ) inc. that are functions of (t) <- also H
    def __init__(self, c, H, AkL, dt, psiLi, R=None, init_t=None): 
        self.c = c
        self.H = H #Hamiltonian
        self.AkL = AkL #Dissipators
        self.K = len(AkL) 
        self.dt = dt #Explicit time step
        self.psiL = np.array(psiLi) #np.array([psiI])#np.array([psiI for i in range(R)])
        self.R = np.power(2*len(AkL),2) if R==None else R #Truncation size
        self.t = 0.0 if init_t==None else init_t
        self.TB = Tbug()
        self.n_workers = self.c['N_Workers']
        self.executor = ProcessPoolExecutor(max_workers=self.n_workers)
    def Jk(self, A, Ht):
        return (-1j*Ht)+((self.K/2)*((A@A)-(dag(A)@A)))
    def Uk2(self, A, Ht):
        return (self.dt*self.Jk(A,Ht))-(1j*np.sqrt(self.K*self.dt)*A)
    def Vk2(self, A, Ht):
        return (self.dt*self.Jk(A,Ht))+(1j*np.sqrt(self.K*self.dt)*A)
    def shutdown(self):
        self.executor.shutdown()
    def step2(self):
        N = self.c['NPts']
        nInit = len(self.psiL)
        nPsi = []
        At = [csr_matrix(A(self.t)) for A in self.AkL]
        Hi = csr_matrix(self.H(self.t))
        UV_list = [Op(A_i, Hi) for A_i in At for Op in (self.Uk2, self.Vk2)]
        #psiList = self.psiL
        
        all_tasks = [(op, psi) for op in UV_list for psi in self.psiL]
        
        # We manually split tasks into chunks (one per worker) 
        # to stop the "small task" overhead.
        num_tasks = len(all_tasks)
        chunk_size = (num_tasks + self.n_workers - 1) // self.n_workers
        chunks = [all_tasks[i:i + chunk_size] for i in range(0, num_tasks, chunk_size)]

        # 3. Parallel Execution on the Warm Pool
        # executor.map handles the distribution
        raw_results = list(self.executor.map(worker_expm_apply, chunks))
        
        nPsi = np.array([vec for sublist in raw_results for vec in sublist])

        Sij = np.dot(nPsi.conj(),nPsi.T)
        w, Uuns = np.linalg.eigh(Sij)
        idx = np.argsort(w)[::-1]
        Ulk = Uuns[:,idx]
        if len(nPsi) > self.R:
            UR = Ulk.T[:self.R]
        else:
            UR = Ulk.T
        rPsi = UR@nPsi
        self.psiL = trimLowEigs(rPsi, w)
        #print(self.psiL.shape)
        ### zero other zections
        #self.psiL = np.array([zero_other_sec(self.c['rv'],self.c['NPts'], psi) for psi in self.psiL])
        #self.psiL = rPsi
        self.t += self.dt
        

    def step(self):
        N = self.c['NPts']
        self.TB.peep(0)
        nInit = len(self.psiL)
        nPsi = []
        At = [csr_matrix(A(self.t)) for A in self.AkL]
        Hi = csr_matrix(self.H(self.t))
        
        jobs = list(product(At, self.psiL))
        K = self.K
        dt = self.dt
        args = [(A, Hi, dt, K, psi) for A in At for psi in self.psiL]
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(evoWFunp, args))
        nPsi = np.array([v for pair in results for v in pair])  # flatten (U, V) pairs
        self.TB.peep(1)
        nPsi = np.array(nPsi)
        Sij = np.dot(nPsi.conj(),nPsi.T)
        w, Uuns = np.linalg.eigh(Sij)
        idx = np.argsort(w)[::-1]
        Ulk = Uuns[:,idx]
        self.TB.peep(2)
        if len(nPsi) > self.R:
            UR = Ulk.T[:self.R]
        else:
            UR = Ulk.T
        self.TB.peep(3)
        rPsi = UR@nPsi
        self.psiL = trimLowEigs(rPsi, w)
        print(self.psiL.shape)
        ### zero other zections
        self.psiL = np.array([zero_other_sec(self.c['rv'],self.c['NPts'], psi) for psi in self.psiL])
        #self.psiL = rPsi
        self.t += self.dt
        self.TB.peep(4)
        
    def step_single(self):
        N = self.c['NPts']
        self.TB.peep(0)
        nInit = len(self.psiL)
        nPsi = []
        At = [csr_matrix(A(self.t)) for A in self.AkL]
        Hi = csr_matrix(self.H(self.t))
        
        jobs = list(product(At, self.psiL))
        K = self.K
        dt = self.dt
        args = [(A, Hi, dt, K, psi) for A in At for psi in self.psiL]
        newPsi = []
        for arg_set in args:
            res = evoWFunp(arg_set)
            newPsi.append(res[0])
            newPsi.append(res[1])
        nPsi = np.array(newPsi)
        #with ProcessPoolExecutor() as executor:
        #    results = list(executor.map(evoWFunp, args))
        #nPsi = np.array([v for pair in results for v in pair])  # flatten (U, V) pairs
        self.TB.peep(1)
        nPsi = np.array(nPsi)
        Sij = np.dot(nPsi.conj(),nPsi.T)
        w, Uuns = np.linalg.eigh(Sij)
        idx = np.argsort(w)[::-1]
        Ulk = Uuns[:,idx]
        self.TB.peep(2)
        if len(nPsi) > self.R:
            UR = Ulk.T[:self.R]
        else:
            UR = Ulk.T
        self.TB.peep(3)
        rPsi = UR@nPsi
        self.psiL = trimLowEigs(rPsi, w)
        print(self.psiL.shape)
        ### zero other zections
        self.psiL = np.array([zero_other_sec(self.c['rv'],self.c['NPts'], psi) for psi in self.psiL])
        #self.psiL = rPsi
        self.t += self.dt
        self.TB.peep(4)
    
    def obs(self, O):
        return np.mean(np.array([psi.conj()@O@psi for psi in self.psiL]))
    def tick(self, O):
        self.step()
        return self.obs(O)
    def getRho(self):
        return np.sum(np.array([np.outer(psi,dag(psi)) for psi in self.psiL]),axis=0)#/len(self.psiL)
    def step1TH(self):
        N = self.c['NPts']
        self.TB.peep(0)
        nInit = len(self.psiL)
        nPsi = []
        At = [csr_matrix(A(self.t)) for A in self.AkL]
        Hi = csr_matrix(self.H(self.t))
        for A in self.AkL:
            for psi in self.psiL:
                nPsi.append(expm_multiply(self.Uk2(A, Hi), psi))
                nPsi.append(expm_multiply(self.Vk2(A, Hi), psi))
        nPsi = np.array(nPsi)
        self.TB.peep(1)
        nPsi = np.array(nPsi)
        Sij = np.dot(nPsi.conj(),nPsi.T)
        w, Uuns = np.linalg.eigh(Sij)
        idx = np.argsort(w)[::-1]
        Ulk = Uuns[:,idx]
        self.TB.peep(2)
        if len(nPsi) > self.R:
            UR = Ulk.T[:self.R]
        else:
            UR = Ulk.T
        self.TB.peep(3)
        rPsi = UR@nPsi
        self.psiL = trimLowEigs(rPsi, w)
        #self.psiL = rPsi
        self.t += self.dt
        self.TB.peep(4)

def sandHr(r,Np,L,R): # inner product flat
    return dag(np.power(r,2)*L)@R#*c['dr']
def get_mask(r,Np,i):
    z = np.linspace(0,0,Np)
    o = np.linspace(0,0,Np) + 1
    return np.concatenate([o if j==i else z for j in range(4)])
    
def zero_other_sec(r, Np, Bpsi):
    norms = [sandHr(r,Np,Bpsi[i*Np:(i+1)*Np],Bpsi[i*Np:(i+1)*Np]) for i in range(4)]
    maxI = np.argmax(np.array(norms))
    return Bpsi * get_mask(r,Np,maxI)
    
    

def trimLowEigs(psiL, w):
    idx = np.argsort(w)[::-1] #get order of weights
    wS = np.abs(w[idx]) # sort weights
    
    wN = np.sum(w)*1e-86 #relative cutoff
    ni = firstBelow(wS, wN)
    return psiL[:ni,:]
def ensTR(psiL):
    return np.sum(np.array([psi.conj()@psi for psi in psiL]))
def ensTRr(c,psiL):
    return np.sum(np.array([sandHB(c,psi,psi) for psi in psiL]))
        
def firstBelow(Lst, n):
    for i in range(len(Lst)):
        if Lst[i] < n:
            return i
def getRho2(psiL):
    return np.sum(np.array([np.outer(psi,dag(psi)) for psi in psiL]),axis=0)#/len(psiL)
 
#def ObsEnsH(c, psiL, psiP):
    #return np.sum(np.array([(psi.conj()*psiP)@(psiP.conj()*psi) for psi in psiL]))
#    return np.sum(np.array([np.sum(np.array([np.trapz(psi[i*c['NPts']:(i+1)*c['NPts']]*(psiP[i*c['NPts']:(i+1)*c['NPts']]) * np.power(c['rv'],2), c['rv'], axis=0) for i in range(4)])) for psi in psiL])) 
def ObsEnsH(c, psiL, psiP):
    #return np.sum(np.array([(psi.conj()*psiP)@(psiP.conj()*psi) for psi in psiL]))
    return np.sum(np.array([sandHB(c,psi,psiP)*(sandHB(c,psi,psiP).conjugate()) for psi in psiL]))    

from scipy.linalg import eigh 
 
def expEigs(c):
    c0 = {}
    RMax = 1e4 #[GeV]
    NPts = 2048
    m = c['M']/2 # = M*M/(M+M) = reduced mass
    ##-x
    xv = np.linspace(-8,np.log(RMax),NPts)
    dx = xv[1]-xv[0]
    rv = np.exp(xv) # r = e^x
    ##-first derivative
    ddx = diags([np.ones(NPts-1), -np.ones(NPts-1)], [1,-1], shape=(NPts, NPts))
    ddx = ddx.toarray()  # Convert to full array for modification
    ddx[0, 0] = -3
    ddx[0, 1] = 4
    ddx[0, 2] = -1
    ddx[-1, -3] = 3
    ddx[-1, -2] = -4
    ddx[-1, -1] = 1
    ddx = ddx/(2*dx)
    ##-second derivative
    d2dx2 = diags([np.ones(NPts-1), -2*np.ones(NPts), np.ones(NPts-1)], offsets=[1,0,-1],shape=(NPts, NPts))
    d2dx2 = d2dx2.toarray()
    d2dx2[0, 0] = 35/12#/np.sqrt(2)
    d2dx2[0, 1] = -104/12#/np.sqrt(2)
    d2dx2[0, 2] = 114/12#/np.sqrt(2)
    d2dx2[0, 3] = -56/12#/np.sqrt(2)
    d2dx2[0, 4] = 11/12#/np.sqrt(2)
    
    d2dx2[-1, -5] = 35/12#/np.sqrt(2)
    d2dx2[-1, -4] = -104/12#/np.sqrt(2)
    d2dx2[-1, -3] = 114/12#/np.sqrt(2)
    d2dx2[-1, -2] = -56/12#/np.sqrt(2)
    d2dx2[-1, -1] = 11/12#/np.sqrt(2)
    d2dx2 = d2dx2/(dx**2)
    ##-kinetic energy term
    Tkin = (-1/(2*m))*dV(np.exp(-2*xv))@(d2dx2 + ddx)
    
    VS = dV(VsR(c,rv,0))
    HS = Tkin + VS
    
    num_states = 400
    eigenvalues, eigenvectors = np.linalg.eig(HS)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    res = {}
    c0['xv'] = xv
    for j in range(num_states):
            eig = np.array(eigenvectors[:,j+1]).T[0]
            res[str(j+1)+'S'] = {'E':eigenvalues[j+1],'wf':eig/np.sqrt(sandX(c0,eig.conj(),eig))}

    
    return res, xv
    
def expEigsO(c):
    c0 = {}
    RMax = 1e4 #[GeV]
    NPts = 2048
    m = c['M']/2 # = M*M/(M+M) = reduced mass
    ##-x
    xv = np.linspace(-8,np.log(RMax),NPts)
    dx = xv[1]-xv[0]
    rv = np.exp(xv) # r = e^x
    ##-first derivative
    ddx = diags([np.ones(NPts-1), -np.ones(NPts-1)], [1,-1], shape=(NPts, NPts))
    ddx = ddx.toarray()  # Convert to full array for modification
    ddx[0, 0] = -3
    ddx[0, 1] = 4
    ddx[0, 2] = -1
    ddx[-1, -3] = 3
    ddx[-1, -2] = -4
    ddx[-1, -1] = 1
    ddx = ddx/(2*dx)
    ##-second derivative
    d2dx2 = diags([np.ones(NPts-1), -2*np.ones(NPts), np.ones(NPts-1)], offsets=[1,0,-1],shape=(NPts, NPts))
    d2dx2 = d2dx2.toarray()
    d2dx2[0, 0] = 35/12#/np.sqrt(2)
    d2dx2[0, 1] = -104/12#/np.sqrt(2)
    d2dx2[0, 2] = 114/12#/np.sqrt(2)
    d2dx2[0, 3] = -56/12#/np.sqrt(2)
    d2dx2[0, 4] = 11/12#/np.sqrt(2)
    
    d2dx2[-1, -5] = 35/12#/np.sqrt(2)
    d2dx2[-1, -4] = -104/12#/np.sqrt(2)
    d2dx2[-1, -3] = 114/12#/np.sqrt(2)
    d2dx2[-1, -2] = -56/12#/np.sqrt(2)
    d2dx2[-1, -1] = 11/12#/np.sqrt(2)
    d2dx2 = d2dx2/(dx**2)
    ##-kinetic energy term
    Tkin = (-1/(2*m))*dV(np.exp(-2*xv))@(d2dx2 + ddx)
    
    VS = dV(VoR(c,rv,0))
    HS = Tkin + VS
    
    num_states = 400
    eigenvalues, eigenvectors = np.linalg.eig(HS)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[-idx]
    eigenvectors = eigenvectors[:, -idx]
    
    res = {}
    c0['xv'] = xv
    for j in range(num_states):
            eig = np.array(eigenvectors[:,j+1]).T[0]
            res[str(j+1)+'S'] = {'E':eigenvalues[j+1],'wf':eig/np.sqrt(sandX(c0,eig.conj(),eig))}

    
    return res, xv

# Coulomb-Sturmian basis functions
def S_nl(n, l, b, r):
    """Reduced Coulomb-Sturmian radial function S_{n,l}(r), normalized so ∫_0^∞ S^2 dr = 1."""
    x = 2.0 * r / b
    # stable prefactor using gammaln
    log_pref = 0.5 * (gammaln(n + 1) - gammaln(n + 2*l + 3)) + 0.5 * np.log(2.0 / b)
    pref = np.exp(log_pref)
    poly = eval_genlaguerre(n, 2*l + 2, x)
    return pref * (x**(l + 1)) * poly * np.exp(-r / b)

def constructBasisSets(c):
    RMax = 1e2 #[GeV]
    NPts = 1024
    m = c['M']/2 # = M*M/(M+M) = reduced mass
    ##-x
    xv = np.linspace(-6,np.log(RMax),NPts)
    dx = xv[1]-xv[0]
    rv = np.exp(xv) # r = e^x
    
    ### Kinetic energy: finite differnce operator
    ##-first derivative
    ddx = diags([np.ones(NPts-1), -np.ones(NPts-1)], [1,-1], shape=(NPts, NPts))
    ddx = ddx.toarray()  # Convert to full array for modification
    ddx[0,:3] = [-3,4,-1]
    #ddx[-1, -3:] = [1, -4, 3]
    ddx[-1, -3:] = [3, -4, 1]
    ddx = ddx/(2*dx)
    ##-second derivative
    d2dx2 = diags([np.ones(NPts-1), -2*np.ones(NPts), np.ones(NPts-1)], offsets=[1,0,-1],shape=(NPts, NPts))
    d2dx2 = d2dx2.toarray()
    d2dx2[0, :5] = np.array([35,-104,114,-56,11])/12
    d2dx2[-1, -5:] = np.array([35,-104,114,-56,11])/12#np.array([-11,56,-114,104,-35])/12
    d2dx2 = d2dx2/(dx**2)
    ##-kinetic energy term
    Tkin = (-1/(2*m))*dV(np.exp(-2*xv))@(d2dx2 - ddx)
    #Tkin = (-1/(2*m))*d2dx2
    

    
    NumF = 5
    #Input basis set
    b_par = 1.05#Controls the length scale of the Coulomb-Sturmian basis functions
    inpSetL0 = [S_nl(n, 0, b_par, rv)/rv for n in range(NumF)]
    inpSetL0_N = [psi/np.sqrt(sandX({'xv':xv}, psi, psi)) for psi in inpSetL0]
    inpSetL1 = [S_nl(n, 1, b_par, rv)/rv for n in range(NumF)]
    inpSetL1_N = [psi/np.sqrt(sandX({'xv':xv}, psi, psi)) for psi in inpSetL1]
    inpS = [inpSetL0_N,inpSetL1_N] # ind = L
    r_inpS = [[np.interp(c['rv'],rv,psi) for psi in inpSetL0_N], [np.interp(c['rv'],rv,psi) for psi in inpSetL1_N]]
    Nr_inpS = [[psi/np.sqrt(sandH(c,psi,psi)) for psi in r_inpS[0]], [psi/np.sqrt(sandH(c,psi,psi)) for psi in r_inpS[1]]]
    
    SL_set = [['s',0],['s',1],['o',0],['o',1]]
    out_set = []
    out_set_EXP = []
    e_set = []
    H_set = []
    H_E = []
    for SL in SL_set:
        H_i = np.array(dV(VsR(c,rv,SL[1])*np.exp(xv)) + Tkin) if SL[0]=='s' else np.array(dV(VoR(c,rv,SL[1])*np.exp(xv)) + Tkin)
        out_i = [H_i@psi for psi in inpS[SL[1]]]
        e_i = [np.sqrt(sandX({'xv':xv}, psi, psi)) for psi in out_i]
        out_R = [np.interp(c['rv'], rv, psi) for psi in out_i]
        out_RN = [psi/np.sqrt(sandH(c,psi,psi)) for psi in out_R]
        H_set.append(np.array(out_RN).T @ dV(e_i) @ np.array(Nr_inpS[SL[1]]))
        out_set.append(out_RN)
        out_set_EXP.append(out_i)
        H_E.append(H_i)
        e_set.append(e_i)
    
    
    
    return Nr_inpS, out_set, H_set, inpS, out_set_EXP, rv, H_E
    #return [inpSetL0_N,inpSetL1_N], np.array([out_s0, out_s1, out_o0, out_o1]), rv, xv, [Hs0, Hs1, Ho0, Ho1]
    
    #Output basis set
    #print('SCHEK: ', [sandX({'xv':xv},p,p) for p in inpSetL0])
    #return inpSetL0, rv, xv
    
    