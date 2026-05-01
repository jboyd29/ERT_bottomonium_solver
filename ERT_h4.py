import numpy as np
import scipy as sp
#import matplotlib.pyplot as plt
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

### TAMU potential
## Real
TAMU_aS = 0.27 #[Num] coupling
TAMU_mS = 0.2 #[GeV]
TAMU_sig = 0.225 #[GeV^2] string tension
def TAMU_mD(T): #[GeV] debeye mass
    return -1.62861 + (14.2767*T) + (22.6503*np.power(T,2)) + (13.8112*np.power(T,3))
def TAMU_cb(T): #[Num]
    return -0.96804 + (14.1067*T) + (16.9140*np.power(T,2)) + (9.12996*np.power(T,3))
#def TAMU_VsR(c,r,T): # real part of potential
#    return ((-4/3)*TAMU_aS*((np.exp(-TAMU_mD(T)*r)/r)+TAMU_mD(T)))-((TAMU_sig/TAMU_mS)*np.exp(-TAMU_mS*r - np.power(TAMU_cb(T)*TAMU_mS*r,2))-1)
#def TAMU_VsR_VAC(c,r): # real part of potential VACUUM
#    return ((-4/3)*TAMU_aS/r) + TAMU_sig*r
def TAMU_VsR(c,r,T):
    if T > 0.0001:
        return ((-4/3)*TAMU_aS*((np.exp(-TAMU_mD(T)*r)/r)+TAMU_mD(T)))-((TAMU_sig/TAMU_mS)*np.exp(-TAMU_mS*r - np.power(TAMU_cb(T)*TAMU_mS*r,2))-1)
    else:
        return ((-4/3)*TAMU_aS/r) + TAMU_sig*r
def TAMU_VoR(c,r,T):
    return (-1/8)*TAMU_VsR(c,r,T)
def TAMU_VsEff(c,r,T,l):
    return TAMU_VsR(c,r,T) + Lmom(c,l)
def TAMU_VoEff(c,r,T,l):
    return TAMU_VoR(c,r,T) + Lmom(c,l)
    
def HTAMUS(c, T, l):
    return c['Tkin'] + diags(TAMU_VsEff(c,c['rv'],T,l), offsets=0, format="csr")
def HTAMUO(c, T, l):
    return c['Tkin'] + diags(TAMU_VoEff(c,c['rv'],T,l), offsets=0, format="csr")

##### WLC (Wilson line correllator) potential

from scipy.interpolate import interp1d

# Data points
a_values = {195: 0.3451416129380123, 251: 0.3540000000000000, 293: 0.2111624792698106, 352: 0.5840000000000000, 500: 2.5510279125459900, 700: -5.6792992804627950}
b_values = {195: 5.7596595792483400, 251: 8.0811004284077700, 293: 16.7261548715366500, 352: 8.7110000000000000, 500: 8.7087552607485900, 700: 9.1014034889180100}
c_values = {195: 1.6987523342470630, 251: 2.3523280492290040, 293: 3.0755235576774710, 352: 4.6518465323646230, 500: 7.7242443254585160, 700: 8.7085875604828500}

# Construct the interpolation functions
a_interp = interp1d(list(a_values.keys()), list(a_values.values()), kind='linear')
b_interp = interp1d(list(b_values.keys()), list(b_values.values()), kind='linear')
c_interp = interp1d(list(c_values.keys()), list(c_values.values()), kind='linear')
HBARC = 0.19732697 # GeV fm

# Define the function to compute phi
def phi_interpolated(r, tem):
    a = a_interp(tem)
    b = b_interp(tem)
    c = c_interp(tem)
    phi = 1. + a * np.exp(-b * r * r) - (1. + a) * np.exp(-c * r * r)
    return phi

def mass_md_or_ms(temperature, mass_values):
    mass_temperatures = [195, 251, 293, 352, 500, 700] # MeV
    return np.interp(temperature, mass_temperatures, mass_values)

def V_WLC(r, temperature):
    alphaS = 0.27
    sigma = 0.225
    c_b_values = [1.2, 1.675, 1.925, 2.3, 3.0, 3.75]
    tem = np.round(temperature * 1000)
    ms_values = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2] # GeV
    md_values = [0.36, 0.81, 0.99, 1.11, 1.6, 2.0] # GeV
    amplitude = [1.02, 1.22277, 1.53819, 1.76918, 2.15539, 2.41190851541762]
    ms = mass_md_or_ms(tem, ms_values)
    md = mass_md_or_ms(tem, md_values)
    am = mass_md_or_ms(tem, amplitude)
    c_b= mass_md_or_ms(tem, c_b_values)
    # print('T =', temperature, 'MeV')
    # print('ms =', ms, 'GeV')
    # print('md =', md, 'GeV')
    Vc = -4./3.*alphaS*np.exp(-md*r)/r - 4./3.*alphaS*md
    Vs = -sigma*np.exp(-ms*r-np.power(c_b*ms*r,2))/ms + sigma/ms
    if temperature < 0.0001:
        if r > 1.1 / HBARC: r = 1.1 / HBARC
        Vc = -4./3.*alphaS/r
        Vs = sigma*r
        return Vc + Vs
    phi = phi_interpolated(r * HBARC, tem)
    return Vc + Vs + am*1j * phi
    
import numpy as np

def V_WLC_vectorized(r, temperature):
    """
    Vectorized version of V_WLC. 
    r: numpy array of distances
    temperature: scalar temperature value
    """
    alphaS = 0.27
    sigma = 0.225
    c_b_values = [1.2, 1.675, 1.925, 2.3, 3.0, 3.75]
    tem = np.round(temperature * 1000)
    ms_values = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2] 
    md_values = [0.36, 0.81, 0.99, 1.11, 1.6, 2.0] 
    amplitude = [1.02, 1.22277, 1.53819, 1.76918, 2.15539, 2.41190851541762]
    
    # These helpers likely return scalars based on 'tem'
    ms = mass_md_or_ms(tem, ms_values)
    md = mass_md_or_ms(tem, md_values)
    am = mass_md_or_ms(tem, amplitude)
    c_b = mass_md_or_ms(tem, c_b_values)

    if temperature < 0.0001:
        # Vectorized clipping for the T=0 case
        r_eff = np.clip(r, None, 1.1 / HBARC)
        Vc = -4./3. * alphaS / r_eff
        Vs = sigma * r_eff
        return Vc + Vs
    else:
        # Standard finite T calculation (element-wise on array r)
        Vc = -4./3. * alphaS * np.exp(-md * r) / r - 4./3. * alphaS * md
        Vs = -sigma * np.exp(-ms * r - np.power(c_b * ms * r, 2)) / ms + sigma / ms

    # Ensure phi_interpolated is also capable of taking a vector r
    phi = phi_interpolated(r * HBARC, tem)
    
    return Vc + Vs + am * 1j * phi
    


def WLC_VsR(c,r,T):
    return np.real(V_WLC_vectorized(r,T))
def WLC_VoR(c,r,T):
    return np.real((-1/8)*V_WLC_vectorized(r,T))
    
def HWLCS(c, T, l):
    return c['Tkin'] + diags(WLC_VsR(c, c['rv'], T) + Lmom(c,l), offsets=0, format="csr")
def HWLCO(c, T, l):
    return c['Tkin'] + diags(WLC_VoR(c, c['rv'], T) + Lmom(c,l), offsets=0, format="csr")

import scipy.sparse as sp
def HWLC(c, T):
    def block(a,b,c,d):
        return sp.block_diag((sp.csr_matrix(a), sp.csr_matrix(b), sp.csr_matrix(c), sp.csr_matrix(d)), format='csr')
    return block(HWLCS(c, T, 0), HWLCS(c, T, 1), HWLCO(c, T, 0), HWLCO(c, T, 1))
    #return sp.block_diag((HWLCS(c, T, 0), HWLCS(c, T, 1), HWLCO(c, T, 0), HWLCO(c, T, 1)), format='csr')

  
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
    
    
def getC0_WLC(c, T): # returns lambda of T for C0 operator !!!!!!!! lVals must= [0,1] !!!!!!!!
    #Structure
    #r = dV(c['rv'])
    r = dV(np.sqrt(np.imag(V_WLC_vectorized(c['rv'], T))))
    print(r.shape)
    Z = np.zeros((c['NPts'], c['NPts']))
    C0_1 = [Z,Z,Z,r*1/np.sqrt(3)]
    C0_2 = [Z,Z,r*1,Z]
    C0_3 = [Z,r*np.sqrt(np.power(NC,2)-1)/np.sqrt(3),Z,Z]
    C0_4 = [r*np.sqrt(np.power(NC,2)-1),Z,Z,Z]
    C0struct = bmat([C0_1, C0_2, C0_3, C0_4])
    return np.sqrt(1/(np.power(NC,2)-1))*C0struct

def getC1_WLC(c, T): # returns lambda of T for C1 operator !!!!!!!! lVals must= [0,1] !!!!!!!!
    #Structure
    #r = dV(c['rv'])
    r = dV(np.sqrt(np.imag(V_WLC_vectorized(c['rv'], T))))
    Z = np.zeros((c['NPts'], c['NPts']))
    C1_1 = [Z,Z,Z,Z]
    C1_2 = [Z,Z,Z,Z]
    C1_3 = [Z,Z,Z,r*1/np.sqrt(3)]
    C1_4 = [Z,Z,r*1,Z]
    C1struct = bmat([C1_1, C1_2, C1_3, C1_4])
    return np.sqrt((np.power(NC,2)-4)/(2*(np.power(NC,2)-1)))*C1struct   
    
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


### ERT Solver

import numpy as np
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
import numpy as np
from scipy.sparse import csr_matrix


class ERTens_1T:
    def __init__(self, c, H, AkL, dt, psiLi, R=None, init_t=None):
        self.c = c
        self.H = H #Hamiltonian
        self.AkL = AkL #Dissipators
        self.K = len(AkL) 
        self.dt = dt #Explicit time step
        self.psiL = np.array(psiLi) #np.array([psiI])#np.array([psiI for i in range(R)])
        self.R = np.power(2*len(AkL),2) if R==None else R #Truncation size
        self.t = 0.0 if init_t==None else init_t
    def Jk(self, A, Ht):
        return (-1j*Ht)+((self.K/2)*((A@A)-(dag(A)@A)))
    def Uk(self, A, Ht):
        return (self.dt*self.Jk(A,Ht))-(1j*np.sqrt(self.K*self.dt)*A)
    def Vk(self, A, Ht):
        return (self.dt*self.Jk(A,Ht))+(1j*np.sqrt(self.K*self.dt)*A)
    def step(self):
        N = self.c['NPts']
        nInit = len(self.psiL)
        nPsi = []
        At = [csr_matrix(A(self.t)) for A in self.AkL] # Jump operators evaluated at current time
        Hi = csr_matrix(self.H(self.t)) # Hamiltonian evaluated at current time
        # Apply each evolution operator (Uk,Vk) for each jump operator Ak on each wavefunction
        nPsi = np.array([expm_multiply(Op(Ai, Hi).tocsr(), psi)/2 for Op in (self.Uk, self.Vk) for Ai in At for psi in self.psiL])
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
        
    def obs(self, O):
        return np.mean(np.array([psi.conj()@O@psi for psi in self.psiL]))
    def tick(self, O):
        self.step()
        return self.obs(O)
    def getRho(self):
        return np.sum(np.array([np.outer(psi,dag(psi)) for psi in self.psiL]),axis=0)#/len(self.psiL)


import numpy as np
from numba import jit, complex128, int32
from scipy.sparse import csr_matrix

# --- JIT KERNELS ---

@jit(nopython=True, cache=True)
def sparse_matvec(data, indices, indptr, vec):
    """Fast JIT-compiled Sparse Matrix-Vector Multiplication."""
    res = np.zeros(len(indptr) - 1, dtype=complex128)
    for i in range(len(indptr) - 1):
        for j in range(indptr[i], indptr[i + 1]):
            res[i] += data[j] * vec[indices[j]]
    return res

@jit(nopython=True, cache=True)
def apply_taylor_expm(data, indices, indptr, psi, order=4):
    """
    Applies exp(M) * psi using a Taylor expansion.
    Since your operators are already scaled by dt, this is highly accurate and JIT-friendly.
    """
    res = psi.copy()
    term = psi.copy()
    for i in range(1, order + 1):
        # term = (M / i) @ term
        term = sparse_matvec(data, indices, indptr, term) / i
        res += term
    return res

@jit(nopython=True, cache=True)
def compute_ensemble_evolution(psiL, op_list_data, op_indices, op_indptr, R):
    """
    Core math loop. Moves through the ensemble and applies 
    evolution operators. 
    """
    n_psi = []
    # We apply both Uk and Vk for each Ak operator
    for i in range(len(op_list_data)):
        data = op_list_data[i]
        for psi in psiL:
            # Taylor exp replaces expm_multiply for speed
            evolved = apply_taylor_expm(data, op_indices, op_indptr, psi) / 2.0
            n_psi.append(evolved)
    
    return np.stack(n_psi)

# --- REPLACED CLASS ---

class ERTens_1T_Optimized:
    def __init__(self, c, H_func, AkL_funcs, dt, psiLi, R=None, init_t=None):
        self.c = c
        self.H_func = H_func # Function returning matrix
        self.AkL_funcs = AkL_funcs 
        self.dt = dt
        self.psiL = np.array(psiLi, dtype=complex128)
        self.K = len(AkL_funcs)
        self.R = (2 * self.K)**2 if R is None else R
        self.t = 0.0 if init_t is None else init_t
        
        # We assume the sparsity pattern (indices/indptr) is constant 
        # to avoid re-allocating memory every step.
        sample_mat = csr_matrix(self.AkL_funcs[0](0.0))
        self.indices = sample_mat.indices
        self.indptr = sample_mat.indptr

    def _get_operator_data(self, A_mat, H_mat):
        """Constructs the Jk, Uk, Vk operator data arrays manually."""
        # Jk = (-1j*H) + (K/2)*(A@A - Adag@A)
        # To keep this fast, we compute the 'data' array for the CSR format
        # This assumes H and A share the same sparsity pattern.
        
        # Uk = dt*Jk - 1j*sqrt(K*dt)*A
        # Vk = dt*Jk + 1j*sqrt(K*dt)*A
        
        # Note: For maximum speed, you should ideally pre-calculate the 
        # Jk operator as a matrix to avoid repeated matrix multiplications.
        
        # Simplified for demonstration: Constructing the combined operator CSR data
        Uk_mat = (self.dt * self._Jk_logic(A_mat, H_mat)) - (1j * np.sqrt(self.K * self.dt) * A_mat)
        Vk_mat = (self.dt * self._Jk_logic(A_mat, H_mat)) + (1j * np.sqrt(self.K * self.dt) * A_mat)
        return Uk_mat.data, Vk_mat.data

    def _Jk_logic(self, A, Ht):
        # Using Scipy here is fine as it's only done once per time-step
        return (-1j * Ht) + ((self.K / 2) * ((A @ A) - (A.getH() @ A)))

    def step(self):
        # 1. Evaluate operators at current time
        Hi = csr_matrix(self.H_func(self.t))
        
        # 2. Build the 'data' arrays for all Uk and Vk
        all_op_data = []
        for A_func in self.AkL_funcs:
            Ai = csr_matrix(A_func(self.t))
            uk_data, vk_data = self._get_operator_data(Ai, Hi)
            all_op_data.append(uk_data)
            all_op_data.append(vk_data)
        
        # 3. Use the JIT engine to propagate the ensemble
        # Convert list to array for Numba
        op_data_stack = np.stack(all_op_data)
        
        n_psi = compute_ensemble_evolution(
            self.psiL, op_data_stack, self.indices, self.indptr, self.R
        )
        
        # 4. Truncation (Linear Algebra on Gram Matrix)
        # This part is best handled by optimized BLAS/LAPACK via numpy
        Sij = n_psi.conj() @ n_psi.T
        w, Uuns = np.linalg.eigh(Sij)
        
        # Sort and truncate
        idx = np.argsort(w)[::-1]
        UR = Uuns[:, idx].T[:self.R]
        
        self.psiL = UR @ n_psi
        self.t += self.dt

    def tick(self, O):
        self.step()
        # Fast expectation value
        return np.real(np.mean([np.vdot(p, O @ p) for p in self.psiL]))


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

