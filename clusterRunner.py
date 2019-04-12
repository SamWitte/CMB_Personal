import numpy as np
import os
from CMB_c import CMB
import multiprocessing as mp
from constants import *
import itertools
import time
t_0 = time.time()


HubbleParam = 67.66
n_s_index = 0.9619
A_s_norm = 2.215e-9
Neff = 3.046
z_reion = 10.
Om_b_load = 0.0492
Om_c_load = 0.266


Ftag = 'StandardUniverse'
OM_b = Om_b_load
OM_c = Om_c_load
OM_g = 5.43e-5
OM_L = 0.7 # Doesnt matter, it calculates with flat Uni

lmax_Pert = 7
process_Num = 10

compute_LP = True
compute_CMB = True
compute_MPS = False
# Note, don't copute MPS and CMB at same time. This requires different kgrid...

if compute_MPS:
    kmin = 5e-4
    kmax = 15.
    knum = 60
else:
    kmin = 8e-4
    kmax = 4e-1
    knum = 120

kTotNum = 5000
lmax = 2500
kgrid_Full = np.logspace(np.log10(kmin), np.log10(kmax), kTotNum)

if compute_MPS:
    kgrid = np.logspace(np.log10(kmin), np.log10(kmax), knum)
else:
    kgrid = np.logspace(np.log10(kmin), np.log10(kmax), knum)

try:
    os.remove('precomputed/*')
except FileNotFoundError:
    pass

recfast_infile = os.getcwd() + '/Recfast++.v2.0/runfiles/parameters_run.ini'
replaceAll(recfast_infile, "N_eff =","N_eff = {:.3f}\n".format(Neff))
replaceAll(recfast_infile, "Omega_m =","Omega_m = {:.3f}\n".format(OM_b + OM_c))
replaceAll(recfast_infile, "Omega_b =","Omega_b = {:.3f}\n".format(OM_b))
replaceAll(recfast_infile, "h100 =","h100 = {:.3f}\n".format(HubbleParam/1e2))
os.system("cd Recfast++.v2.0/")
os.system("./Recfast++.v2.0/Recfast++ Recfast++.v2.0/runfiles/parameters_run.ini")

recfast_F = np.loadtxt("outputs/Xe_Recfast++.Rec_corrs_CT2010.dat")
np.savetxt('precomputed/xe_working.dat', np.column_stack((1. / (1. + recfast_F[:,0]), recfast_F[:,1] )))
np.savetxt('precomputed/tb_working.dat',np.column_stack((1. / (1. + recfast_F[:,0]), recfast_F[:,-1] )))
os.remove("outputs/Xe_Recfast++.Rec_corrs_CT2010.dat")

SetCMB = CMB(OM_b, OM_c, OM_g, OM_L, kmin=kmin, kmax=kmax, knum=kTotNum, lmax=lmax,
             lvals=1, Ftag=Ftag, lmax_Pert=lmax_Pert,
             HubbleParam=HubbleParam, n_s_index=n_s_index, A_s_norm=A_s_norm,
             z_reion=z_reion, Neff=Neff, killF=True)

def CMB_wrap(kval):
    try:
        chk = len(kval)
        klist = kval
    except:
        klist = [kval]
    returnlist = np.zeros(len(klist), dtype=object)
    for i,kv in enumerate(klist):
        indx = np.where(kv == kgrid)[0][0]
        hold = SetCMB.runall(kv, indx, compute_LP=compute_LP, compute_TH=False,
                compute_CMB=True, compute_MPS=False)
#        print('Finsihed Computing k: ', kv)
        if hold is None:
            returnlist[i] = [indx, False]
        else:
            returnlist[i] = [indx, hold]
    return returnlist


ThetaTabTot = np.zeros_like(SetCMB.ThetaTabTot)
mps_tks = np.zeros((len(kgrid), 2))
kfail = []
ellTab = SetCMB.ThetaTabTot[0, :]
ThetaTabTot[0,:] = ellTab

def runPl(kgrid, processes=1):

    if __name__ == '__main__':
        pool = mp.Pool(processes=processes)
        hold_info = pool.map_async(CMB_wrap, kgrid)
        structure = hold_info.get()
        pool.close()
    flat_list = [item for sublist in structure for item in sublist]
    return flat_list

def get_ThetaTab(x):
    integ2 = [S_SW[:, i] * SetCMB.sphB[x](listC[i*len(tau_list):len(tau_list)*(i+1)]) +
              S_D[:, i] * SetCMB.sphB_D[x](listC[i*len(tau_list):len(tau_list)*(i+1)]) + S_ISW[:, i] *
              SetCMB.sphB[x](listC[i*len(tau_list):len(tau_list)*(i+1)]) for i in range(len(kgrid_Full))]
    return [np.trapz(integ2[i], tau_list) for i in range(len(integ2))]

structure = []
if compute_LP:
    print('Running Pool...')
    t_k = time.time()
    structure = runPl(kgrid, processes=process_Num)
    list_ord = []
    t_k_end = time.time()
    print('Time to Compute k-evolution: ', t_k_end - t_k)
    for i in range(len(structure)):
        list_ord.append(structure[i][0])
        structure[i] = structure[i][1]
    structure = np.asarray(structure)[list_ord]


    if compute_CMB:
        print('Interpolating Sources and Calculating Bessel Functions...')
        t_k = time.time()
        tau_list, S_SW, S_D, S_ISW = SetCMB.theta_construction(structure, kgrid)

        t_k_end = time.time()
        print('Time to Compute Theta Construction: ', t_k_end - t_k)
        listC = [x*y for x,y in list(itertools.product(kgrid_Full, (SetCMB.eta0 - tau_list)))]
        par_scan = range(len(ellTab))
        if __name__ == '__main__':
            pool = mp.Pool(processes=process_Num)
            structureTheta = pool.map(get_ThetaTab, par_scan)
            pool.close()
        t_k = time.time()
        print('Time to Compute Theta Table: ',  t_k - t_k_end)
        for i in range(len(structureTheta)):
            SetCMB.ThetaTabTot[1:, i] = structureTheta[i]

        SetCMB.computeCMB(kgrid_Full, SetCMB.ThetaTabTot)
        t_k_end = time.time()
        print('Time to Compute Cls: ',  t_k_end - t_k)
    else:
        for i in range(len(structure)):
            if isinstance(structure[i][1], bool):
                kfail.append(structure[i][0])
            else:
                mps_tks[structure[i][0], 0] = kgrid[structure[i][0]]
                mps_tks[structure[i][0], 1] = structure[i][1]

if compute_MPS:
    print('Kfail: ', kgrid[kfail], len(kgrid[kfail]))
    for i in range(len(kgrid))[::-1]:
        if i in kfail:
            kgrid = np.delete(kgrid, [i], axis=0)
            mps_tks = np.delete(mps_tks, [i], axis=0)
    SetCMB.runall(1, 1, compute_LP=False, compute_TH=False,
               compute_CMB=False, compute_MPS=compute_MPS, ThetaTab=mps_tks)


print(time.time() - t_0)
