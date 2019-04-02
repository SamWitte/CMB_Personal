import glob
import os
import numpy as np

LCDM = True
if LCDM:
    tag = 'StandardUniverse_CL_Table'
    fileOut = 'Data/LCDM_TT_Cls.dat'

combine_files = False

if combine_files:
    existingF = np.loadtxt(fileOut)

filesLoad = glob.glob('../OutputFiles/' + tag + '*.dat')

data_collect = []

for f in filesLoad:
    dataL = np.loadtxt(f)
    omegaB = float(f[f.find('_Ob_')+4:f.find('_Oc_')])
    omegaC = float(f[f.find('_Oc_')+4:f.find('_H0_')])
    hub = float(f[f.find('_H0_')+4:f.find('_Neff_')])
    ns = float(f[f.find('_Ns_')+4:f.find('_As_')])
    As = float(f[f.find('_As_')+4:f.find('_zreion_')])
    zre = float(f[f.find('_zreion_')+8:f.find('_.dat')])

    ell_vals = dataL[:, 0]
    cls = np.log10(dataL[:,1] * 1e12)
    vals = [omegaB, omegaC, hub, ns, np.log10(As), zre]
    data_in = np.concatenate((vals, cls ))
    data_collect.append(data_in)

if combine_files:
    data_collect = np.vstack(( data_collect, existingF))

np.savetxt(fileOut, data_collect, fmt='%.6e')
np.savetxt('Data/Ell_Values.dat', ell_vals)
