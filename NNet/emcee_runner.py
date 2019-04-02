import numpy as np
import tensorflow as tf
import emcee
from scipy.interpolate import interp1d
import os
from Network import *
import itertools
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import matplotlib.pyplot as plt
import corner
import corner_plot as cp
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times','Palatino']})
#rc('text', usetex=True)

LCDM = True
if LCDM:
    metaF = 'MetaGraphs/LCDM_TT_Cls/LCDM_TT_Cls_Graph_Global_'
    dataF = 'Data/LCDM_TT_Cls.dat'
    arrayName = '_LCDM_'
    tag = ''

Nhidden = 100

BurnPTS = 400
NSTEPS = 1e4
ndim, nwalkers = 6, 100


filePTS = 'mcmc_pts/MCMC_pts_'+arrayName + tag + '_.dat'
scterPlt = 'mcmc_pts/MCMC_PLT_'+arrayName + tag + '_.pdf'
cornerPLT = 'mcmc_pts/Corner_'+arrayName + tag + '_.pdf'

params_low = [0.01, 0.2, 60., -9.5, 0.93, 6.]
params_space = [0.06, 0.12, 20, 2, 0.07, 12]

nnet = MLP_Nnet(HiddenNodes=Nhidden, LCDM=LCDM)
nnet.main_nnet()
nnet.load_matrix_elems()
modeler = ImportGraph(metaF, dataF)

ell_vals = np.loadtxt('Data/Ell_Values.dat')
com_likeli = np.loadtxt('Data/COM_PowerSpect_CMB-TT-hiL-binned.txt')


def lnprior(theta):
    omb, omc, h0, ass, ns, zre = theta
    if (0.01 < omb < 0.07) and (0.2 < omc < 0.32) and (60 < h0 < 80) and (-9.5 < ass < -7.5) and (0.93 < ns < 1.) and (6 < zre < 18):
        return 0.
    else:
        return -np.inf

def ln_like(theta):
    omb, omc, h0, ass, ns, zre = theta
    preds = modeler.run_yhat([theta])
    clTT = interp1d(np.log10(ell_vals), preds, kind='cubic', fill_value=0., bounds_error=False)
    tt_pred = np.zeros_like(com_likeli[:, 2])
    for i in range(len(tt_pred)):
        ellV = range(com_likeli[i, 1], com_likeli[i,2]+1)
        tt_pred[i] = np.trapz(np.power(10., clTT(np.log10(ellV))),  ellV) / float(len(ellV))
#        tt_pred[i] = quad(lambda x: np.power(10., clTT(np.log10(x))), com_likeli[i, 1], com_likeli[i,2])[0] / (com_likeli[i,2] - com_likeli[i,1])

    likeli = np.sum(((com_likeli[:, 3] - tt_pred) / com_likeli[:,4] ) ** 2.)
    return -likeli

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_like(theta)

pos = np.asarray([params_low + params_space*np.random.rand(ndim) for i in range(nwalkers)])
f = open("chain.dat", "w")
f.close()

print 'Running Sampler.'
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=5)
sampler.run_mcmc(pos, NSTEPS)

#try:
#    print 'Autocorrelation Time: ', sampler.get_autocorr_time()
#except:
#    pass

print 'Making Plots...'

samples = sampler.chain[:, BurnPTS:, :].reshape((-1, ndim))
#print samples
np.savetxt(filePTS, samples)
print '2sigma limit: ', np.percentile(samples[:,0], [95])

axis_labels=[r"$\Omega_b$", r"$\Omega_c$",r"$H_0$",r"$log10(A_s)$",r"$n_s$",r"$z_{reion}$"]
fig2 = cp.corner_plot(samples,axis_labels=axis_labels, cmap='Pastel1', fname=cornerPLT)
