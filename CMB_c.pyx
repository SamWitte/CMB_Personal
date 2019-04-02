import os
from boltzmann_c import *
from scipy.integrate import quad
from scipy.special import spherical_jn
import itertools
import cython
cimport numpy as cnp
import numpy as np
from constants import *
#from multiprocessing import *
#from pathos.multiprocessing import ProcessingPool as Pool
#from functools import partial
import math

path = os.getcwd()

cdef extern from "math.h":
    #float INFINITY
    double exp(double)
    double sqrt(double)
    double log(double)
    double log10(double)

@cython.boundscheck(False) # turn off bounds-checking for entire function
class CMB(object):

    def __init__(self, OM_b, OM_c, OM_g, OM_L, kmin=5e-3, kmax=0.5, knum=200,
                 lmax=2500, lvals=250, Ftag='StandardUniverse', lmax_Pert=5, multiverse=False,
                 OM_b2=0., OM_c2=0., OM_g2=0., OM_L2=0., HubbleParam=67.77, n_s_index=0.9619,
                 A_s_norm=3.044, z_reion=10., Neff=3.045, Nbrane=0, killF=False):

        self.HubbleParam = HubbleParam
        self.n_s_index = n_s_index
        self.A_s = A_s_norm
        self.Neff = Neff
        self.z_reion = z_reion

        self.OM_b = OM_b
        self.OM_c = OM_c
        self.OM_g = OM_g
        self.OM_L = OM_L
        self.OM_nu = (7./8) * (4./11.)**(4./3) * Neff * OM_g

        self.OM_b2 = OM_b2
        self.OM_c2 = OM_c2
        self.OM_g2 = OM_g2
        self.OM_L2 = OM_L2
        self.OM_nu2 = (7./8) * (4./11.)**(4./3) * Neff * OM_g2

        self.OM_M = self.OM_b + self.OM_c + (self.OM_c2 + self.OM_b2) * Nbrane
        self.OM_R = self.OM_g  + self.OM_nu + (self.OM_g2 + self.OM_nu2) * Nbrane
        self.OM_Lam = self.OM_L + self.OM_L2 * Nbrane

        self.Nbrane = Nbrane
        if OM_b2 != 0.:
            self.PressFac = (OM_g2 / OM_b2) / (OM_g / OM_b2)
        else:
            self.PressFac = 0.

        self.eCDM = OM_c + OM_c2 * Nbrane

        self.kmin = kmin
        self.kmax = kmax
        self.knum = knum
        self.k_remove = []
        self.Ftag = Ftag
        if multiverse:
            self.f_tag = '_Nbranes_{:.0e}_PressFac_{:.2e}_eCDM_{:.2e}'.format(self.Nbrane, self.PressFac, self.eCDM)
        else:
            self.f_tag = ''

        self.cparam_tag = '_Ob_{:.4e}_Oc_{:.4e}_H0_{:.4e}_Neff_{:.4e}_Ns_{:.4e}_As_{:.4e}_zreion_{:.2f}_'.format(self.OM_b, self.OM_c,
                    HubbleParam, Neff, n_s_index, A_s_norm, self.z_reion)

        self.lmax = lmax
        self.lvals = lvals

        self.lmax_Pert = lmax_Pert
        self.lmin = 10

        self.multiverse = multiverse
        self.init_pert = -1/6.

        ell_val = list(range(self.lmin, self.lmax, 20))
        indxT = len(ell_val)
        for i in list(range(indxT)):
            if (i%2 == 1) and (ell_val[indxT - i - 1] > 300):
#                ell_val.remove(indxT - i - 1)
                ell_val.pop(indxT - i - 1)

        if killF:
            self.clearfiles()
            self.loadfiles(tau=True)
        else:
            self.loadfiles(tau=False)

        self.ThetaFile = path + '/OutputFiles/' + self.Ftag
        if self.multiverse:
            self.ThetaFile += '_Nbrane_{:.0e}_PressFac_{:.2e}_eCDM_{:.2e}_ThetaCMB_Table.dat'.format(self.Nbrane, self.PressFac, self.eCDM)
        else:
            self.ThetaFile += '_ThetaCMB_Table.dat'

        self.ThetaTabTot = np.zeros((self.knum+1, len(ell_val)))
        self.ThetaTabTot[0,:] = ell_val

        self.fill_inx = 0


    def runall(self, kVAL, kindx, compute_LP=False, compute_TH=False,
               compute_CMB=False, compute_MPS=False, kgrid=None, ThetaTab=None):
        t_0 = time.time()

        LP_fail = False
        cdef cnp.ndarray[double] ret_arr, theta_arr

        if kVAL is None:
            if compute_MPS:
                self.kgrid = np.logspace(log10(self.kmin), log10(self.kmax), self.knum)
            else:
                self.kgrid = np.logspace(log10(self.kmin), log10(self.kmax), self.knum)

        if compute_LP:
            sources = self.kspace_linear_pert(kVAL, compute_CMB)
            t_1 = time.time()
            if sources is None:
                LP_fail = True
                print('FAIL PERT: ', kVAL)
                return None
            if not compute_TH:
                return sources

        if compute_TH and not LP_fail:
            print('Computing Theta Files...\n')
            if not compute_LP:
                print('Must compute LP....')

            theta_arr = self.theta_integration(kVAL, sources)
            t_2 = time.time()
            print(t_2 - t_1, t_1 - t_0)
            return theta_arr


        if compute_CMB:
            print('Computing CMB...\n')
            self.computeCMB(kgrid, ThetaTab)
        if compute_MPS:
            print('Computing Matter Power Spectrum...\n')
            self.MatterPower(ThetaTab)
        return

    def clearfiles(self):
        if os.path.isfile(path + '/precomputed/ln_a_CT_working.dat'):
            os.remove(path + '/precomputed/ln_a_CT_working.dat')

        if os.path.isfile(path + '/precomputed/xe_working' + self.f_tag + '.dat'):
            os.remove(path + '/precomputed/xe_working' + self.f_tag + '.dat')
        if os.path.isfile(path + '/precomputed/tb_working' + self.f_tag + '.dat'):
            os.remove(path + '/precomputed/tb_working' + self.f_tag + '.dat')

        if os.path.isfile(path + '/precomputed/working_expOpticalDepth' + self.f_tag + '.dat'):
            os.remove(path + '/precomputed/working_expOpticalDepth' + self.f_tag + '.dat')
        if os.path.isfile(path + '/precomputed/working_VisibilityFunc' + self.f_tag + '.dat'):
            os.remove(path + '/precomputed/working_VisibilityFunc' + self.f_tag + '.dat')

    def loadfiles(self, tau=False):
        if not self.multiverse:
            SingleUni = Universe(1., self.OM_b, self.OM_c, self.OM_g, self.OM_L,
                                 self.OM_nu, hubble_c=self.HubbleParam, zreion=self.z_reion)
            self.ct_to_scale = lambda x: SingleUni.ct_to_scale(x)
            self.scale_to_ct = lambda x: SingleUni.scale_to_ct(x)
            if tau:
                SingleUni.tau_functions()
            self.eta0 = SingleUni.eta_0
            self.H_0 = SingleUni.H_0
        else:
            ManyUni = ManyBrane_Universe(self.Nbrane, 1., [self.OM_b, self.OM_b2], [self.OM_c, self.OM_c2],
                                          [self.OM_g, self.OM_g2], [self.OM_L, self.OM_L2],
                                          [self.OM_nu, self.OM_nu2], hubble_c=self.HubbleParam, zreion=self.z_reion)
            self.ct_to_scale = lambda x: ManyUni.ct_to_scale(x)
            self.scale_to_ct = lambda x: ManyUni.scale_to_ct(x)
            if tau:
                ManyUni.tau_functions()
            self.eta0 = ManyUni.eta_0
            self.H_0 = ManyUni.H_0
        opt_depthL = np.loadtxt(path + '/precomputed/working_expOpticalDepth' + self.f_tag + '.dat')
        self.opt_depth = interp1d(opt_depthL[:,0], opt_depthL[:,1], kind='cubic', bounds_error=False, fill_value=0.)
        visfunc = np.loadtxt(path + '/precomputed/working_VisibilityFunc' + self.f_tag + '.dat')
        self.Vfunc = interp1d(visfunc[:,0], visfunc[:,1], kind='cubic', fill_value=0.)
        self.eta_start = self.scale_to_ct(np.min(visfunc[:,0]))
        return

    def load_bessel(self):
        self.sphB = np.zeros(len(self.ThetaTabTot[0,:]), dtype=object)
        self.sphB_D = np.zeros(len(self.ThetaTabTot[0,:]), dtype=object)
        xlist = np.linspace(0, self.kmax * self.eta0, 4000)
        for i in range(len(self.ThetaTabTot[0,:])):
            self.sphB[i] = interp1d(xlist, spherical_jn(int(self.ThetaTabTot[0, i]), xlist), kind='cubic',
                                    fill_value=0., bounds_error=False)
            self.sphB_D[i] = interp1d(xlist, spherical_jn(int(self.ThetaTabTot[0, i]), xlist, derivative=True),
                                      kind='cubic', fill_value=0., bounds_error=False)
        return

    def kspace_linear_pert(self, kVAL, compute_TH):
        try:
            chk = len(kVAL)
            kgrid = kVal
        except TypeError:
            kgrid = [kVAL]

        for k in kgrid:
            stepsize = 1e-2
            if not self.multiverse:
                SingleUni = Universe(k, self.OM_b, self.OM_c, self.OM_g, self.OM_L, self.OM_nu,
                                     stepsize=stepsize, accuracy=1e-3, lmax=self.lmax_Pert,
                                     hubble_c=self.HubbleParam, zreion=self.z_reion)
                soln = SingleUni.solve_system(compute_TH)
            else:
                ManyUni = ManyBrane_Universe(self.Nbrane, k, [self.OM_b, self.OM_b2], [self.OM_c, self.OM_c2],
                                          [self.OM_g, self.OM_g2], [self.OM_L, self.OM_L2],
                                          [self.OM_nu, self.OM_nu2], accuracy=1e-2,
                                          stepsize=stepsize, lmax=self.lmax_Pert, hubble_c=self.HubbleParam, zreion=self.z_reion)
                soln = ManyUni.solve_system(compute_TH)

        return soln


    def theta_integration(self, double kVAL, cnp.ndarray[double, ndim=2] sources):
        cdef int i, j
        cdef int slen = len(sources[:,0])

        cdef cnp.ndarray[double] ell_tab = self.ThetaTabTot[0,:]
        cdef int ellen = len(ell_tab)
        cdef cnp.ndarray[double] thetaVals = np.zeros(ellen)

        cdef cnp.ndarray[double] vis = np.zeros(slen)
        cdef cnp.ndarray[double] expD0 = np.zeros(slen)
        cdef cnp.ndarray[double] der2_pi = np.zeros(slen)
        cdef double h1, h2
        for i in range(slen):
            vis[i] = self.visibility(sources[i, 0])
            expD0[i] = self.exp_opt_depth(sources[i, 0])
#            if i > 1:
#                h2 = sources[i,0] - sources[i-1, 0]
#                h1 = sources[i-1,0] - sources[i-2, 0]
#                if (h1 == 0.) or (h2 == 0.):
#                    der2_pi[i] = 0.
#                    continue
#                der2_pi[i] = 2.*(h2*(sources[i, 2]*vis[i]) - (h1+h2)*(sources[i-1, 2]*vis[i-1]) +
#                        h1*(sources[i-2, 2]*vis[i-2]))/(h1*h2*(h1+h2))
#            else:
#                der2_pi[i] = 0.

        cdef cnp.ndarray[double] integ1 = np.zeros(slen)
        cdef double eta_v, sphB, sphB2

        for i in range(ellen):
            
            for j in range(slen):
                eta_v = sources[j, 0]
                sphB = spherical_jn(int(ell_tab[i]), kVAL * (self.eta0 - eta_v))
                if self.eta0 - eta_v != 0:
                    sphB2 = (spherical_jn(int(ell_tab[i] - 1), kVAL * (self.eta0 - eta_v)) - \
                    (ell_tab[i]+1.) * sphB / (kVAL * (self.eta0 - eta_v)))
                else:
                    sphB2 = 0.

                integ1[j] = vis[j] * (sources[j, 1] + der2_pi[j] * (3. / 4.) / kVAL**2.) * sphB + \
                    vis[j] * sources[j, 3] * sphB2 + expD0[j] * sources[j, 4] * sphB
                if j > 0:
                    thetaVals[i] += 0.5 * (integ1[j] + integ1[j - 1]) * (eta_v - sources[j - 1, 0])

        return thetaVals

    def theta_construction(self, sources, kgrid_i):
        self.load_bessel()
        cdef int i, j, z
        cdef double ell, kk, eta_v, vis, expD


        cdef cnp.ndarray[double] kgrid_new = np.logspace(log10(self.kmin), log10(self.kmax), self.knum)
        cdef cnp.ndarray[double] tau_list = sources[0, :, 0]
        if tau_list[-1] == self.eta0:
            tau_list[-1] = self.eta0 - 1.
        cdef cnp.ndarray[double] integ1 = np.zeros_like(tau_list)
        sounce_interps_SW = np.zeros(len(tau_list), dtype=object)
        sounce_interps_D = np.zeros(len(tau_list), dtype=object)
        sounce_interps_ISW = np.zeros(len(tau_list), dtype=object)

        for i in range(len(tau_list)):
            sounce_interps_SW[i] = interp1d(np.log10(kgrid_i), sources[:, i,  1],
                    kind='cubic', fill_value=0., bounds_error=False)
            sounce_interps_D[i] = interp1d(np.log10(kgrid_i), sources[:, i, 2],
                    kind='cubic', fill_value=0., bounds_error=False)
            sounce_interps_ISW[i] = interp1d(np.log10(kgrid_i), sources[:, i, 3],
                    kind='cubic', fill_value=0., bounds_error=False)

        sources_SW = np.zeros((len(tau_list), self.knum))
        sources_D = np.zeros((len(tau_list), self.knum))
        sources_ISW = np.zeros((len(tau_list), self.knum))
        for i,tau in enumerate(tau_list):
            vis =  self.visibility(tau)
            expD = self.exp_opt_depth(tau)
            sources_SW[i, :] =  vis * sounce_interps_SW[i](np.log10(kgrid_new))
            sources_D[i, :] =  vis * sounce_interps_D[i](np.log10(kgrid_new))
            sources_ISW[i, :] =  expD * sounce_interps_ISW[i](np.log10(kgrid_new))
        return tau_list, sources_SW, sources_D, sources_ISW

    def computeCMB(self, kgrid, ThetaTab):
        print('Computing CMB...')
        cdef cnp.ndarray[double] ell_tab = ThetaTab[0,:]
        cdef cnp.ndarray[double, ndim=2] CL_table = np.zeros((len(ell_tab), 2))
        cdef double GF, ell

        if not self.multiverse:
            GF = ( self.OM_M / self.growthFactor(1.))**2.
        else:
            GF = ( self.OM_M / self.growthFactor(1.))**2.

        for i in range(len(ell_tab)):
            ell = ell_tab[i]
            CL_table[i, 0] = ell
            CL_table[i, 1] =  ell * (ell + 1) * trapz( (ThetaTab[1:, i] / self.init_pert)**2. *
                                                  (kgrid / 0.05)**(self.n_s_index - 1.) / kgrid, kgrid) * self.A_s

            if math.isnan(CL_table[i, 0]):
                print(i, ell)
                print(np.abs(ThetaTab[1:, i]/self.init_pert)**2.)
                print(ThetaTab[1:, i])
                print(cL_interp(np.log10(kgrid)))
                exit()

        Cl_name = path + '/OutputFiles/' + self.Ftag + '_CL_Table' + self.cparam_tag
        if self.multiverse:
            Cl_name += '_Nbrane_{:.0e}_PressFac_{:.2e}_eCDM_{:.2e}.dat'.format(self.Nbrane, self.PressFac, self.eCDM)
        else:
            Cl_name += '.dat'
        np.savetxt(Cl_name, CL_table)
        return

    def growthFactor(self, a):
        # D(a)
        if not self.multiverse:
            prefac = 5. * self.OM_M / 2. *(self.Hubble(a) / self.H_0) * self.H_0**3.
        else:
            prefac = 5.* self.OM_M /2. *(self.Hubble(a) / self.H_0) * self.H_0**3.

        integ_pt = quad(lambda x: 1./(x*self.Hubble(x)**3.), 0., a)[0]
        return prefac * integ_pt

    def exp_opt_depth(self, eta):
        return self.opt_depth(self.ct_to_scale(eta))

    def visibility(self, eta):
        return self.Vfunc(self.ct_to_scale(eta))

    def vis_max_eta(self):
        etaL = np.logspace(0, log10(self.eta0), 10000)
        visEval = self.visibility(etaL)
        return etaL[np.argmax(visEval)]

    def MatterPower(self, Tktab):
        Tktab[:, 1] /= Tktab[0, 1]
        # T(k) = \Phi(k, a=1) / \Phi(k = Large, a= 1)
        # P(k,a=1) = 2 pi^2 * \delta_H^2 * k / H_0^4 * T(k)^2
        # Tktab = self.TransferFuncs()
        #kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
        PS = np.zeros_like(Tktab[:, 0])
        for i, k in enumerate(Tktab[:, 0]):
            PS[i] = k*Tktab[i, 1]**2. * k**(self.n_s_index - 1.)
        if self.multiverse:
            np.savetxt(path + '/OutputFiles/' + self.Ftag +
                       '_MatterPowerSpectrum_Nbrane_{:.0e}_PressFac_{:.2e}_eCDM_{:.2e}.dat'.format(self.Nbrane,self.PressFac,self.eCDM),
                       np.column_stack((Tktab[:, 0], PS)))
        else:
            np.savetxt(path + '/OutputFiles/' + self.Ftag + '_MatterPowerSpectrum.dat', np.column_stack((Tktab[:, 0], PS)))
        return

    def TransferFuncs(self):
        if self.multiverse:
            Minfields = np.loadtxt(path + '/OutputFiles/' + self.Ftag +
                        '_FieldEvolution_{:.4e}_Nbrane_{:.0e}_PressFac_{:.2e}_eCDM_{:.2e}.dat'.format(self.kmin, self.Nbrane,
                        self.PressFac, self.eCDM))
        else:
            Minfields = np.loadtxt(path + '/OutputFiles/' + self.Ftag + '_FieldEvolution_{:.4e}.dat'.format(self.kmin))
        LargeScaleVal = Minfields[-1, 1]
        #kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
        Tktab = np.zeros_like(self.kgrid)
        for i,k in enumerate(self.kgrid):
            if self.multiverse:
                field =  np.loadtxt(path + '/OutputFiles/'+ self.Ftag +
                                    '_FieldEvolution_{:.4e}_Nbrane_{:.0e}_PressFac_{:.2e}_eCDM_{:.2e}.dat'.format(k, self.Nbrane,
                                    self.PressFac, self.eCDM))
            else:
                field =  np.loadtxt(path + '/OutputFiles/' + self.Ftag + '_FieldEvolution_{:.4e}.dat'.format(k))
            Tktab[i] = field[-1, 1] / LargeScaleVal
        return Tktab


    def Hubble(self, a):
        hubble = self.H_0 * np.sqrt(self.OM_R/a**4 + self.OM_M/a**3 + self.OM_Lam)
        return hubble

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def trapz(double[:] y, double[:] x):
    if len(x) != len(y):
        raise ValueError('x and y must be same length')
    cdef long npts = len(x)
    cdef double tot = 0
    cdef unsigned int i

    for i in range(npts-1):
        tot += 0.5*(y[i]+y[i+1])*(x[i+1]-x[i])
    return tot




