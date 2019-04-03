import os
from scipy.linalg import solve
from scipy.integrate import quad, odeint
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.special import zeta
from constants import *
import time
import cython
cimport numpy as cnp
import numpy as np
#import statsmodels.api as sm
#from matrix_build import *
#import warnings
#warnings.filterwarnings("error", category=UserWarning)

path = os.getcwd()

cdef extern from "math.h":
    double exp(double)
    double sqrt(double)
    double log(double)
    double log10(double)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.cdivision(True)
@cython.nonecheck(False)
class Universe(object):

    def __init__(self, k, omega_b, omega_cdm, omega_g, omega_L, omega_nu, accuracy=1e-3,
                 stepsize=0.01, lmax=5, testing=False, hubble_c=67.66, zreion=10):


        self.omega_b = omega_b
        self.omega_cdm = omega_cdm
        self.omega_g = omega_g
        self.omega_nu = omega_nu
        self.omega_M = omega_cdm + omega_b
        self.omega_R = omega_g + omega_nu
        self.omega_L = 1. - self.omega_M - self.omega_R
        self.H_0 = hubble_c / 2.998e5 # units Mpc^-1
        self.zreion = zreion

        self.n_bary = self.omega_b * rho_critical  * (hubble_c / 100.)**2.

        self.Lmax = lmax
        self.stepsize = stepsize

        self.k = k

        self.accuracy = accuracy
        self.TotalVars = 8 + 3*self.Lmax
        self.step = 0

        self.Theta_Dot = np.zeros(self.Lmax+1 ,dtype=object)
        self.Theta_P_Dot = np.zeros(self.Lmax+1 ,dtype=object)
        self.Neu_Dot = np.zeros(self.Lmax+1 ,dtype=object)

        self.combined_vector = np.zeros(self.TotalVars ,dtype=object)
        self.Psi_vec = []
        self.combined_vector[0] = self.Phi_vec = []
        self.combined_vector[1] = self.dot_rhoCDM_vec = []
        self.combined_vector[2] = self.dot_velCDM_vec = []
        self.combined_vector[3] = self.dot_rhoB_vec = []
        self.combined_vector[4] = self.dot_velB_vec = []
        for i in range(self.Lmax + 1):
            self.combined_vector[5+i*3] = self.Theta_Dot[i] = []
            self.combined_vector[6+i*3] = self.Theta_P_Dot[i] = []
            self.combined_vector[7+i*3] = self.Neu_Dot[i] = []

        self.compute_funcs()
        eta_matter_rad = self.scale_to_ct(5. * self.omega_R / self.omega_M)
        xc = max(eta_matter_rad * self.k, 1e3)
        self.gamma_supp = lambda x: 0.5 * (1. - np.tanh((x - xc) / 50.))
        self.tflip_RCA = 1.

        self.testing = testing
        if self.testing:
            self.aLIST = []
            self.etaLIST = []
            self.csLIST = []
            self.hubLIST = []
            self.dtauLIST = []
            self.xeLIST = []
            self.TbarLIST=[]

        return

    def compute_funcs(self, preload=False):
        self.Thermal_sln()
#        backgrounds = np.loadtxt(path + '/precomputed/LCDM_background.dat')
#        self.Tb = interp1d(np.log10(backgrounds[:,0]), np.log10(backgrounds[:,3]), kind='linear',
#                            bounds_error=False, fill_value='extrapolate')
#        short_b_xe = backgrounds[backgrounds[:,0] < 1., 2]
#        short_b = backgrounds[backgrounds[:,0] < 1., 0]
#        self.Xe = interp1d(np.log10(short_b), np.log10(short_b_xe), kind='cubic',
#                            bounds_error=False, fill_value=np.log10(1.16380))
#        self.Csnd_them = interp1d(np.log10(backgrounds[:,0]), backgrounds[:,4], kind='cubic',
#                          bounds_error=False, fill_value='extrapolate')
        cdef int i
        cdef cnp.ndarray[double] a0_init, eta_list
        if not os.path.isfile(path + '/precomputed/ln_a_CT_working.dat'):
            a0_init = np.logspace(-9, 0.1, 1e4)
            eta_list = np.zeros_like(a0_init)
            for i in range(len(a0_init)):
                eta_list[i] = self.conform_T(a0_init[i])
            np.savetxt(path + '/precomputed/ln_a_CT_working.dat', np.column_stack((a0_init, eta_list)))
        else:
            load_lna = np.loadtxt(path + '/precomputed/ln_a_CT_working.dat')
            eta_list = load_lna[:, 1]
            a0_init = load_lna[:, 0]

        self.eta_0 = self.conform_T(1.)
        self.ct_to_scaleI = interp1d(np.log10(eta_list), np.log10(a0_init), kind='cubic',
                                    bounds_error=True)
        self.scale_to_ctI = interp1d(np.log10(a0_init), np.log10(eta_list), kind='cubic',
                                    bounds_error=True)

        fileVis = path + '/precomputed/working_VisibilityFunc.dat'
        if os.path.exists(fileVis):
            visfunc = np.loadtxt(fileVis)
            self.Vfunc = interp1d(visfunc[:,0], visfunc[:,1], kind='cubic', fill_value=0., bounds_error=False)
        return

    def clearfiles(self):
        if os.path.isfile(path + '/precomputed/ln_a_CT_working.dat'):
            os.remove(path + '/precomputed/ln_a_CT_working.dat')

        if os.path.isfile(path + '/precomputed/xe_working.dat'):
            os.remove(path + '/precomputed/xe_working.dat')
        if os.path.isfile(path + '/precomputed/tb_working.dat'):
            os.remove(path + '/precomputed/tb_working.dat')

        if os.path.isfile(path + '/precomputed/working_expOpticalDepth.dat'):
            os.remove(path + '/precomputed/working_expOpticalDepth.dat')
        if os.path.isfile(path + '/precomputed/working_VisibilityFunc.dat'):
            os.remove(path + '/precomputed/working_VisibilityFunc.dat')

    def Thermal_sln(self):

        self.tb_fileNme = path + '/precomputed/tb_working.dat'
        self.Xe_fileNme = path + '/precomputed/xe_working.dat'
        if not os.path.isfile(self.tb_fileNme) or not os.path.isfile(self.Xe_fileNme):
            lgz = 4.5
            ionizing_he = True
            fst = True
            lgz_list = [lgz]
            xe_he_list = [1.16381]
            fhe = 0.245 / (4. * (1. - 0.245))
            while ionizing_he:
                lgz -= 0.01
                sln = fsolve(lambda x: self.saha(x, lgz, first=fst, helium=True), 1. + 2.*fhe)[0]
                lgz_list.append(lgz)
                xe_he_list.append(sln)
                if sln <= (fhe + 1. + 1e-3):
                    fst = False
                if sln <= (1. + 1e-3):
                    ionizing_he = False

            he_xe_tab = np.column_stack((1. / (1. + 10.**np.asarray(lgz_list)), np.asarray(xe_he_list)))
            he_xe_tab = he_xe_tab[10.**np.asarray(lgz_list) > 3e3]

            tvalsHe = np.linspace(3e3, 1e3, 1000)
            val_sln_he = odeint(lambda x,y: self.thermal_funcs(x,y,hydro=False), [fhe - 1e-4, 2.7255 * (1. + tvalsHe[0])], tvalsHe)

            he2_tab = np.asarray([[1. / (1. + tvalsHe[i]), val_sln_he[i,0]] for i in range(len(val_sln_he)) if val_sln_he[i, 0] > 1e-6])

            tvals = np.linspace(3e3, 1e-2, 10000)
            y0 = [0.99999, 2.7255 * (1. + tvals[0])]
            val_sln = odeint(self.thermal_funcs, y0, tvals)
            avals = 1. / (1. + tvals)

            fhe = 0.16381 / 2.
            tanhV = (fhe + 1.) / 2. * (1. + np.tanh( 2.*((1.+self.zreion)**(3./2.) - (1.+ tvals)**1.5 ) / (3.*0.5*np.sqrt(1.+ tvals)) ))
            zreionHE = 3.5
            tanhV += fhe / 2. * (1. + np.tanh( 2.*((1.+zreionHE)**(3./2.) - (1.+ tvals)**1.5 ) / (3.*0.5*np.sqrt(1.+ tvals)) ))

            val_sln[:,0] += tanhV
            he2_tab_inc = 10.**interp1d(np.log10(he2_tab[:,0]), np.log10(he2_tab[:,1]), fill_value=-100., bounds_error=False, kind='cubic')(np.log10(avals))
            self.Xe_dark = np.vstack((he_xe_tab, np.column_stack((avals, val_sln[:,0] + he2_tab_inc))))

            tvals2 = np.linspace(1e4, 1e-2, 10000)
            termpB = odeint(self.recast_Tb, 2.7255 * (1. + tvals2[0]), tvals2,
                           args=(self.Xe_dark, ))

            self.Tb_drk = np.column_stack((1. / (1. + tvals2), termpB))

            np.savetxt(self.tb_fileNme, self.Tb_drk)
            np.savetxt(self.Xe_fileNme, self.Xe_dark)
        else:
            try:
                self.Tb_drk = np.loadtxt(self.tb_fileNme)
                self.Xe_dark = np.loadtxt(self.Xe_fileNme)
            except:
                print('fail to load xe and tb dark')
                raise ValueError

        self.Tb = interp1d(np.log10(self.Tb_drk[:,0]), np.log10(self.Tb_drk[:,1]), kind='cubic',
                            bounds_error=False, fill_value=0.)
        self.Xe = interp1d(np.log10(self.Xe_dark[:,0]), np.log10(self.Xe_dark[:,1]), kind='cubic',
                            bounds_error=False, fill_value=np.log10(1.16381))

        return

    def saha(self, xe, lgz, first=True, helium=True):
        tg = 2.7255 * (1. + 10.**lgz) * kboltz
        fhe = 0.16381 / 2.
        hh = 4.135e-15
        if first and helium:
            ep0 = 54.4
            lhs = (xe - 1. - fhe) * xe
            rhf = np.exp(- ep0 / tg) * (1. + 2.*fhe - xe)
        elif helium and not first:
            ep0 = 24.6
            lhs = (xe - 1.)*xe
            rhf = 4. * np.exp(- ep0 / tg) * (1. + fhe - xe)
        else:
            ep0 = 13.6
            lhs = xe**2.
            rhf = np.exp(- ep0 / tg) * (1. - xe)

        rhpre = (2. * np.pi * 5.11e5 * tg)**(3./2.) / (self.n_bary * (1. + 10.**lgz)**3. * hh**3.)
        units = (1. / 2.998e10)**3.
        return lhs - units * rhpre * rhf

    def thermal_funcs(self, val, z, hydro=True):
        xe, T = val
        if hydro:
            return [self.xeDiff([xe], z, T)[0], self.dotT_normal([T], z, xe)]
        else:
            return [self.xeDiff_he([xe], z, T)[0], self.dotT_normal([T], z, xe)]

    def heliumI(self, val, z):
        return self.xeDiff_he(val, z)

    def recast_Tb(self, val, z, xe_list):
        xe_inter = interp1d(xe_list[:,0], np.log10(xe_list[:,1]), kind='linear',
                            bounds_error=False, fill_value=np.log10(1.16381))
        return self.dotT_normal(val, z, 10.**xe_inter(z))

    def dotT_normal(self, T, z, xe):
        # d log (Tb) / d z
        cdef double thompson_xsec = 6.65e-25 # cm^2
        cdef double aval = 1. / (1. + z)
        cdef double Yp = 0.245
        cdef double Mpc_to_cm = 3.086e24
        cdef double mol_wei = (1. + 4. * Yp) / (1. + 2. * Yp + xe) * 0.931

        cdef double n_b = self.n_bary * (1. + z)**3. * Yp
        cdef double hub = self.hubble(aval)
        cdef double omega_Rat = self.omega_g / self.omega_b

        return (2. * T[0] * aval  - (1./hub)*(8./3.)*(mol_wei/5.11e-4) *
                omega_Rat * xe * n_b * thompson_xsec * (2.7255*(1.+z) - T[0])*Mpc_to_cm)

    def dotT(self, T, lgz, xe, a):
        # d ln (T) / d ln (a)
        cdef double thompson_xsec = 6.65e-25 # cm^2
        cdef double aval = 1. / (1. + 10.**lgz)
        cdef double Yp = 0.245
        cdef double Mpc_to_cm = 3.086e24
        cdef double mol_wei = (1. + 4. * Yp) / (1. + 2. * Yp + xe) * 0.931
        cdef double n_b = self.n_bary * (1.+10.**lgz)**3.
        cdef double hub = self.hubble(aval)
        cdef double omega_Rat = self.omega_g / self.omega_b

        return (-2. + (1./ (hub * aval * T))*(8./3.)*(mol_wei/5.11e-4) *
                omega_Rat * xe * n_b * thompson_xsec * (2.7255*(1.+10.**lgz) - T)*Mpc_to_cm)

    def Cs_Sqr(self, a):
        cdef double kb = 8.617e-5/1e9 # GeV/K
        cdef double thompson_xsec = 6.65e-25 # cm^2
        cdef double facxe = 10.**self.Xe(log10(a))
        cdef double Yp = 0.245
        cdef double epsil = 1e-3
        cdef double mol_wei
        mol_wei = (1. + 4. * Yp) / (1. + 2. * Yp + facxe) * 0.931
        cdef double Tb
        if (1./a - 1.) < 1e4:
            Tb = 10.**self.Tb(log10(a))
            Tb2 = 10.**self.Tb(log10(a) - epsil)
            tbderiv = (log10(Tb) - log10(Tb2)) / epsil * log(10.)
        else:
            tbderiv = -2.
            Tb = 2.7225 / a

        cdef double val_r = kb * Tb / mol_wei * (1. - 1./3. * tbderiv)
        return val_r

    def xeDiff(self, val, y, tgas):
        ep0 =  10.2343 # eV
        epG =  13.6 # eV


        kb = 8.617e-5 # ev/K
        Mpc_to_cm = 3.086e24
        me = 5.11e5 # eV
        aval = 1. / (1.+y)
        Yp = 0.245
        n_b = self.n_bary / aval**3.
        hub = self.hubble(aval)
        alphaH = 1.14 * 1e-19 * 4.309 * (tgas/1e4)**-0.6166 / (1. + 0.6703 * (tgas/1e4)**0.53) * 1e2**3. / 2.998e10 # cm^2

        beta = alphaH*(2.*np.pi*me*kb*tgas/4.135e-15**2.)**(3./2.) / 2.998e10**3. # 1/cm

        kh = (121.5682e-9)**3./(8.*np.pi*hub) * 1e6 # cm^3 * Mpc
        lambH = 8.22458 / 2.998e10 * Mpc_to_cm # 1 / Mpc
        preFac = (1. + kh *lambH * n_b * (1. - Yp) * (1 - val[0]))  # unitless
        preFac /= (1. + kh * (lambH+beta*Mpc_to_cm*np.exp(-(epG - ep0)/(kb*tgas))) * n_b * (1. - Yp) * (1 - val[0])) # Unitless

        return [preFac*aval/hub*(-(1.-val[0])*beta*np.exp(-epG/(kb*tgas)) + val[0]**2.*n_b*(1.-Yp)*alphaH)*Mpc_to_cm]

    def xeDiff_he(self, val, y, tgas):
        # tgas = 2.7225 * (1. + y)
        ep0 =  24.6 # eV
        nu_2s = 20.6 # 2.998e8 / 60.1404e-9 * 4.135e-15
        nu_2p = 2.998e8 / 58.4334e-9 * 4.135e-15
        nu_diff2 =  (nu_2p - ep0)
        kb = 8.617e-5 # ev/K
        Mpc_to_cm = 3.086e24
        me = 5.11e5 # eV
        aval = 1. / (1.+y)
        Yp = 0.245
        fhe = Yp / (4. * (1. - Yp))
        n_b = self.n_bary / aval**3.
        hub = self.hubble(aval)
        alphaH = 10.**-16.744 / (np.sqrt(tgas / 3.)*(1.+tgas/3.)**(1.-0.711)*(1.+tgas/10.**5.114)**(1.+0.711)) * 1e6 / 2.998e10 # cm^2
        beta = alphaH*(2.*np.pi*me*kb*tgas/4.135e-15**2.)**(3./2.) / 2.998e10**3 * np.exp(-(ep0 - nu_2s) /(kb*tgas))  # 1/cm

        kh = (58.4334e-9)**3./(8.*np.pi*hub) * 1e6 # cm^3 * Mpc
        lambH = 51.3 / 2.998e10 * Mpc_to_cm # 1 / Mpc
        preFac = (1. + kh *lambH * n_b * (1-Yp) * (fhe - val[0]) * np.exp(- nu_diff2 / (kb*tgas)))  # unitless
        preFac /= (1. + kh * (lambH + beta * Mpc_to_cm) * n_b * (1-Yp) * (fhe - val[0]) * np.exp(- nu_diff2 / (kb*tgas))) # Unitless

        return [preFac*aval/hub*(val[0]*(1. + val[0])*n_b*(1-Yp)*alphaH  -  (fhe - val[0])*beta*np.exp(-nu_2s / (kb*tgas)))*Mpc_to_cm]


    def tau_functions(self):
        self.fileN_optdep = path + '/precomputed/working_expOpticalDepth.dat'
        self.fileN_visibil = path + '/precomputed/working_VisibilityFunc.dat'
        cdef double Mpc_to_cm = 3.086e24
        cdef double Yp, n_b, thompson_xsec, hubbs, xevals
        cdef cnp.ndarray[double] avals, tau, etavals, dtau, vis
        cdef int i
        if not os.path.isfile(self.fileN_visibil) or not os.path.isfile(self.fileN_optdep):
            print('File not found... calculating...')
            avals = np.logspace(-4.5, 0, 10000)
            Yp = 0.245
            thompson_xsec = 6.65e-25 # cm^2
            tau = np.zeros_like(avals)
            etavals = np.zeros_like(avals)
            dtau = np.zeros_like(avals)
            vis = np.zeros_like(avals)
            n_b = self.n_bary
            for i in range(len(avals)):
                xevals = 10.**self.Xe(log10(avals[i]))
                dtau[i] = -xevals * (1. - Yp) * n_b * thompson_xsec * Mpc_to_cm / avals[i]**2.
                etavals[i] = 10.**self.scale_to_ctI(log10(avals[i]))
            for i in range(len(avals) - 1):
                tau[i + 1] = np.trapz(-dtau[i:], etavals[i:])
                vis[i + 1] = -dtau[i + 1] * exp(-tau[i + 1])
            tau[0] = tau[1]
            np.savetxt(self.fileN_optdep, np.column_stack((avals, np.exp(-tau))))
            np.savetxt(self.fileN_visibil, np.column_stack((avals, vis)))

        return

    def init_conds(self, eta_0, aval):
        cdef double OM = self.omega_M * self.H_0**2./self.hubble(aval)**2./aval**3.
        cdef double OR = self.omega_R * self.H_0**2./self.hubble(aval)**2./aval**4.
        cdef double ONu = self.omega_nu * self.H_0**2./self.hubble(aval)**2./aval**4.
        cdef double rfactor = ONu / (0.75*OM*aval + OR)
        cdef double HUB = self.hubble(aval)

        self.inital_perturb = -1./6.
        cdef double initP = self.inital_perturb
        cdef int i

        self.Psi_vec.append(initP)
        self.Phi_vec.append(-(1.+2.*rfactor/5.)*initP)
        self.dot_rhoCDM_vec.append(-3./2.*initP)
        self.dot_velCDM_vec.append(1./2.*eta_0*self.k*initP)
        self.dot_rhoB_vec.append(-3./2.*initP)
        self.dot_velB_vec.append(1./2*eta_0*self.k*initP)

        self.Theta_Dot[0].append(-1./2.*initP)
        self.Theta_Dot[1].append(1./6.*eta_0*self.k*initP)
        self.Neu_Dot[0].append(-1./2.*initP)
        self.Neu_Dot[1].append(1./6.*eta_0*self.k*initP)
        self.Neu_Dot[2].append(1./30.*(self.k*eta_0)**2.*initP)

        for i in range(self.Lmax + 1):
            if i > 1:
                self.Theta_Dot[i].append(0.)
            self.Theta_P_Dot[i].append(0.)
            if i > 2:
                self.Neu_Dot[i].append(0.)

        self.step = 0
        return


    def solve_system(self, compute_TH):
        self.timeT = 0.
        cdef double eta_st = np.min([1e-3/self.k, 1e-1/0.7]) # Initial conformal time in Mpc

        cdef double y_st = log(self.ct_to_scale(eta_st))
        self.init_conds(eta_st, exp(y_st))
        self.eta_vector = [eta_st]
        self.y_vector = [y_st]


        cdef int try_count = 0
        cdef int try_max = 40
        FailRUN = False
        last_step_up = False

        cdef double y_use, eta_use, y_diff, test_epsilon, a_use
        cdef int i


        while (self.eta_vector[-1] < (self.eta_0 - 1.)):

            if try_count > try_max:
                print('FAIL TRY MAX....Breaking.')
                FailRUN=True
                return
            y_use = self.y_vector[-1] + self.stepsize
            eta_use = self.scale_to_ct(exp(y_use))
            if (eta_use > self.eta_0):
                eta_use = self.eta_0
                y_use = 1.
            self.eta_vector.append(eta_use)

            y_diff = y_use - self.y_vector[-1]
            if y_diff < 1e-10:
                print('ydiff is 0... failing...')
                return

            self.y_vector.append(y_use)
            a_use = exp(y_use)

#            if self.step%5000 == 0:
#                print('Last a: {:.7e}, New a: {:.7e}'.format(exp(self.y_vector[-2]), a_use))
            if ((y_diff > eta_use*a_use*self.hubble(a_use)) or
                (y_diff > a_use*self.hubble(a_use)/self.k)):
                self.stepsize *= 0.5
                self.eta_vector.pop()
                self.y_vector.pop()
                continue

            self.step_solver(y_use, eta_use, a_use)

            test_epsilon = abs(self.epsilon_test(a_use))
#            print(self.eta_vector[-1], np.exp(self.y_vector[-1]), self.y_vector[-1], self.stepsize, test_epsilon)

            if test_epsilon > self.accuracy:
                self.stepsize *= 0.5
                a_target = self.y_vector[-1] - 0.05
                if self.y_vector[0] >= a_target:
                    a_target += 0.02
                jchk = -1
                while (self.y_vector[jchk]  > a_target) and (len(self.y_vector) > 1):
                    self.eta_vector.pop()
                    self.y_vector.pop()
                    self.Psi_vec.pop()
                    for i in range(self.TotalVars):
                        self.combined_vector[i].pop()
                    jchk -= 1
                try_count += 1
                continue
            self.step += 1
            if (test_epsilon < 1e-3*self.accuracy) and not last_step_up:
                self.stepsize *= 1.0
                last_step_up = True
                #print 'Increase Step Size'
            else:
                last_step_up = False
            try_count = 0

        if not FailRUN and self.testing:
            print('Saving File...')
            self.save_system()

        if not compute_TH:
            return self.combined_vector[0][-1]

        sources = np.zeros((len(self.eta_vector), 4))
        cdef double aval, phi_term_back, psi_term_back, eta_back
        sources[:, 0] = self.eta_vector

        pi_polar = [self.combined_vector[6][i] + self.combined_vector[11][i] + self.combined_vector[12][i] for i in range(len(self.eta_vector))]

        sources[:, 2] = np.asarray(self.combined_vector[4])
        der2_pi = np.zeros_like(sources[:, 0])

        for i in range(len(self.eta_vector)):
            aval = self.ct_to_scale(sources[i, 0])
            if (i > 1) and (i < len(self.eta_vector) - 1):
                h2 = sources[i + 1, 0] - sources[i, 0]
                h1 = sources[i, 0] - sources[i - 1, 0]
                if (h1 != 0.) and (h2 != 0.):
                    vis1 = self.Vfunc(self.ct_to_scale(sources[i, 0]))
                    vis2 = self.Vfunc(self.ct_to_scale(sources[i + 1, 0]))
                    vis0 = self.Vfunc(self.ct_to_scale(sources[i - 1, 0]))
                    der2_pi[i] = 2.*(h2*(pi_polar[i + 1]*vis2) - (h1+h2)*(pi_polar[i]*vis1) +
                            h1*(pi_polar[i - 1]*vis0))/(h1*h2*(h1+h2))
                    der2_pi[i] *= 3. / (4. * self.k**2.)
            if i > 0:
                psi_term_back = self.Psi_vec[i - 1]
                phi_term_back = self.combined_vector[0][i - 1]
                eta_back = self.eta_vector[i - 1]
            else:
                psi_term_back = 0.
                phi_term_back = 0.
                eta_back = 0.

            sources[i, 3] = ((self.Psi_vec[i] - psi_term_back) - (self.combined_vector[0][i] - phi_term_back)) / \
                    (self.eta_vector[i] - eta_back)
        sources[:, 1] = [self.combined_vector[5][i] + self.Psi_vec[i] for i in range(len(self.eta_vector))]
#        sources[:, 1] = [self.combined_vector[5][i] + self.Psi_vec[i] + pi_polar[i] / 4. +
#                            der2_pi[i] for i in range(len(self.eta_vector))]

        source_interp = np.zeros(( len(global_a_list), 4 ))
        s_int_1 = interp1d(sources[:, 0], sources[:, 1], kind='cubic', fill_value=0., bounds_error=False)
        s_int_2 = interp1d(sources[:, 0], sources[:, 2], kind='cubic', fill_value=0., bounds_error=False)
        s_int_3 = interp1d(sources[:, 0], sources[:, 3], kind='cubic', fill_value=0., bounds_error=False)
        source_interp[:, 0] = self.scale_to_ct(global_a_list)
        source_interp[:,1] = s_int_1(source_interp[:, 0])
        source_interp[:,2] = s_int_2(source_interp[:, 0])
        source_interp[:,3] = s_int_3(source_interp[:, 0])

        return source_interp

    def step_solver(self, double lna, double eta, double aval):
        cdef double tau_n

        if self.step > 0:
            tau_n = (lna - self.y_vector[-2]) / (self.y_vector[-2] - self.y_vector[-3])
        else:
            tau_n = (lna - self.y_vector[-2]) / self.y_vector[-2]

        cdef double delt = (lna - self.y_vector[-2])
        cdef cnp.ndarray[double, ndim=2] Jmat = self.matrix_J(eta, aval)
        ysol = solve((1.+2.*tau_n)/(1.+tau_n)*np.eye(self.TotalVars) - delt*Jmat, self.b_vector(tau_n),
                     overwrite_a=True, overwrite_b=True, check_finite=False)

        for i in range(self.TotalVars):
            self.combined_vector[i].append(ysol[i])
        self.Psi_vec.append(-12.*(aval**2./self.k**2.*(self.rhoNeu(aval)*self.combined_vector[13][-1] +
                    self.rhoG(aval)*self.combined_vector[11][-1])) - self.combined_vector[0][-1])
        return

    def b_vector(self, tau):
        cdef cnp.ndarray[double] bvec = np.zeros(self.TotalVars)
        cdef int i
        for i in range(self.TotalVars):
            if self.step == 0:
                bvec[i] = (1.+tau)*self.combined_vector[i][-1]
            else:
                bvec[i] = (1.+tau)*self.combined_vector[i][-1] - tau**2./(1.+tau)*self.combined_vector[i][-2]
        return bvec

    def matrix_J(self, double eta, double a_val):
        cdef double CsndB
        cdef cnp.ndarray[double, ndim=2] Jma = np.zeros((self.TotalVars, self.TotalVars))
        cdef double RR = (4.*self.rhoG(a_val))/(3.*self.rhoB(a_val))
        cdef double HUB = self.hubble(a_val)
        cdef double dTa = -10.**self.Xe(log10(a_val))*(1. - 0.245)*self.n_bary*6.65e-29*1e4/a_val**2./3.24078e-25
        if a_val > 1e-4:
            CsndB = self.Cs_Sqr(a_val)
        else:
            CsndB = self.Cs_Sqr(1e-4) * 1e-4 / a_val
#            CsndB =  self.Csnd_them(log10(1e-4)) * 1e-4 / a_val
        cdef double rG = self.rhoG(a_val)
        cdef double rN = self.rhoNeu(a_val)
        cdef double rB = self.rhoB(a_val)
        cdef double rC = self.rhoCDM(a_val)
        cdef gammaSup = self.gamma_supp(self.k * eta)

        if self.testing:
            self.aLIST.append(a_val)
            self.etaLIST.append(eta)
            self.hubLIST.append(HUB)
            self.csLIST.append(CsndB)
            self.dtauLIST.append(dTa)
            self.xeLIST.append(10.**self.Xe(log10(a_val)))
            self.TbarLIST.append(10.**self.Tb(log10(a_val)))

        if np.abs(HUB * a_val / dTa) < 1e-2 and np.abs(self.k / dTa) < 1e-2:
            tflip_TCA = True
        else:
            tflip_TCA = False

        cdef cnp.ndarray[double] PsiTerm = np.zeros(self.TotalVars)
        PsiTerm[0] += -1.
        PsiTerm[11] += -12.*(a_val/self.k)**2.*rG
        PsiTerm[13] += -12.*(a_val/self.k)**2.*rN

        # Phi Time derivative
        Jma[0,:] += PsiTerm
        Jma[0, 0] += -((self.k/(HUB*a_val))**2.)/3.
        Jma[0, 1] += 1./(HUB**2.*2.)*rC
        Jma[0, 3] += 1./(HUB**2.*2.)*rB
        Jma[0, 5] += 2./(HUB**2.)*rG
        Jma[0, 7] += 2./(HUB**2.)*rN

        # CDM density
        Jma[1,2] += -self.k/(HUB*a_val)
        Jma[1,:] += -3.*Jma[0,:]

        # CDM velocity
        Jma[2,2] += -1.
        Jma[2,:] += self.k/(HUB*a_val)*PsiTerm

        # Baryon density
        Jma[3,4] += -self.k / (HUB*a_val)
        Jma[3,:] += -3.*Jma[0,:]

        # Theta 0
        Jma[5,8] += -self.k / (HUB*a_val) * gammaSup
        Jma[5,:] += -Jma[0,:] * gammaSup

        # Baryon velocity
        if not tflip_TCA:
            Jma[4,4] += -1. + dTa * RR / (HUB*a_val)
            Jma[4,:] += self.k/(HUB*a_val)*PsiTerm
            Jma[4,3] += self.k * CsndB / (HUB * a_val)
            Jma[4,8] += -3.*dTa * RR / (HUB * a_val)
        else:
            Jma[4,4] += -1./(1.+RR) + 2.*(RR/(1.+RR))**2. + 2.*RR*HUB*a_val/\
                        ((1.+RR)**2.*dTa)
            Jma[4,3] += CsndB*self.k/(HUB*a_val*(1.+RR))
            Jma[4,5] += RR*self.k*(1./(HUB*a_val*(1+RR)) +
                        2./((1.+RR)**2.*dTa))
            Jma[4,11] += -RR*self.k/(2.*HUB*a_val*(1+RR))
            Jma[4,8] += -6.*(RR/(1.+RR))**2.
            Jma[4,:] += (self.k/(HUB*a_val) + RR*self.k / (dTa*(1.+RR)**2.))* PsiTerm
            Jma[4,:] += -(RR*self.k/(dTa*(1.+RR)**2.)) * CsndB*Jma[3,:]
            Jma[4,:] += (RR*self.k/(dTa*(1.+RR)**2.)) * Jma[5,:]

        # ThetaP 0
        Jma[6,6] += dTa / (2.*HUB*a_val) * gammaSup
        Jma[6,11] += - dTa / (2.*HUB*a_val) * gammaSup
        Jma[6,12] += - dTa / (2.*HUB*a_val) * gammaSup
        Jma[6,9] += - self.k / (HUB*a_val) * gammaSup

        # Neu 0
        Jma[7,10] += -self.k / (HUB*a_val) * gammaSup
        Jma[7,:] += -Jma[0,:] * gammaSup

        # Theta 1
        if not tflip_TCA:
            if a_val < self.tflip_RCA:
                Jma[8,11] += -2.*self.k / (3.*HUB*a_val) * gammaSup
                Jma[8,8] += dTa / (HUB*a_val) * gammaSup
                Jma[8,4] += -dTa / (3.*HUB*a_val) * gammaSup
            Jma[8,5] += self.k / (3.*HUB*a_val) * gammaSup
            Jma[8,:] += self.k*PsiTerm / (3.*HUB*a_val) * gammaSup
        else:
            Jma[8,4] += -1./(3.*RR) * gammaSup
            Jma[8,3] += CsndB*self.k/(HUB*a_val*RR*3.) * gammaSup
            Jma[8,5] += self.k/(3.*HUB*a_val) * gammaSup
            Jma[8,11] += -self.k/(6.*HUB*a_val) * gammaSup
            Jma[8,:] += (1.+RR)*self.k/(3.*RR*HUB*a_val)*PsiTerm * gammaSup
            Jma[8,:] += -Jma[4,:]/(3.*RR) * gammaSup

        # ThetaP 1
        Jma[9,6] += self.k / (3.*HUB*a_val) * gammaSup
        Jma[9,12] += -2.*self.k / (3.*HUB*a_val) * gammaSup
        Jma[9,9] += dTa / (HUB*a_val) * gammaSup
        # Neu 1
        Jma[10,:] += self.k * PsiTerm / (3.*HUB*a_val) * gammaSup
        Jma[10,7] += self.k / (3.*HUB*a_val) * gammaSup
        Jma[10,13] += -2.*self.k/ (3.*HUB*a_val) * gammaSup

        cdef int i, elV, inx

        # Theta 2
        Jma[11,8] += 2.*self.k / (5.*HUB*a_val) * gammaSup
        Jma[11,14] += -3.*self.k / (5.*HUB*a_val) * gammaSup
        Jma[11,11] += 9.*dTa / (10.*HUB*a_val) * gammaSup
        Jma[11,6] += -dTa / (10.*HUB*a_val) * gammaSup
        Jma[11,12] += -dTa /(10.*HUB*a_val) * gammaSup
        # ThetaP 2
        Jma[12,9] += 2.*self.k / (5.*HUB*a_val) * gammaSup
        Jma[12,15] += -3.*self.k / (5.*HUB*a_val) * gammaSup
        Jma[12,12] += 9.*dTa / (10.*HUB*a_val) * gammaSup
        Jma[12,11] += -dTa / (10.*HUB*a_val) * gammaSup
        Jma[12,6] += -dTa / (10.*HUB*a_val) * gammaSup

        # Neu 2
        Jma[13,10] += 2.*self.k/ (5.*HUB*a_val) * gammaSup
        Jma[13,16] += -3.*self.k/ (5.*HUB*a_val) * gammaSup

        for i in range(14, 14 + self.Lmax - 3):
            elV = i - 14 + 3
            inx = i - 14
            # Photons
            Jma[14+3*inx,14+3*inx] += dTa / (HUB*a_val) * gammaSup
            Jma[14+3*inx,14+3*inx-3] += self.k*elV/((2.*elV + 1.)*(HUB*a_val)) * gammaSup
            Jma[14+3*inx,14+3*inx+3] += -self.k*(elV+1.)/((2.*elV + 1.)*(HUB*a_val)) * gammaSup

            # Neutrinos
            Jma[14+3*inx+2,14+3*inx+2-3] += self.k*elV/((2.*elV + 1.)*(HUB*a_val)) * gammaSup
            Jma[14+3*inx+2,14+3*inx+2+3] += -self.k*(elV+1.)/((2.*elV + 1.)*(HUB*a_val)) * gammaSup

            # Polarization
            Jma[14+3*inx+1,14+3*inx+1] += dTa / (HUB*a_val) * gammaSup
            Jma[14+3*inx+1,14+3*inx+1-3] += self.k*elV/((2.*elV + 1.)*(HUB*a_val)) * gammaSup
            Jma[14+3*inx+1,14+3*inx+1+3] += -self.k*(elV+1.)/((2.*elV + 1.)*(HUB*a_val)) * gammaSup


            # Theta Lmax
            Jma[-3, -3-3] += self.k / (HUB*a_val) * gammaSup
            Jma[-3, -3] += (-(self.Lmax+1.)/eta + dTa) / (HUB*a_val) * gammaSup
            # ThetaP Lmax
            Jma[-2, -2-3] += self.k / (HUB*a_val) * gammaSup
            Jma[-2, -2] += (-(self.Lmax+1.)/eta + dTa) / (HUB*a_val) * gammaSup
            # Nu Lmax
            Jma[-1, -1-3] += self.k / (HUB*a_val) * gammaSup
            Jma[-1, -1] += -(self.Lmax+1.)/(eta*HUB*a_val) * gammaSup

        return Jma

    def scale_to_ct(self, scale):
        return 10.**self.scale_to_ctI(np.log10(scale))

    def ct_to_scale(self, ct):
        return 10.**self.ct_to_scaleI(np.log10(ct))

    def scale_a(self, eta):
        return self.ct_to_scale(eta)

    def conform_T(self, a):
        return quad(lambda x: 1./self.H_0 / x**2. / np.sqrt(self.omega_R/x**4.+self.omega_M/x**3.+self.omega_L),
                    0., a, epsabs=1e-10, limit=100)[0]

    def get_H0(self, tuni):
        units = 1. / 3.154e7 / 2.998e8 / 3.24078e-23
        return quad(lambda x: 1. / tuni / x / np.sqrt(self.omega_R/x**4.+self.omega_M/x**3.+self.omega_L),
                    0., 1., epsabs=1e-6, limit=30)[0] * units

    def hubble(self, a):
        return self.H_0*np.sqrt(self.omega_R*a**-4.+self.omega_M*a**-3.+self.omega_L)

    def rhoCDM(self, a):
        return self.omega_cdm * self.H_0**2. * a**-3.

    def rhoB(self, a):
        return self.omega_b * self.H_0**2. * a**-3.

    def rhoG(self, a):
        return self.omega_g * self.H_0**2. * a**-4.

    def rhoNeu(self, a):
        return self.omega_nu * self.H_0**2. * a**-4.

    def epsilon_test(self, a):


        cdef double denom = (self.omega_M*a**-3. + self.omega_R*a**-4. + self.omega_L)
        cdef double phiTerm = -2./3.*(self.k/(a*self.H_0))**2.*self.combined_vector[0][-1]
        cdef double denTerm = (self.omega_cdm*self.combined_vector[1][-1]+self.omega_b*self.combined_vector[3][-1])*a**-3. +\
                  4.*(self.omega_g*self.combined_vector[5][-1]+self.omega_nu*self.combined_vector[7][-1])*a**-4.
        cdef double velTerm = 3.*a*self.hubble(a)/self.k*(
                 (self.omega_cdm*self.combined_vector[2][-1]+self.omega_b*self.combined_vector[4][-1])*a**-3. +
                 4.*(self.omega_g*self.combined_vector[8][-1]+self.omega_nu*self.combined_vector[10][-1])*a**-4.)
        return (phiTerm + denTerm + velTerm)/denom



    def save_system(self):

        if self.testing:
            psi_term = np.zeros(len(self.eta_vector))
            for i in range(len(self.eta_vector)):
                aval = 10.**self.ct_to_scale(np.log10(self.eta_vector[i]))

            sve_tab = np.zeros((len(self.eta_vector), self.TotalVars+2))
            sve_tab[:,0] = self.eta_vector
            sve_tab[:,-1] = self.Psi_vec
            for i in range(self.TotalVars):
                sve_tab[:,i+1] = self.combined_vector[i]
            np.savetxt(path + '/OutputFiles/StandardUniverse_FieldEvolution_{:.4e}.dat'.format(self.k), sve_tab, fmt='%.8e', delimiter='    ')
            np.savetxt(path+'/OutputFiles/StandardUniverse_Background.dat',
                        np.column_stack((self.aLIST, self.etaLIST, self.xeLIST, self.hubLIST, self.csLIST, self.dtauLIST, self.TbarLIST)))
        return


def interp1(double[:] x, double[:] y, double x0):
    cdef int i = 0
    cdef double res

    if x0 > x[-1]:
        return y[-1]
    if x0 < x[0]:
        return y[0]

    while (x[i] < x0) and (i <= len(x)):
        i = i + 1

    if x[i] == x0:
        return y[i]
    if i == 0:
        return y[i] - (y[i+1] - y[i]) * (x0 - x[i]) / (x[i+1] - x[i])

    res = y[i-1] + (y[i] - y[i-1]) * (x0 - x[i-1]) / (x[i] - x[i-1])
    return res
