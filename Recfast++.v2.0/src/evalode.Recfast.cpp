//========================================================================================
// Author: Jens Chluba & Luke Hart
// First implementation: Oct 2010
// CITA, University of Toronto
// All rights reserved.
//========================================================================================
// May     2018: A2s1s control now as parameter that is passed on [JC]
// April   2018: restructured code variables and functions
// March   2017: added control over photoionization evaluation [JC]
// Oct-Dec 2016: added effects for variations of alpha and me/mp
// Oct 11  2014: added switch that allows to change T=TR instead of T=TM for beta_ion
// Oct 10  2014: added heating by magnetic fields

//========================================================================================
// This module contains the evaluation of the RECFAST ODE-system. The code was initially
// based on the C-version for RECFAST from Seager 1999
//========================================================================================

#include <cmath>
#include <iostream>
#include <cstdlib>

#include "Recfast++.h"
#include "cosmology.Recfast.h"
#include "evalode.Recfast.h"
#include "DM_annihilation_decay.Recfast.h"
#include "Magnetic_fields.Recfast.h"
#include "Variation_constants.Recfast.h"

using namespace std;
using namespace RECFAST_physical_constants; // defined in constants.Recfast.h
using namespace RECFAST_atomic_constants;   // defined in constants.Recfast.h
using namespace Recfastpp_Cosmology;        // defined in cosmology.Recfast.h

//========================================================================================
// Global Variables; defined in Recfast++.h
//========================================================================================
RecPars RecVars;

double RECFAST_atomic_constants::RF_Lam2s1sH=8.22458;           // Recfast default value

void Set_evalode_Vars(const RecPars &RecInputs)
{ RecVars=RecInputs; RECFAST_atomic_constants::RF_Lam2s1sH=RecInputs.A2s1s;}

const RecPars& Get_evalode_Vars(){ return RecVars; }

//========================================================================================
// Boltzmann factor
//========================================================================================
double Boltzmann(double gj, double gi, double E, double T )
{ return max(1.0e-300, ( (gj/gi) * exp(-E/(RF_kBoltz*T)) )); }

//========================================================================================
// Saha-Boltzmann factor for detailed balance relations
//========================================================================================
double SahaBoltz(double gi, double gc, double ge, double E_ion, double T)
{
    double c1 = (RF_hPlanck/(2.0*RF_PI*RF_mElect)) * (RF_hPlanck/RF_kBoltz);
    return (ge * gi/(2.0*gc) * pow(c1, 1.5) * pow(T, -1.5)
               * exp(min(680.0, E_ion/(RF_kBoltz*T))) );
}

//========================================================================================
// case B recombination rate for Hydrogen (in m^3 / s) including fudge factor [F~1.14]
//========================================================================================
double alphaH_func(double TM)
{
  double a1 = 4.309, a2 = -0.6166, a3=0.6703, a4=0.5300, t=TM/1.0e+4;
  return RecVars.F * a1*1.0e-19*pow(t, a2)/(1.0+a3*pow(t, a4));
}

//----------------------------------------------------------------------------------------
// photo-ionization rate HI where T=TM or Tr depending on the setting of 'eval_ion_Tr'
//----------------------------------------------------------------------------------------
double betaH_func(double T)
{ return alphaH_func(T)/SahaBoltz(2.0, 1.0, 1.0, RF_EionH2s, T); }

//========================================================================================
// case B recombination rate for neutral Helium (in m^3 / s)
//========================================================================================
double alphaHe_func(double TM)
{
  double a1 = pow(10.0, -16.744), a2 = 0.711, T0=3.0, T1=pow(10.0, 5.114);
  double sqrt_TMT0=sqrt(TM/T0), sqrt_TMT1=sqrt(TM/T1); 
  return a1/(sqrt_TMT0*pow(1.0+sqrt_TMT0, 1.0-a2)*pow(1.0+sqrt_TMT1, 1.0+a2));
}

//----------------------------------------------------------------------------------------
// photo-ionization rate HeI where T=TM or Tr depending on the setting of 'eval_ion_Tr'
//----------------------------------------------------------------------------------------
double betaHe_func(double T)
{ return alphaHe_func(T)/SahaBoltz(1.0, 2.0, 1.0, RF_EionHe2s, T); }

//========================================================================================
// evaluation of Recfast-system
//========================================================================================
void evaluate_Recfast_System(double z, double y[], double fvec[], int n)
{
    //====================================================================================
    // standard variables
    //====================================================================================
    double xe, xp, xHep, nH1, nHe1;
    double TR, TM, Comp;
    double Hz, t_cos;        // Hubble and cosmic time
    
    double nHTot, fHe;       // number of H nuclei and NHe/NH
    double alphaH, betaH;    // recombination and photo-ionization rates
    double alphaHe, betaHe;  // recombination and photo-ionization rates
    double BH, BHe, BHe2p2s; // Boltzmann and detailed balance factors
    
    double KH, KHe;          // Sobolev-terms for helium and hydrogen Ki = tauS/(A1g NH1).
                             // This is related to the effective Lyman-alpha transition
                             // rate A1g_eff = PS A1g as Ki ~ (A1g_eff NH1)^-1

    double CH, CHe;          // Peebles inhibition factors for HeI & HI
    
    //====================================================================================
    // get Hubble & NH
    //====================================================================================
    Hz=H_z(z);
    nHTot = NH(z);
    fHe = f_He();
    
    //====================================================================================
    // set variables
    //====================================================================================
    xHep = y[0]; 
    xp   = y[1];
    TM   = y[2];
    t_cos= y[3];
    xe   = xp+xHep;
    
    nH1 = (1.0-xp)*nHTot;
    nHe1 = nHTot*(fHe-xHep);
    
    //====================================================================================
    // radiation temperature at z
    //====================================================================================
    TR = TCMB(z);
    
    //====================================================================================
    // Compton-cooling term
    //====================================================================================
    Comp = 8.0*RF_sigmaT*RF_aRad*pow(TR, 4)/(3.0*Hz*(1.0+z)*RF_mElect*RF_cLight);
    
    //====================================================================================
    // added variables to include variation of fundamental constants (Oct-Dec 2016, LH)
    //====================================================================================
    Variables_Fund_Consts VCD=Variables_Fund_Consts();
    Set_var_const_scale_factors(z, VCD);

    // rescale local variables
    double A2s1sH=RF_Lam2s1sH*VCD.fA2g, A2s1sHe=RF_Lam2s1sHe*VCD.fA2g;
    double lambda21H=RF_LyalphaH/VCD.fl, lambda21He=RF_LyalphaHe/VCD.fl;
    double TMeval=TM/VCD.fT, TReval=TR/VCD.fT;
    Comp*=VCD.fC;

    //====================================================================================
    // H recombination coefficient (including fugde-factor F~1.14)
    //====================================================================================
    alphaH = alphaH_func(TMeval);

    //====================================================================================
    // He recombination coefficient
    //====================================================================================
    alphaHe = alphaHe_func(TMeval);

    //====================================================================================
    // photo-ionization rates and Boltzmann factors
    //====================================================================================
    if(RecVars.eval_ion_Tr) // should be evaluated at T=TR to be physically consistent!
    {
        BH  = Boltzmann(1.0, 1.0, RF_E2s1sH,  TReval);
        betaH  = betaH_func(TReval);

        BHe = Boltzmann(1.0, 1.0, RF_E2s1sHe, TReval);
        // energy levels of HeI 2s and 2p differ -> 2. boltzmann factor
        BHe2p2s = 1.0/Boltzmann(1.0, 1.0, RF_E2p2sHe, TReval);
        betaHe  = betaHe_func (TReval);
    }
    else // this is the usual (but incorrect) evaluation that Recfast would do T==TM
    {
        BH  = Boltzmann(1.0, 1.0, RF_E2s1sH,  TMeval);
        betaH  = betaH_func (TMeval);

        BHe = Boltzmann(1.0, 1.0, RF_E2s1sHe, TMeval);
        // energy levels of HeI 2s and 2p differ -> 2. boltzmann factor
        BHe2p2s = 1.0/Boltzmann(1.0, 1.0, RF_E2p2sHe, TMeval);
        betaHe  = betaHe_func (TMeval);
    }
    
    //====================================================================================
    // Explicit rescalings for photo-ionization and recombination coefficients (added LH)
    //====================================================================================
    alphaH*=VCD.fAp; alphaHe*=VCD.fAp;
    betaH *=VCD.fBp; betaHe *=VCD.fBp;

    //====================================================================================
    // Sobolev-terms for helium and hydrogen Ki = tauS/(A1g NH1) == lam^3/ [8 pi H]
    //====================================================================================
    KHe = pow(lambda21He, 3)/(Hz*8.0*RF_PI);
    KH  = pow(lambda21H , 3)/(Hz*8.0*RF_PI);
    
    //====================================================================================
    // Inhibition factors for HeI & HI
    //====================================================================================
    CHe = (1.0+KHe*A2s1sHe*nHe1*BHe2p2s)/(1.0+KHe*(A2s1sHe+betaHe)*nHe1*BHe2p2s);
    CH  = (1.0+KH *A2s1sH *nH1         )/(1.0+KH *(A2s1sH +betaH )*nH1         );
    
    //====================================================================================
    // derivatives; 1 == He, 2 == H,  3 == TM, 4 == t_cos
    //====================================================================================
    fvec[0] = (alphaHe*xe*xHep*nHTot - betaHe*(fHe-xHep)*BHe)*CHe/((1.0+z)*Hz);
    fvec[1] = (alphaH *xe*xp  *nHTot - betaH *(1.0-xp  )*BH )*CH /((1.0+z)*Hz);
    fvec[2] = Comp*xe/(1.0+xe+fHe)*(TM-TR) + 2.0*TM/(1.0+z);
    fvec[3] = -1.0/(Hz*(1.0+z));

    //====================================================================================
    // add terms due to DM annihilation
    //====================================================================================
    if(RecVars.f_ann>0.0)
        evaluate_DM_annihilation_terms(z, Hz, xp, xHep, CHe, CH, RecVars.f_ann, fvec);
    
    //====================================================================================
    // add terms due to DM decay
    //====================================================================================
    if(RecVars.f_dec>0.0)
        evaluate_DM_decay_terms(z, Hz, t_cos, xp, xHep, CHe, CH,
                                RecVars.f_dec, RecVars.Gamma_dec, fvec);
    
    //====================================================================================
    // add terms due to magnetic field heating
    //====================================================================================
    if(RecVars.B0>0.0)
        evaluate_magnetic_field_heating_RF(z, Hz, fHe, nHTot, xp, xHep,
                                           TM, nHTot/(1.0-Y_p()),
                                           RecVars.B0, RecVars.nB, fvec);

    //====================================================================================
    // add collisional processes for cases with potentially lots of heating
    //====================================================================================
    if(RecVars.B0>0.0 || RecVars.f_dec>0.0)
        evaluate_magnetic_field_heating_RF_colls(z, Hz, fHe, nHTot, xp, xHep, TM, fvec);

    return;
}


//========================================================================================
// wrapper for ODE-solver
//========================================================================================
void fcn(int *neq, double *z, double *y, double *f)
{
    evaluate_Recfast_System(*z, y, f, *neq);
    return;
}

//========================================================================================
// wrapper for ODE-solver with rescaled derivative for Xe
// this is important to stitch the recfast solution to the output of the multi-level code
//========================================================================================
double fcn_rescaled_fac=1.0;
void set_rescaling(double ff){ fcn_rescaled_fac=ff; return; }

void fcn_rescaled(int *neq, double *z, double *y, double *f)
{
    evaluate_Recfast_System(*z, y, f, *neq);
    f[1]*=fcn_rescaled_fac;
    return;
}

//========================================================================================
//========================================================================================
