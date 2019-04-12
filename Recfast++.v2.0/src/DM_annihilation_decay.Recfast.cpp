//===========================================================================================
// Author: Jens Chluba
// Last modification: Feb 2018
// CITA, University of Toronto
// All rights reserved.
//===========================================================================================

//===========================================================================================
// This simple module allows to add the ionizing effect of DM-annihilation and decays.
// The important equations as implemented here were given in Chluba 2010 and then augmented
// to allow for decaying particles. Original works are Chen & Kamionkowski 2004 and
// Padmanabhan & Finkbeiner 2005.
//===========================================================================================

#include <cmath>
#include <iostream>
#include <cstdlib>

#include "Recfast++.h"
#include "cosmology.Recfast.h"
#include "DM_annihilation_decay.Recfast.h"

using namespace std;
using namespace RECFAST_physical_constants; // defined in constants.Recfast.h
using namespace Recfastpp_Cosmology;        // defined in cosmology.Recfast.h

bool RECFAST_add_excitations=1;             // switch the effect of extra excitations on/off

//===========================================================================================
// branching into ionizations according to Chen & Kamionkowski 2004
// (see also Chluba 2010)
//===========================================================================================
double branching_ions_Chen(double x){ return (1.0-x)/3.0; }
double branching_ex_Chen  (double x){ return (1.0-x)/3.0; }
double branching_heat_Chen(double xp, double ZHeII, double fHe)
{ return (1.0 + 2.0*xp + fHe*(1.0 + 2.0*ZHeII))/3.0/(1.0+fHe); }

//===========================================================================================
// branching into ionizations according to Shull & van Steenberg 1985
// (see also Chluba 2010)
//===========================================================================================
double branching_ions_SvSt(double x){ return 0.4*pow(1.0-pow(x, 0.4), 7.0/4.0); }


//===========================================================================================
// evaluation of ODE terms given heating rate
//===========================================================================================
void evaluate_ionization_heating_excitation_terms(double z, double Hz,
                                                  double xp, double xHep,
                                                  double CHe, double CH,
                                                  double dE_dz, double fvec[])
{
    //---------------------------------------------------------------------------------------
    // different branching ratios. Simple approximations here but this can be made more
    // comprehensive using Slatyer et al., if needed
    //---------------------------------------------------------------------------------------
    double fHe=f_He();
    double bHe_ion=branching_ions_Chen(xHep/fHe);
    double bHI_ion=branching_ions_Chen(xp);
    double b_heat=branching_heat_Chen(xp, xHep/fHe, fHe);
    // for temperature equation
    double fheat=2.0/3.0*1.602176487e-19/RF_kBoltz/(1.0+fHe+xp+xHep);
    
    //=======================================================================================
    // return different terms (HeI, HI, Te)
    //=======================================================================================
    fvec[0]+=bHe_ion/24.6*fHe/(1.0+fHe)*dE_dz;
    fvec[1]+=bHI_ion/13.6    /(1.0+fHe)*dE_dz;
    fvec[2]+=fheat*b_heat*dE_dz;
    
    //=======================================================================================
    // additional excitations
    //=======================================================================================
    if(RECFAST_add_excitations)
    {
        double bHe_ex=branching_ex_Chen(xHep/fHe);
        double bHI_ex=branching_ex_Chen(xp);
        
        fvec[0]+=(1.0-CHe)*bHe_ex/21.0*fHe/(1.0+fHe)*dE_dz;
        fvec[1]+=(1.0-CH )*bHI_ex/10.2    /(1.0+fHe)*dE_dz;
    }

    return;
}


//===========================================================================================
// evaluation of ODE terms for annihilation
//===========================================================================================
void evaluate_DM_annihilation_terms(double z, double Hz,
                                    double xp, double xHep,
                                    double CHe, double CH,
                                    double fDM, 
                                    double fvec[])
{
    //=======================================================================================
    // fDM [eV/s] gives annihilation efficiency; typical value fDM=2.0e-24 eV/s
    // (see Chluba 2010 for definitions)
    //=======================================================================================
    double Fcal=-(1.0+z)*(1.0+z)/Hz; // '-' from transformation dt --> dz
    double dE_dz=fDM * Fcal;  // here the factor 1/NH was canceled with definition of dE_dt

    //=======================================================================================
    // add corresponding terms
    //=======================================================================================
    evaluate_ionization_heating_excitation_terms(z, Hz, xp, xHep, CHe, CH, dE_dz, fvec);

    return;
}

//===========================================================================================
// evaluation of ODE terms for decay
//-------------------------------------------------------------------------------------------
// fdec      == fraction of dark matter that is decaying. 0<= fdec <= 1
// Gamma_dec == decay rate in sec. This is >~ 10^-17 sec for long-lived particles
//===========================================================================================
void evaluate_DM_decay_terms(double z, double Hz, double t_cos,
                             double xp, double xHep,
                             double CHe, double CH,
                             double fdec, double Gamma_dec,
                             double fvec[])
{
    //=======================================================================================
    // total energy deposition in eV per hydrogen nucleus and per dz
    //=======================================================================================
    double rho_cdm_NH=Omega_cdm()/Omega_H() * 938.28e+6; // eV
    double f_eff=0.1; // energy transfer efficiency factor
    double dE_dz=-rho_cdm_NH * f_eff * fdec*Gamma_dec*exp(-Gamma_dec*t_cos)/(Hz*(1.0+z));
    
    //=======================================================================================
    // add corresponding terms
    //=======================================================================================
    evaluate_ionization_heating_excitation_terms(z, Hz, xp, xHep, CHe, CH, dE_dz, fvec);
    
    return;
}

//===========================================================================================
//===========================================================================================

