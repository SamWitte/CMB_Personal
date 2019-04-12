//===========================================================================================
// Author: Jens Chluba & Luke Hart
// First implementation: Dec 2016
// The University of Manchester
// All rights reserved.
//===========================================================================================

//===========================================================================================
// This simple module allows to include the effects of variations of alpha and me into
// Recfast++. Simple time-dependent variations based on phenomenological models
// are also allowed.
//===========================================================================================

#include <cmath>
#include <iostream>
#include <cstdlib>

#include "Recfast++.h"
#include "evalode.Recfast.h"
#include "Variation_constants.Recfast.h"

using namespace std;

//===========================================================================================
// run mode settings
//===========================================================================================
// The following is a description of all the different runmode settings and which
// value corresponds to which mode. The modes are accessed by loading the data using
// Get_evalode_Vars() from 'evalode.Recfast.cpp'
//===========================================================================================
// 0 - no rescaling
// 1 - Rescaling of Boltzmann factor exponentials (i.e., temperatures)
// 2 - Rescaling of Thomson scattering cross section
// 3 - Rescaling of 2s1s 2 photon rate
// 4 - Rescaling of alpha and beta coefficients
// 5 - Rescaling of Lyman alpha rates
// 6 - Rescale everything
//===========================================================================================
void Set_var_const_scale_factors(double aS, double mS, int rmode, Variables_Fund_Consts &D)
{
    // don't rescale if state is the same
    if(D.aS==aS && D.mS==mS) return;
    
    if(rmode == 0){ Reset_var_const_scale_factors(D); return; }
    
    if (rmode > 6) throw_error_RF("Set_var_const_scale_factors", "Invalid mode", 1);
    
    D.aS=aS;
    D.mS=mS;
    D.fT=D.fl=D.fA2g=D.fC=D.fAp=D.fBp=D.fm=1.0;
    
    // Rescale the temperatures here
    if (rmode == 1 || rmode == 6) D.fT=pow(aS, 2)*mS;
    
    // Re-scaled Thomson cross section and cooling
    // extra 1/me from rho_g/me in coefficient is neglected....
    if (rmode == 2 || rmode == 6) D.fC=pow(aS/mS, 2);
    
    // Rescaling of the two photon rates for the Lambda values
    if (rmode == 3 || rmode == 6) D.fA2g=pow(aS, 8)*mS;
    
    // rescalings for photo-ionization, recombination coefficients & Saha
    if (rmode == 4 || rmode == 6) {
        D.fAp=pow(aS/mS, 2);
        D.fBp=pow(aS, 5)*mS;
        D.fm=mS;
    }
    
    // rescaling of wavelength to change Ly-a channel
    if (rmode == 5 || rmode == 6) D.fl=pow(aS, 2)*mS;
    
    return;
}

//===========================================================================================
void Set_var_const_scale_factors(double z, Variables_Fund_Consts &D)
{
    double p  =Get_evalode_Vars().pS;
    double aS =Get_evalode_Vars().aS, mS=Get_evalode_Vars().mS;
    double alp=( aS<=0.0 ? 1.0 : aS*pow((1.0+z)/1100.0, p) );
    double me =( mS<=0.0 ? 1.0 : mS*pow((1.0+z)/1100.0, p) );
    
    Set_var_const_scale_factors(alp, me, Get_evalode_Vars().RM, D);
    return;
}

//===========================================================================================
// restore initial values
//===========================================================================================
void Reset_var_const_scale_factors(Variables_Fund_Consts &D)
{
    D.aS=D.mS=D.fT=D.fl=D.fA2g=D.fC=D.fAp=D.fBp=D.fm=1.0;
    return;
}

//===========================================================================================
// rescale the pivot redshifts in Recfast++ for the different regions
//===========================================================================================
void Rescale_redshift_switch(double &z)
{
    Variables_Fund_Consts D=Variables_Fund_Consts();
    Set_var_const_scale_factors(z,D);
    z*=D.fT;
    return;
}
//========================================================================================
//========================================================================================
