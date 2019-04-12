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

#ifndef VAR_CONSTS_H
#define VAR_CONSTS_H

//===========================================================================================
// structure to communicate current state of factors
//===========================================================================================
struct Variables_Fund_Consts
{
    // default constructor
    Variables_Fund_Consts() : aS(1.0), mS(1.0),
                              fT(1.0), fl(1.0), fA2g(1.0),
                              fC(1.0), fAp(1.0), fBp(1.0), fm(1.0) {}

    double aS, mS; // aS=alpha/alpha_r & mS=me/me_r
    double fT;     // temperature scale factor
    double fl;     // wavelength scale factor
    double fA2g;   // two-photon scale factor
    double fC;     // Compton cooling scale factor
    double fAp;    // normalization phot-coefficient scale factor
    double fBp;    // normalization rec-coefficient scale factor
    double fm;     // mass scale factor for Saha
};

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
void Set_var_const_scale_factors(double z, Variables_Fund_Consts &D);

//===========================================================================================
// restore initial values
//===========================================================================================
void Reset_var_const_scale_factors(Variables_Fund_Consts &D);

//===========================================================================================
// rescale pivot redshifts in recombination history
//===========================================================================================
void Rescale_redshift_switch(double &z);

#endif

//===========================================================================================
//===========================================================================================
