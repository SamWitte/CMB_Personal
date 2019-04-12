//========================================================================================
// Author: Jens Chluba (Oct. 2014)
// Johns Hopkins University
//========================================================================================
// 21.02.2018: added separate function to evaluate collisions
// 30.03.2017: added set functions for parameters related to PMF [JC]
// 01.12.2016: added settings for transition from no heating before z>1088

//========================================================================================
// This simple module allows to add the effect of heating by magnetic fields.
// Expressions from Sethi & Subramanian, 2005 and Kunze & Komatsu, 2013 are used.
//========================================================================================

#include <cmath>
#include <iostream>
#include <cstdlib>

#include "Recfast++.h"
#include "evalode.Recfast.h"
#include "cosmology.Recfast.h"
#include "Magnetic_fields.Recfast.h"

using namespace std;
using namespace RECFAST_physical_constants; // defined in constants.Recfast.h
using namespace RECFAST_atomic_constants;

//========================================================================================
// define kD mapping for given B0 and nB
//========================================================================================
double kD_func(double B0, double nB)
{
    //=======================================================================
    // KK parametrization and interpretation; B0 and nB unchanged; (default)
    //=======================================================================
    double kD=286.91/B0; // in Mpc^-1; cos(theta)=1;
    
    //======================================================================
    // set dissipation scale and convert B_lam --> B0 when using PF vars
    //======================================================================
    if(Get_evalode_Vars().KK_or_PF_Lorentz)
    {
        double lambda_Mpc=1.0, k_lambda=2.0*RF_PI/lambda_Mpc;
        double B_lambda=B0, nBp3=nB+3.0, nBp5=nB+5.0;
        double h0=Recfastpp_Cosmology::Get_Cosmology().h100;
        
        kD=pow(2.9e+4*h0, 1.0/nBp5)/pow(B_lambda, 2.0/nBp5)*pow(k_lambda, nBp3/nBp5);
        //kD=pow(5.5e+4*h0, 1.0/nBp5)/pow(B_lambda, 2.0/nBp5)*pow(k_lambda, nBp3/nBp5);
    }

    return kD;
}

//========================================================================================
// magnetic field decay rate following Kunze & Komatsu, 2013.
//========================================================================================
double Gamma_decay_RF(double z, double Hz, double B0, double nB)
{
    double zi=1088.0;
    
    // B0 is in nG
    double kd=kD_func(B0, nB);
    double m=2.0*(nB+3.0)/(nB+5.0);
    double td_ti=14.8/B0/kd;
    double rhoB=0.1*B0*B0*1.0e-18/(8.0*RF_PI)*pow(1.0+z, 4);  // ergs cm^-3 == 0.1 J m^-3

    double Gamma=0.0;
    double a=log(1.0+td_ti);
    double sig=5.0e+3;        // width if the Gaussian transition
    int trans=1;              // choose type of transition.
    
    if(z<zi) Gamma=rhoB*Hz*1.5*m*pow(a, m)/pow(a+1.5*log((1.0+zi)/(1.0+z)), m+1.0);
    // make smooth transition from no heating to heating case
    else if(z>=zi && trans==0)
    {
        double G_zi=rhoB*Hz*1.5*m/a;
        //Gamma=0.0;
        //Gamma=G_zi*pow((1.0+zi)/(1.0+z), 4);
        //Gamma=G_zi*exp(-pow(z-zi, 2)/sig);
        Gamma=G_zi*exp(-pow(z-zi, 2)/sig)*pow((1.0+zi)/(1.0+z), 4);
    }
    else if(z>=zi && trans==1)
    {
        // smooth 1. derivative transition. Gaussian at z>=z0, polynomial at zi<=z<=z0
        double G_zi=rhoB*Hz*1.5*m/a*pow((1.0+zi)/(1.0+z), 4);
        double dG_dzi=G_zi*(4.0+1.5*(1.0+m)/a)/(1.0+zi);
        double k=7.0;
        double z0=zi*1.001;
        double g=dG_dzi/( k*pow(zi-z0, k-1) );
        double g0=G_zi-g*pow(zi-z0, k);
        
        if(z<=z0) Gamma=g0+g*pow(z-z0, k);
        else Gamma=g0*exp(-pow(z-z0, 2)/sig);
    }
    
    return Gamma;
}

//========================================================================================
// magnetic field heating rate for ambipolar diffusion Kunze & Komatsu, 2013.
//========================================================================================
double Gamma_ambi_RF(double z, double Tm, double nbar, double Xp, double B0, double nB)
{
    // fit to the integral KK
    double L2=0.8313*pow(nB+3.0, 1.105)*(1.0-1.020e-2*(nB+3.0));
    
    // fit to the integral of Finelli
    if(Get_evalode_Vars().KK_or_PF_Lorentz)
        L2=0.6615*pow(nB+3.0, 0.8874)*(1.0-0.1367*(nB+3.0)+7.574e-3*pow(nB+3.0, 2));
    
    // similar to Sethi & Subramanian, 2005
    //double L2=1.0;
    
    // B0 is in nG
    double rhoB=0.1*B0*B0*1.0e-18/(8.0*RF_PI)*pow(1.0+z, 4);  // ergs cm^-3 == 0.1 J m^-3
    double sig=0.649*pow(Tm, 0.375)*1.0e-9*1.0e-6;            // m^3 / sec
    double g=sig/(2.0*RF_mHatom);
    double rhobar=RF_mHatom*nbar;
    double kd=kD_func(B0, nB);
    double Gamma=(1.0-Xp)/Xp/g*pow((1.0+z)*kd/(RF_Mpc*1.0e-2)*rhoB/rhobar, 2)*L2;
    
    // fit that reproduces KK. It seems this is ~1/(8pi)^2 times smaller than the correct rate
    //double Gamma=(1.0-Xp)/Xp*pow(1.0+z, 4)/pow(Tm, 0.375)*5.7e-40*L2*pow(B0/3.0, 2);
    
    //cout << pow(1.0+z, 4)/pow(Tm, 0.375)*5.7e-40*pow(B0/3.0, 2)*pow(8.0*RF_PI, 2) << " "
    //     << Gamma/L2/((1.0-Xp)/Xp) << endl;

    return Gamma;
    //return Gamma/pow(8.0*RF_PI, 2);
}

//========================================================================================
// evaluation of ODE terms for magnetic heating
//========================================================================================
void evaluate_magnetic_field_heating_RF(double z, double Hz, double fHe, double NH,
                                        double xp, double xHep, double Tm, double nbar,
                                        double B0, double nB, double fvec[])
{
    double Xe=xp+xHep;
//    double fheat=2.0/3.0/RF_kBoltz/NH;    // old expression for heat capacity
    double fheat=2.0/3.0/RF_kBoltz/(1.0+fHe+Xe)/NH;
    
    double dE_dz=0.0;
    if(Get_evalode_Vars().turb) dE_dz+=Gamma_decay_RF(z, Hz, B0, nB)/Hz/(1.0+z);
    if(Get_evalode_Vars().ambi) dE_dz+=Gamma_ambi_RF(z, Tm, nbar, xp, B0, nB)/Hz/(1.0+z);
    
    fvec[2]-=fheat*dE_dz;
   
    return;
}

//========================================================================================
// evaluation of ODE terms for collisions
//========================================================================================
void evaluate_magnetic_field_heating_RF_colls(double z, double Hz, double fHe, double NH,
                                              double xp, double xHep, double Tm, double fvec[])
{
    double Xe=xp+xHep;
    double dE_dz=0.0, fheat=2.0/3.0/RF_kBoltz/(1.0+fHe+Xe)/NH;
    
    //====================================================================================
    // add collisional ionization for H & He (kci in m^3 / sec from Bell et al. 1983)
    // Usually this does not make a difference due to the exponential suppression.
    //====================================================================================
    double kciH =5.466e-9*1.0e-6*1.07*sqrt(Tm/1.0e+4)*exp(-157821.462/Tm);
    double kciHe=5.466e-9*1.0e-6*0.37*sqrt(Tm/1.0e+4)*exp(-285471.174/Tm);
    
    fvec[0]-=kciHe*Xe*(fHe-xHep)*NH/Hz/(1.0+z);
    fvec[1]-=kciH *Xe*(1.0 - xp)*NH/Hz/(1.0+z);
    
    // cooling terms
    dE_dz-=13.6*1.602176487e-19*kciH *Xe*(1.0 - xp)*NH*NH/Hz/(1.0+z);
    dE_dz-=24.6*1.602176487e-19*kciHe*Xe*(fHe-xHep)*NH*NH/Hz/(1.0+z);
    
    fvec[2]-=fheat*dE_dz;
    
    return;
}

//========================================================================================
//========================================================================================
