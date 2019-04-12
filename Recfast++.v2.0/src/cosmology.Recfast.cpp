//====================================================================================================================
// Author: Jens Chluba
// first implementation: June 2010
// Last modification: April 2018
// CITA, University of Toronto
// All rights reserved.
//====================================================================================================================
// 23.04.2018: added new structure for cosmology parameters
// 22.02.2018: added cosmic time for radiation-dominated era
// 08.06.2012: added option to use external hubble factor

#include <cmath>
#include <iostream>

#include "Recfast++.h"
#include "cosmology.Recfast.h"

using namespace std;
using namespace RECFAST_physical_constants; // defined in constants.Recfast.h

namespace Recfastpp_Cosmology
{
//====================================================================================================================
// Global Variables; defined in cosmology.Recfast.h
//====================================================================================================================
struct Cosmological_variables
{
    double fHe, H0;
    
    CosmoPars Cosmo;
    
    //========================================================================================
    // set default parameters
    //========================================================================================
    Cosmological_variables()
    {
        CosmoPars();
        H0 = Cosmo.h100*100.0*1.0e+5/RF_Mpc;
        fHe= Cosmo.Yp/(4.0*RF_fac_mHemH*(1.0-Cosmo.Yp));
    }
    
    //========================================================================================
    // show variables
    //========================================================================================
    void show() const
    {
        cout << " H0= " << H0 << " km s^-1 Mpc^-1 and fHe= " << fHe << endl;
        Cosmo.show();
    }
};

Cosmological_variables CosmoVars;

void Set_Cosmology(const CosmoPars &CosmoInputs)
{
    CosmoVars.Cosmo=CosmoInputs;
    CosmoVars.H0 = CosmoVars.Cosmo.h100*100.0*1.0e+5/RF_Mpc;
    CosmoVars.fHe= CosmoVars.Cosmo.Yp/(4.0*RF_fac_mHemH*(1.0-CosmoVars.Cosmo.Yp));
    
    return;
}

const CosmoPars& Get_Cosmology(){ return CosmoVars.Cosmo; }

//====================================================================================================================
// Hubble-function in 1/sec
//====================================================================================================================
double H_z_loc(double z)
{
  double Fnu, Zeq, z1=1.0+z;
  
  Fnu = CosmoVars.Cosmo.Neff*(7.0/8.0)*pow(4.0/11.0, 4.0/3.0);
  Zeq = 3.0*pow(CosmoVars.H0*RF_cLight, 2)
          /(8.0*RF_PI*RF_G*RF_aRad*(1.0+Fnu))
          /(pow(CosmoVars.Cosmo.T0, 4))*CosmoVars.Cosmo.Omega_m-1.0;

  return CosmoVars.H0*sqrt(CosmoVars.Cosmo.Omega_L+pow(z1, 2)
                 *(CosmoVars.Cosmo.Omega_k+z1*CosmoVars.Cosmo.Omega_m*(1.0+z1/(1.0+Zeq)) ) );
}

//====================================================================================================================
// allow setting Hubble function from outside of Recfast++ (added 08.06.2012)
//====================================================================================================================
double (*H_z_ptr)(double)=H_z_loc;
void set_H_pointer(double (*Hz_p)(double)){ H_z_ptr=Hz_p; return; }
void reset_H_pointer(){ H_z_ptr=H_z_loc; return; }

//====================================================================================================================
double H_z(double z)
{ return H_z_ptr(z); }

double t_cos_rad(double z)
{ return 1.0/2.0/H_z(z); }

//====================================================================================================================
// simple density parameters
//====================================================================================================================
double Omega_cdm(){ return CosmoVars.Cosmo.Omega_m-CosmoVars.Cosmo.Omega_b; }
double Omega_b(){ return CosmoVars.Cosmo.Omega_b; }
double Omega_H(){ return CosmoVars.Cosmo.Omega_b*(1.0-CosmoVars.Cosmo.Yp); }
double Y_p(){ return CosmoVars.Cosmo.Yp; }
double f_He(){ return CosmoVars.fHe; }

//====================================================================================================================
// hydrogen number density in m^-3
//====================================================================================================================
double NH(double z)
{
  double mu_H=1.0/(1.0-CosmoVars.Cosmo.Yp);
  return 3.0*pow(CosmoVars.H0, 2)*CosmoVars.Cosmo.Omega_b/(8.0*RF_PI*RF_G*RF_mHatom*mu_H)*pow(1.0+z, 3);
}

//====================================================================================================================
// CMB temperature at z
//====================================================================================================================
double TCMB(double z){ return CosmoVars.Cosmo.T0*(1.0+z); }

//====================================================================================================================
// compute total contribution from relativistic particles (photon & neutrinos)
//====================================================================================================================
double calc_Orel(double TCMB0, double Nnu, double h100)
{ 
    double H0=h100*100.0*1.0e+5/RF_Mpc;
    double a=RF_aRad*pow(TCMB0, 4)/RF_cLight/RF_cLight; 
    double b=3.0*pow(H0, 2)/(8.0*RF_PI*RF_G);  
    return a/b*(1.0+Nnu*(7.0/8.0)*pow(4.0/11.0, 4.0/3.0));
}
}
//====================================================================================================================
//====================================================================================================================
