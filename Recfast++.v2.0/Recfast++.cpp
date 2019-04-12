//====================================================================================================================
// Author: Jens Chluba & Luke Hart
// First implementation: Oct 2010
// CITA, University of Toronto
// All rights reserved.
//====================================================================================================================
// 20.05.2018: all recombination settings are now centrally controlled (JC)
// 23.04.2018: added structures to controll all the parameters and data passed on
// 21.02.2018: added cosmic time setup for internal use and decaying particles
// 10.01.2018: compute better high-z Te solution (JC)
// 03.04.2017: added switches using Saha and changed redshift grid (JC)
// 04.01.2017: added new routines for separate H and HeI correction functions (LH)
// 05.12.2016: added dependence of Saha_HeIII on fundamental constants

//====================================================================================================================
//
// Main Recfast++ header with all required function to access recombination problem as standalone code or from
// CosmoRec. the solution to the RECFAST ODE-system is computed. Different driver routines are provided.
//
//--------------------------------------------------------------------------------------------------------------------
// important note: all free particle fractions are relative to the number oh hydrogen
// nuclei. I.e. xe=ne/nH, xp=np/nH etc.
//====================================================================================================================

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

#include "Recfast++.h"
#include "cosmology.Recfast.h"
#include "evalode.Recfast.h"
#include "Rec_corrs_CT.Recfast.h"
#include "Variation_constants.Recfast.h"    // 04.12.16: added JC
#include "ODE_solver.Recfast.h"

using namespace std;
using namespace RECFAST_physical_constants; // defined in constants.Recfast.h
using namespace RECFAST_atomic_constants;   // defined in constants.Recfast.h
using namespace Recfastpp_Cosmology;        // defined in cosmology.Recfast.h

//====================================================================================================================
// simple error message
//====================================================================================================================
void throw_error_RF(string funcname, string message, int k)
{
    cerr << " " << funcname << " : " << message << endl;
    exit(k);
}

//====================================================================================================================
// Saha formula for HeIII
//====================================================================================================================
double SahaBoltz_HeIII(double nH, double fHe, double T)
{
    double c1 = 2.0*RF_PI*RF_mElect*RF_kBoltz*T/(RF_hPlanck*RF_hPlanck);
    double g = pow(c1, 1.5)*exp(-RF_EionHeII/(RF_kBoltz*T))/nH, d=1.0+fHe-g;
    double Xe= 0.5*(d + sqrt(d*d + 4.0*g*(1.0+2.0*fHe)));
    
    return Xe;
}

//====================================================================================================================
// Saha formula for HeII
//====================================================================================================================
double SahaBoltz_HeII(double nH, double fHe, double T)
{
    double c1 = 2.0*RF_PI*RF_mElect*RF_kBoltz*T/(RF_hPlanck*RF_hPlanck);
    double g = 4.0*pow(c1, 1.5)*exp(-RF_EionHeII/(RF_kBoltz*T))/nH, d=1.0-g;
    double Xe= 0.5*(d + sqrt(d*d + 4.0*g*(1.0+fHe)));
    
    return Xe;
}

//====================================================================================================================
// compute quasi-stationary Te temperature (with sigT rescaling...)
//====================================================================================================================
double Te_QS(double z, double Xe, double fC=1.0)
{
    double TR=TCMB(z), Hz=H_z(z);
    double Comp=8.0*RF_sigmaT*RF_aRad*pow(TR, 4)/(3.0*Hz*RF_mElect*RF_cLight);
    double kappa_cool=Comp*fC*Xe/(1.0+Xe+f_He());
    return TR*(1.0-1.0/(1.0+kappa_cool));
}

//====================================================================================================================
// functions to communicate with cosmology object
//====================================================================================================================
int Xe_frac(const CosmoPars &Cosmo, const RecPars &Rec, vector<double> &zarr,
            vector<double> &Xe_H, vector<double> &Xe_He, vector<double> &Xe,
            vector<double> &TM)
{
    vector<double> dXe, dX_H;

    int flg=Xe_frac(Cosmo, Rec, zarr, Xe_H, Xe_He, Xe, dXe, dX_H, TM);
    
    return flg;
}

//====================================================================================================================
// functions to communicate with cosmology object
//====================================================================================================================
int Xe_frac(const CosmoPars &Cosmo, const RecPars &Rec, vector<double> &zarr,
            vector<double> &Xe_H, vector<double> &Xe_He, vector<double> &Xe,
            vector<double> &dXe, vector<double> &TM)
{
  vector<double> dX_H;

  int flg=Xe_frac(Cosmo, Rec, zarr, Xe_H, Xe_He, Xe, dXe, dX_H, TM);

  return flg;
}  

//====================================================================================================================
// compute ionization history like in Recfast-module
//====================================================================================================================
int Xe_frac(const CosmoPars &Cosmo, const RecPars &Rec, vector<double> &zarr,
            vector<double> &Xe_H, vector<double> &Xe_He, vector<double> &Xe,
            vector<double> &dXe, vector<double> &dX_H, vector<double> &TM)
{
    //====================================================================================
    // Set the cosmology object and evalode parts
    //====================================================================================
    Set_Cosmology(Cosmo);
    Set_evalode_Vars(Rec);

    //====================================================================================
    // allocate memory for data (if needed)
    //====================================================================================
    long int j=0, nzpts=Rec.npts;
    
    if(nzpts!=(int)zarr.size()) zarr.resize(nzpts);
    if(nzpts!=(int)Xe_H.size()) Xe_H.resize(nzpts);
    if(nzpts!=(int)Xe_He.size()) Xe_He.resize(nzpts);
    if(nzpts!=(int)Xe.size()) Xe.resize(nzpts);
    if(nzpts!=(int)dXe.size()) dXe.resize(nzpts);
    if(nzpts!=(int)dX_H.size()) dX_H.resize(nzpts);
    if(nzpts!=(int)TM.size()) TM.resize(nzpts);

    //====================================================================================
    double zc_HeIII=9000.0, zc_HeIIIS=4800.0, zc_HeII=3600.0;
    // setting for quasi-constant parts and recombination parts
    int n_low_res=100, n_high_res=(nzpts-2*n_low_res)/3;
    
    //====================================================================================
    // Set in starting values
    //====================================================================================
    double fHe=f_He(), Xe_Saha, Saha_eps_I=5.0e-4, Saha_eps_II=1.0e-5;
    zc_HeIII*=2.726/Cosmo.T0; zc_HeIIIS*=2.726/Cosmo.T0; zc_HeII*=2.726/Cosmo.T0;
    
    //====================================================================================
    // inclusion of fundamental constants [04.12.16: added JC]
    //====================================================================================
    Variables_Fund_Consts VC_Data=Variables_Fund_Consts();

    //====================================================================================
    // Rescale the pivot redshifts for the different regions of recombination
    // as defined in Variations_constants.Recfast.cpp [added LH]
    //====================================================================================
    Rescale_redshift_switch(zc_HeIII);  // HeIII recombined from ionized states
    Rescale_redshift_switch(zc_HeIIIS); // HeIII <--> HeII
    Rescale_redshift_switch(zc_HeII);   // HeII <--> HeI
    
    //====================================================================================
    // above zc_HeIII everything is completely ionized
    //====================================================================================
    double z=Rec.zstart, zend=max(zc_HeIII, Rec.zend), Dz=(z-zend)/n_low_res;
    if (z > zc_HeIII && z > zend)
    {
        for (j=0; z>=zend+Dz; z-=Dz, j++)
        {
            Set_var_const_scale_factors(z, VC_Data);
            zarr[j] = z;
            Xe_H[j] = 1.0;
            Xe_He[j]= fHe;
            Xe[j]   = 1.0+2.0*fHe;
            dXe[j]  = 0.0;
            dX_H[j] = 0.0;
            // 10.01.2018: compute better high-z Te solution [JC]
            TM[j]   = Te_QS(z, Xe[j], VC_Data.fC);

            // check Saha value
            Xe_Saha = SahaBoltz_HeIII(NH(z)*VC_Data.fAp/VC_Data.fBp, fHe, TM[j]/VC_Data.fT);
            if(fabs(Xe[j]/Xe_Saha-1.0)>=Saha_eps_I){ zend = z; break; }
        }
    }
    
    //====================================================================================
    // now HeIII recombination starts and HeII <--> HeIII are in Saha equilibrium
    //====================================================================================
    z=zend; zend=max(zc_HeIIIS, Rec.zend); Dz=(z-zend)/n_high_res;
    if (z > zc_HeIIIS && z > zend)
    {
        for (; z>=zend+Dz; z-=Dz, j++)
        {
            Set_var_const_scale_factors(z, VC_Data);
            zarr[j] = z;
            Xe_H[j] = 1.0;
            Xe_He[j]= fHe;
            // 05.12.2016: T--> T/fT and nH --> nH/(aS mS)^3 [added JC]
            Xe[j]   = SahaBoltz_HeIII(NH(z)*VC_Data.fAp/VC_Data.fBp, fHe, TCMB(z)/VC_Data.fT);
            dXe[j]   = 0.0;
            dX_H[j]   = 0.0;
            // 10.01.2018: compute better high-z Te solution [JC]
            TM[j]   = Te_QS(z, Xe[j], VC_Data.fC);

            // check Saha value
            if(fabs(Xe[j]/(1.0+fHe)-1.0)<=Saha_eps_II){ zend = z; break; }
        }
    }
    
    //====================================================================================
    // Only need to start integrating ODE before HeI recombination. Anything before that 
    // time is just a constant ionization fraction. xe = 1.0 + fHe 
    //====================================================================================  
    z=zend; zend=max(zc_HeII, Rec.zend); Dz=(z-zend)/n_low_res;
    if (z > zc_HeII && z > zend)
    {
        for (; z>=zend+Dz; z-=Dz, j++)
        {
            Set_var_const_scale_factors(z, VC_Data);
            zarr[j] = z;
            Xe_H[j] = 1.0;
            Xe_He[j]= fHe;
            Xe[j]   = 1.0+fHe;
            dXe[j]   = 0.0;
            dX_H[j]   = 0.0;
            // 10.01.2018: compute better high-z Te solution [JC]
            TM[j]   = Te_QS(z, Xe[j], VC_Data.fC);

            // check Saha value
            Xe_Saha = SahaBoltz_HeII(NH(z)*VC_Data.fAp/VC_Data.fBp, fHe, TM[j]/VC_Data.fT);
            if(fabs(Xe[j]/Xe_Saha-1.0)>=Saha_eps_I){ zend = z; break; }
        }
    }

    z=zend;
    Set_var_const_scale_factors(z, VC_Data);
    zarr[j] = z;
    Xe_H[j] = 1.0;
    Xe_He[j]= fHe;
    Xe[j]   = 1.0+fHe;
    dXe[j]   = 0.0;
    dX_H[j]   = 0.0;
    // 10.01.2018: compute better high-z Te solution [JC]
    TM[j]   = Te_QS(z, Xe[j], VC_Data.fC);

    //====================================================================================
    // reading recombination correction as given by Chluba & Thomas 2010
    //====================================================================================
    if(Rec.include_CF==1) read_recombination_correction(Rec.verbosity);
    
    //====================================================================================
    // Now He I recombination starts
    //====================================================================================
    if (z > Rec.zend)
    {
        double zs, ze;
        
        //================================================================================
        // initialise zarr (linear in z)
        //================================================================================
        double Dz=(z-Rec.zend)/(double)(nzpts-j-1);
        long int i;
        for(zarr[j]=z, i=j+1; i<nzpts; i++) zarr[i]=zarr[i-1]-Dz;
        zarr[nzpts-1]=0.0;
        
        //================================================================================
        // Set starting solution. Everything is ionized, 
        // and the matter temperature = the radiation temperature           
        // 0 == xHe = nHe/nHTot
        // 1 == xp = np/nHTot
        // 2 == TM (matter temperature)
        // 3 == t(z) total cosmic time
        //================================================================================
        
        //================================================================================
        // set up Solution memory for ODE-solver
        //================================================================================
        ODE_solver_Solution Sz;
        zs=zarr[j];
        //
        Sz.z=zs;
        Sz.y.resize(4);
        Sz.y[0]=Xe_He[j];
        Sz.y[1]=Xe_H[j];
        Sz.y[2]=TM[j];
        Sz.y[3]=t_cos_rad(zs);  // just used internally
        //
        Sz.dy.resize(4);
        ODE_Solver_set_up_solution_and_memory(zs, Sz, fcn);     
        
        //================================================================================
        // solve for Xe(z)
        //================================================================================
        for(i=j+1; i<nzpts; i++)
        {
            ze=zarr[i];

            ODE_Solver_Solve_history(zs, ze, Sz);
            
            //============================================================================
            // save step
            //============================================================================
            zs=ze;
            Xe_He[i]= Sz.y[0];
            Xe_H[i] = Sz.y[1];
            TM[i]   = Sz.y[2];
            Xe[i]   = Xe_H[i]+Xe_He[i];
            if(Rec.include_CF==1)
            {
                Set_var_const_scale_factors(ze, VC_Data); // 04.12.16: added JC
                Recombination_correction_factor(ze/VC_Data.fT, Xe[i], Xe_H[i], Xe_He[i]);
            }
            //
            dXe[i]  = Sz.dy[0]+Sz.dy[1];
            dX_H[i] = Sz.dy[1];
        }
    }
    
    return 0;
}

//====================================================================================================================
// to start computation with a given initial solution at low z
// zarr contains the points at which the solution should be obtained
// rescaling of f_Xe from initial condition is used; For this dXei is needed
//====================================================================================================================
int Xe_frac_rescaled(const CosmoPars &Cosmo, const RecPars &Rec, const vector<double> &zarr,
                     vector<double> &Xe_H, vector<double> &Xe_He, vector<double> &Xe,
                     vector<double> &TM,
                     double Xe_Hi, double Xe_Hei, double Xei, double TMi, double dXei)
{
    //====================================================================================
    // Set the cosmology object and evalode parts
    //====================================================================================
    Set_Cosmology(Cosmo);
    Set_evalode_Vars(Rec);

    //====================================================================================
    // allocate memory for data (if needed)
    //====================================================================================
    long int i, j=0, nzpts=Rec.npts;
    
    if(nzpts!=(int)zarr.size())
        throw_error_RF("Xe_frac_rescaled", "redshift settings inconsistent", 1);
    
    if(nzpts!=(int)Xe_H.size()) Xe_H.resize(nzpts);
    if(nzpts!=(int)Xe_He.size()) Xe_He.resize(nzpts);
    if(nzpts!=(int)Xe.size()) Xe.resize(nzpts);
    if(nzpts!=(int)TM.size()) TM.resize(nzpts);
    
    //====================================================================================
    // Set in starting values
    //====================================================================================
    Xe_H[j] = Xe_Hi;
    Xe_He[j]= Xe_Hei;
    TM[j]   = TMi;
    Xe[j]   = Xei;
    
    //====================================================================================
    // inclusion of fundamental constants [04.12.16: added JC]
    //====================================================================================
    Variables_Fund_Consts VC_Data=Variables_Fund_Consts();

    //====================================================================================
    // reading recombination correction as given by Chluba & Thomas 2010
    //====================================================================================
    if(Rec.include_CF==1) read_recombination_correction(Rec.verbosity);
        
    //====================================================================================
    // everything starts with HeI recombination
    //====================================================================================
    if (Rec.zstart > Rec.zend)
    {
        double zs, ze;
        
        //================================================================================
        // set up Solution memory for ODE-solver
        //================================================================================
        ODE_solver_Solution Sz;
        zs=zarr[j];
        //
        Sz.z=zs;
        Sz.y.resize(4);
        Sz.y[0]=Xe_He[j];
        Sz.y[1]=Xe_H[j];
        Sz.y[2]=TM[j];
        Sz.y[3]=t_cos_rad(zs);  // just used internally
        //
        Sz.dy.resize(4);
        ODE_Solver_set_up_solution_and_memory(zs, Sz, fcn_rescaled);        
        
        //================================================================================
        // determine rescaling factor
        //================================================================================
        double fcn_Xe[4];
        int neq=4;
        fcn(&neq, &zs, &Sz.y[0], fcn_Xe);
        double ff=dXei/fcn_Xe[1];
        set_rescaling(ff);
        
        if(Rec.verbosity>=1) cout << " Xe_frac:: rescale factor= " << ff << endl;
        
        //================================================================================
        // solve for Xe(z)
        //================================================================================
        for(i=j+1; i<nzpts; i++)
        {
            ze=zarr[i];

            ODE_Solver_Solve_history(zs, ze, Sz);
            
            //============================================================================
            // save step
            //============================================================================
            zs=ze;
            Xe_He[i]= Sz.y[0];
            Xe_H[i] = Sz.y[1];
            TM[i]   = Sz.y[2];
            Xe[i]   = Xe_H[i]+Xe_He[i];
            Xe[i]   = Xe_H[i]+Xe_He[i];
            if(Rec.include_CF==1)
            {
                Set_var_const_scale_factors(ze, VC_Data); // 04.12.16: added JC
                Recombination_correction_factor(ze/VC_Data.fT, Xe[i], Xe_H[i], Xe_He[i]);
            }
        }
    }
    
    return 0;
}
//====================================================================================================================
//====================================================================================================================
