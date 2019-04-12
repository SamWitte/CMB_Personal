//====================================================================================================================
// Authors: Jens Chluba & Luke Hart
// First implementation: Oct 2010
// CITA, University of Toronto
// All rights reserved.
//====================================================================================================================
// 23.04.2018: added simple parser setup and rearranged structures (significantly)
// 21.02.2018: added decaying particle case
// 05.12.2016: added simple redshift scaling of constants ~(1+z)^p
// 01.12.2016: changed setup of Tr flag and added setup for variation of fundamental
//             constants (developed by Luke Hart)
// 14.11.2014: added setup for magnetic field heating modes

#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

#include "Recfast++.h"
#include "cosmology.Recfast.h"
#include "evalode.Recfast.h"
#include "Magnetic_fields.Recfast.h"
#include "Variation_constants.Recfast.h"
#include "parser.Recfast.h"

using namespace std;

//====================================================================================================================
// relevant data structure for runs
//====================================================================================================================
struct main_Settings
{
    string path, addname;
    int verbosity;
    
    CosmoPars Cosmo;  // structure defined in cosmology.Recfast.h
    RecPars Rec;      // structure defined in evalode.Recfast.h
    
    //========================================================================================
    // set default parameters
    //========================================================================================
    main_Settings()
    {
        CosmoPars();
        RecPars();
        path="./outputs/";
        addname=".dat";
        verbosity=1;
    }
    
    //========================================================================================
    // show variables
    //========================================================================================
    void show() const
    {
        cout << " verbosity level = " << verbosity << endl;
        cout << " output path= " << path << " addname= " << addname << endl;
        Cosmo.show();
        Rec.show();
    }
};

//====================================================================================================================
// show parameters as they have been loaded (add parameters to the list here to show them; useful for debugging too)
//====================================================================================================================
void show_recombination_settings(const CosmoPars &Cosmo, const RecPars &Rec)
{
    if(Rec.verbosity<=0) return;
    
    cout << "\n Recfast++ :: show_recombination_settings: " << endl;
    cout << " running Recfast++ history for the following parameters: " << endl;
    
    cout << "\n zs: " << Rec.zstart << "\t ze: " << Rec.zend << "\t nzpts: " << Rec.npts << endl;
    cout << " Y_p: " << Cosmo.Yp << "\t TCMB: " << Cosmo.T0 << endl;
    cout << " OmegaM: " << Cosmo.Omega_m << "\t OmegaB: " << Cosmo.Omega_b
         << " OmegaL: " << Cosmo.Omega_L << "\t OmegaK: " << Cosmo.Omega_k << endl;
    cout << " Omega_rel (photons & neutrinos): "
         << Recfastpp_Cosmology::calc_Orel(Cosmo.T0, Cosmo.Neff, Cosmo.h100)
         << " Neff= " << Cosmo.Neff << endl;
    
    cout << " Hubble constant in units of 100 km s-1 Mpc-1: " << Cosmo.h100 << endl;
    cout << " Fudge factor for H recombination: " << Rec.F << endl;
    
    cout << " Using HI A2s1s = " << Rec.A2s1s << " s^-1";
    if(Rec.A2s1s==8.22458) cout << " (default value)";
    cout << endl;
    
    if(Rec.f_ann!=0.0)
        cout << " DM-annihilations with efficiency " << Rec.f_ann << " will be included " << endl;
    
    if(Rec.f_dec!=0.0)
        cout << " DM-decay parameters f_dec= " << Rec.f_dec << " and Gamma_dec= " << Rec.Gamma_dec << endl;
    
    if(Rec.include_CF)
        cout << " Including recombination corrections following Chluba & Thomas 2010 / Hart & Chluba 2017" << endl;
    
    if(Rec.eval_ion_Tr) cout << " Using T=Tr for photo-ionization rate" << endl;
    else cout << " Using T=Te for photo-ionization rate (Recfast default)" << endl;
    
    if(Rec.B0>0.0) cout << " Will include heating by magnetic fields "
                          << "(decay==" << Rec.turb
                          << ", ambi==" << Rec.ambi
                          << ", Lorentz==" << Rec.KK_or_PF_Lorentz << ")" << endl;

    if(Rec.aS!=1.0 || Rec.mS!=1.0 || Rec.pS!=0.0)
        cout << " Variation of fundamental constants: alp/alp_r== " << Rec.aS
             << " me/me_r== " << Rec.mS
             << " pow== " << Rec.pS
             << " mode== " << Rec.RM << endl;
    
    cout << endl;

    return;
}

//====================================================================================================================
// reading parameters from file. To add parameters make changes to CosmoPars and RecPars structures
//====================================================================================================================
void read_parameter_file(string fname, main_Settings &inp)
{
    //==============================================================================================
    // read values (if given) from file
    //==============================================================================================
    bool found; int ival; double dval; string sval;
    file_content pfc;
    parser_read_file(fname, pfc, 0);
    
    //==============================================================================================
    // reset verbosity level
    //==============================================================================================
    parser_read(pfc, "verbosity level Recfast++", ival, found, 0);
    if(found) inp.verbosity=inp.Rec.verbosity=ival;

    if(inp.verbosity>-1)
        cout << " Recfast++ :: read_parameter_file : reading parameters from: " << fname << endl;

    bool show_params=(inp.verbosity > 3 ? 1 : 0);
    // read again to show the parameter...
    if(show_params) parser_read(pfc, "verbosity level Recfast++", ival, found, show_params);

    //==============================================================================================
    // path etc
    //==============================================================================================
    parser_read(pfc, "path for output", sval, found, show_params);
    if(found) inp.path=sval;

    parser_read(pfc, "addition to name for output", sval, found, show_params);
    if(found) inp.addname=sval;
    
    //==============================================================================================
    // cosmological parameters
    //==============================================================================================
    parser_read(pfc, "Yp", dval, found, show_params);
    if(found) inp.Cosmo.Yp=dval;
    
    parser_read(pfc, "T0", dval, found, show_params);
    if(found) inp.Cosmo.T0=dval;
    
    parser_read(pfc, "Omega_m", dval, found, show_params);
    if(found) inp.Cosmo.Omega_m=dval;
    
    parser_read(pfc, "Omega_b", dval, found, show_params);
    if(found) inp.Cosmo.Omega_b=dval;

    parser_read(pfc, "Omega_k", dval, found, show_params);
    if(found) inp.Cosmo.Omega_k=dval;
    
    parser_read(pfc, "Omega_L", dval, found, show_params);
    if(found) inp.Cosmo.Omega_L=dval;

    parser_read(pfc, "h100", dval, found, show_params);
    if(found) inp.Cosmo.h100=dval;
    
    parser_read(pfc, "N_eff", dval, found, show_params);
    if(found) inp.Cosmo.Neff=dval;

    //----------------------------------------------------------------------------------------------
    // set Omega_L according to Omega_K
    //----------------------------------------------------------------------------------------------
    if(inp.Cosmo.Omega_L<=0.0)
    {
        double Omega_rel=Recfastpp_Cosmology::calc_Orel(inp.Cosmo.T0, inp.Cosmo.Neff, inp.Cosmo.h100);
        inp.Cosmo.Omega_L=1.0-inp.Cosmo.Omega_m-inp.Cosmo.Omega_k-Omega_rel;
    }

    //==============================================================================================
    // recombination parameters
    //----------------------------------------------------------------------------------------------
    // info for redshift output
    //----------------------------------------------------------------------------------------------
    parser_read(pfc, "npts", ival, found, show_params);
    if(found) inp.Rec.npts=ival;
    if(inp.Rec.npts<1000) throw_error_RF("read_parameter_file", "Choose npts > 1000", 1);
    
    parser_read(pfc, "zstart", dval, found, show_params);
    if(found) inp.Rec.zstart=dval;
    if(inp.Rec.zstart<3500) throw_error_RF("read_parameter_file", "Choose zstart > 3500", 1);
    
    parser_read(pfc, "zend", dval, found, show_params);
    if(found) inp.Rec.zend=dval;
    if(inp.Rec.zend<0) throw_error_RF("read_parameter_file", "Choose zstart >=0", 1);
    
    //----------------------------------------------------------------------------------------------
    // standard physics settings
    //----------------------------------------------------------------------------------------------
    parser_read(pfc, "Recfast fudge factor", dval, found, show_params);
    if(found) inp.Rec.F=dval;
    if(inp.Rec.F==0) inp.Rec.F=1.14;
    if(inp.Rec.F<0) throw_error_RF("read_parameter_file", "Choose fudge >=0", 1);

    parser_read(pfc, "include correction function", ival, found, show_params);
    if(found) inp.Rec.include_CF=(bool)ival;

    parser_read(pfc, "A2s1s", dval, found, show_params);
    if(found && dval>0.0) inp.Rec.A2s1s=dval;
    
    //==============================================================================================
    // different non-standard cases
    //----------------------------------------------------------------------------------------------
    // annihilation
    //----------------------------------------------------------------------------------------------
    parser_read(pfc, "f_ann", dval, found, show_params);
    if(found) inp.Rec.f_ann=dval;
    if(inp.Rec.f_ann<0) throw_error_RF("read_parameter_file", "f_ann >=0", 1);
    
    //----------------------------------------------------------------------------------------------
    // decay
    //----------------------------------------------------------------------------------------------
    parser_read(pfc, "f_dec", dval, found, show_params);
    //if(found) inp.Rec.f_dec=min(1.0, max(0.0, dval));    // to make sure f_dec >=0 and f_dec<=1
    if(found) inp.Rec.f_dec=dval;
    if(inp.Rec.f_dec<0) throw_error_RF("read_parameter_file", "f_dec >=0", 1);

    if(inp.Rec.f_dec>0)
    {
        inp.Rec.eval_ion_Tr=1;   // use physical evaluation of temperature for decay
        parser_read(pfc, "Gamma_dec", dval, found, show_params);
        if(found) inp.Rec.Gamma_dec=dval;
        if(inp.Rec.Gamma_dec<0) throw_error_RF("read_parameter_file", "Gamma_dec >=0", 1);
    }
    
    //----------------------------------------------------------------------------------------------
    // magnetic field heating
    //----------------------------------------------------------------------------------------------
    parser_read(pfc, "B0", dval, found, show_params);
    if(found) inp.Rec.B0=dval;
    if(inp.Rec.B0<0) throw_error_RF("read_parameter_file", "B0 >=0", 1);

    if(inp.Rec.B0>0.0)
    {
        parser_read(pfc, "include turbulent decay", ival, found, show_params);
        if(found) inp.Rec.turb=(bool)ival;
        
        parser_read(pfc, "include ambipolar diffusion", ival, found, show_params);
        if(found) inp.Rec.ambi=(bool)ival;

        if(inp.Rec.turb || inp.Rec.ambi)
        {
            parser_read(pfc, "nB", dval, found, show_params);
            if(found) inp.Rec.nB=dval;
            if(inp.Rec.nB<-3.0) throw_error_RF("read_parameter_file", "nB > -3.0", 1);
            
            parser_read(pfc, "Paoletti-Finelli Lorentz", ival, found, show_params);
            if(found) inp.Rec.KK_or_PF_Lorentz=(bool)ival;

            inp.Rec.eval_ion_Tr=1;       // use physical evaluation of temperature
        }
    }

    //----------------------------------------------------------------------------------------------
    // variation of alpha and me/mp
    //----------------------------------------------------------------------------------------------
    parser_read(pfc, "Variation mode", ival, found, show_params);
    if(found) inp.Rec.RM=ival;
    if(inp.Rec.RM>6) throw_error_RF("read_parameter_file", "Variation mode <=6", 1);

    if(inp.Rec.RM>0)
    {
        parser_read(pfc, "alp/alp_ref", dval, found, show_params);
        if(found) inp.Rec.aS=dval;

        parser_read(pfc, "me/me_ref", dval, found, show_params);
        if(found) inp.Rec.mS=dval;
        
        if(inp.Rec.aS<0 && inp.Rec.mS<0){ inp.Rec.aS=inp.Rec.mS=1.0; inp.Rec.RM=0; }
        else if((inp.Rec.aS<0 && inp.Rec.mS==1.0) || (inp.Rec.aS==1.0 && inp.Rec.mS<0))
        { inp.Rec.aS=inp.Rec.mS=1.0; inp.Rec.RM=0; }
        else
        {
            parser_read(pfc, "power for (1+z)^p", dval, found, show_params);
            if(found) inp.Rec.pS=dval;

            inp.Rec.do_separate_H_He=1;
        }
    }

    //==============================================================================================
    // cleanup
    //==============================================================================================
    parser_free(pfc);
    
    //==============================================================================================
    // show read parameters
    //==============================================================================================
    if(inp.verbosity>2) inp.show();
    
    return;
}

//====================================================================================================================
void write_Recfast_output(string oname, const vector<double> &zarr,
                          const vector<double> &Xe_H, const vector<double> &Xe_He,
                          const vector<double> &Xe, const vector<double> &TM,
                          int verbosity)
{
    if(verbosity>-1)
        cout << " Recfast++ :: write_Recfast_output : writing output into: " << oname << endl;

    ofstream ofile(oname.c_str());
    ofile.precision(8);
    
    ofile << "###############################################################" << endl;
    ofile << "# Output from " << RECFASTPP_VERSION << ". The Colums are:     " << endl;
    ofile << "# z | Xe = Ne/NH | Xe_H = Ne_H/NH | Xe_He = Ne_He/NH | TM [K]  " << endl;
    ofile << "###############################################################" << endl;
    
    for(int i=0; i<(int)zarr.size(); i++)
        ofile << zarr[i] << " " << Xe[i] << " " << Xe_H[i] << " " << Xe_He[i] << " " << TM[i] << endl;
    
    ofile.close();
    
    return;
}

//====================================================================================================================
//
// main program
//
//====================================================================================================================
int main(int narg, char *args[])
{
    //====================================================================================
    // System call to clear screen 
    //====================================================================================
    system("clear");
    
    if(narg < 2) throw_error_RF("Recfast-error", "usage $ ./Recfast++ parameters.dat", 1);
    
    string fname=args[narg-1];

    //====================================================================================
    // reading parameters
    //====================================================================================
    main_Settings InputPars;
    read_parameter_file(fname, InputPars);
    show_recombination_settings(InputPars.Cosmo, InputPars.Rec);
    
    //====================================================================================
    // running Recfast part
    //====================================================================================
    if(InputPars.verbosity>-1)
        cout << " Recfast++ :: Entering computation (" << RECFASTPP_VERSION << ")" << endl;
    
    //====================================================================================
    // main code call to get Recfast++ recombination history
    //====================================================================================
    vector<double> zarr, Xe_H, Xe_He, Xe, TM;
    Xe_frac(InputPars.Cosmo, InputPars.Rec, zarr, Xe_H, Xe_He, Xe, TM);

    //====================================================================================
    // write outputs to file
    //====================================================================================
    string oname=InputPars.path+"Xe_Recfast++";
    if(InputPars.Rec.f_ann!=0.0) oname+=".DM_annihilations";
    if(InputPars.Rec.f_dec!=0.0) oname+=".DM_decay";
    if(InputPars.Rec.include_CF==1) oname+=".Rec_corrs_CT2010";
    
    if(InputPars.Rec.B0>0.0)
    {
        oname+=".PMF_heating";
        if(InputPars.Rec.turb) oname+=".turb";
        if(InputPars.Rec.ambi) oname+=".ambi";
        if(InputPars.Rec.KK_or_PF_Lorentz) oname+=".PF_Lorentz";
    }
    
    if((InputPars.Rec.aS!=1.0 || InputPars.Rec.pS!=0.0) && InputPars.Rec.aS>0.0) oname+=".alp";
    if((InputPars.Rec.mS!=1.0 || InputPars.Rec.pS!=0.0) && InputPars.Rec.mS>0.0) oname+=".me";
    if(InputPars.Rec.pS!=0.0) oname+=".power";
    if(InputPars.Rec.RM==1) oname+=".Boltz";
    else if(InputPars.Rec.RM==2) oname+=".sigT";
    else if(InputPars.Rec.RM==3) oname+=".A2s1s";
    else if(InputPars.Rec.RM==4) oname+=".phot";
    else if(InputPars.Rec.RM==5) oname+=".Ly-a";
    else if(InputPars.Rec.RM==6) oname+=".all";
    
    if(InputPars.Rec.eval_ion_Tr) oname+=".Tr";
    oname+=InputPars.addname;
    
    write_Recfast_output(oname, zarr, Xe_H, Xe_He, Xe, TM, InputPars.verbosity);

    if(InputPars.verbosity>-1) cout << " Recfast++ :: Run completed. Exiting. " << endl;

    return 0;
}

//====================================================================================================================
//====================================================================================================================
