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

#ifndef DM_ANNIHILATION_DECAY_H
#define DM_ANNIHILATION_DECAY_H

//===========================================================================================
// evaluation of ODE terms given heating rate
//===========================================================================================
void evaluate_ionization_heating_excitation_terms(double z, double Hz,
                                                  double xp, double xHep,
                                                  double CHe, double CH,
                                                  double dE_dz, double fvec[]);

//===========================================================================================
// evaluation of ODE terms for annihilation
//===========================================================================================
void evaluate_DM_annihilation_terms(double z, double Hz,
                                    double xp, double xHep,
                                    double CHe, double CH,
                                    double fDM,
                                    double fvec[]);

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
                             double fvec[]);

#endif
//===========================================================================================
//===========================================================================================
