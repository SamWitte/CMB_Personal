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

#ifndef EVALODE_H
#define EVALODE_H

//========================================================================================
// Set variables of evalode and access them
//========================================================================================
void Set_evalode_Vars(const RecPars &RecInputs);
const RecPars& Get_evalode_Vars();

//========================================================================================
// evaluation of Recfast-system
//========================================================================================
void evaluate_Recfast_System(double z, double y[], double fvec[], int n);

//========================================================================================
// wrapper for ODE-solver
//========================================================================================
void fcn(int *neq, double *z, double *y, double *f);

//========================================================================================
// wrapper for ODE-solver with rescaled derivative for Xe
// this is important to stitch the recfast solution to the output of the multi-level code
//========================================================================================
void set_rescaling(double ff);
void fcn_rescaled(int *neq, double *z, double *y, double *f);

#endif
//========================================================================================
//========================================================================================
