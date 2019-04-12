//========================================================================================
// Author: Jens Chluba & Luke Hart
// Last modification: Jan 2017
// CITA, University of Toronto
// All rights reserved.
//========================================================================================
// 04.01.2017: added new routines for separate H and HeI correction functions (LH)

//========================================================================================
// This module allows to add the recombination corrections as described by 
// Chluba & Thomas 2010.
//========================================================================================

#ifndef REC_CORRS_CT_H
#define REC_CORRS_CT_H

//========================================================================================
// to load the correction factors
//========================================================================================
void read_recombination_correction(int mflag=1);

//========================================================================================
// correction factor 1+DNe/Ne as explained by Rubino-Martin et al. 2010 and LH & JC, 2017
//========================================================================================
void Recombination_correction_factor(double z, double &Xe, double &Xe_H, double &Xe_He);

#endif

//========================================================================================
//========================================================================================
