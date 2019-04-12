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

#ifndef MAGNETIC_FIELDS_H
#define MAGNETIC_FIELDS_H

//========================================================================================
// evaluation of ODE terms for magnetic heating
//========================================================================================
void evaluate_magnetic_field_heating_RF(double z, double Hz, double fHe, double NH,
                                        double xp, double xHep, double Tm, double nbar,
                                        double B0, double nB, double fvec[]);

//========================================================================================
// evaluation of ODE terms for collisions
//========================================================================================
void evaluate_magnetic_field_heating_RF_colls(double z, double Hz, double fHe, double NH,
                                              double xp, double xHep, double Tm,
                                              double fvec[]);

#endif

//========================================================================================
//========================================================================================
