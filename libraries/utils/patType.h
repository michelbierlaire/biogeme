//-*-c++-*------------------------------------------------------------
//
// File name : patType.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri Dec 18 23:29:08 1998
//
//--------------------------------------------------------------------

#ifndef patType_h
#define patType_h

#include <cfloat>

#include <vector>
#include <cmath>

#include "patString.h"


/**
 */
typedef short patBoolean;

#ifdef LONGDOUBLE

typedef long double patReal ;
typedef long double patPythonReal ;

const patReal patEPSILON = LDBL_EPSILON ;

const patReal patMaxReal = LDBL_MAX ;

const patReal patMinReal = LDBL_MIN ;

const patReal patOne(1.0);

const patReal patZero(0.0);

const patReal patSQRT_EPSILON = powl(patEPSILON,patReal(0.5)) ;

#else

typedef double patReal ;
typedef double patPythonReal ;

const patReal patEPSILON = DBL_EPSILON ;

const patReal patMaxReal = DBL_MAX ;

const patReal patMinReal = DBL_MIN ;

const patReal patOne = 1.0;

const patReal patZero = 0.0;

const patReal patSQRT_EPSILON = pow(patEPSILON,0.5) ;

#endif
/**
 */
typedef unsigned long patULong  ;

/**
 */
typedef unsigned long long patULongLong ;

/**
 */
typedef time_t patUnitTime ;

/**
 */
typedef vector<patString> patStringVector ;
#endif
