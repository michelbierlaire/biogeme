//-*-c++-*------------------------------------------------------------
//
// File name : patZhengTest.h
// Author :    \URL[Michel Bierlaire]{http://transp-or2.epfl.ch}
// Date :      Sun Dec 16 10:45:08 2007
//
//--------------------------------------------------------------------

#ifndef patZhengTest_h
#define patZhengTest_h

#include "patType.h"
#include "patError.h"
#include "patVariables.h"

/**
   Compute the Theng test 

Zheng, J. X. (1996) A consistent test of functional form via nonparametric
estimation techniques Journal of Econometrics 75(2), 263{
289.

This code is derived from the Ox code by Mogens Fosgerau

 */

class patZhengTest {
  
public:
  patZhengTest(patVariables* aVar,
	       patVariables* aResid,
	       patReal bw,
	       patError*& err) ;

  patReal compute(patError*& err) ;

private:
  patVariables* theVar ;
  patVariables* theResid ;
  patReal bandwidth ;

};
#endif
