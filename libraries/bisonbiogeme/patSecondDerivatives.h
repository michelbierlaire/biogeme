//-*-c++-*------------------------------------------------------------
//
// File name : patSecondDerivatives.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sun Apr 30 08:02:03 2006
//
//--------------------------------------------------------------------

#ifndef patSecondDerivatives_h
#define patSecondDerivatives_h

#include "patType.h"

/**
   Data structure to compute the second derivatives of the model
 */
class patSecondDerivatives {

 public:
  patSecondDerivatives(unsigned long nBeta) ;
  void setToZero() ;
  // Second derivatives
  vector<vector<patReal> > secondDerivBetaBeta ;


};

/**
 */
ostream& operator<<(ostream &str, const patSecondDerivatives& x) ;


#endif
