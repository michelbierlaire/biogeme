//-*-c++-*------------------------------------------------------------
//
// File name : patSecondDerivatives.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Mon May  1 10:33:37 2006
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <iostream>
#include "patSecondDerivatives.h"

patSecondDerivatives::patSecondDerivatives(unsigned long nBeta) :
    secondDerivBetaBeta(nBeta,vector<patReal>(nBeta,0.0)) {

};

void patSecondDerivatives::setToZero() {
  for (vector<vector<patReal> >::iterator i = secondDerivBetaBeta.begin() ;
       i != secondDerivBetaBeta.end() ;
       ++i) {
    fill(i->begin(),i->end(),0.0) ;
  }
}


ostream& operator<<(ostream &str, const patSecondDerivatives& x) {
  for (vector<vector<patReal> >::const_iterator i = x.secondDerivBetaBeta.begin() ;
       i != x.secondDerivBetaBeta.end() ;
       ++i) {
    for (vector<patReal>::const_iterator j = i->begin() ;
	 j != i->end() ;
	 ++j) {
      if (j != i->begin()) {
	str << '\t' ;
      }
      str << *j ;
    }
    str << endl ;
  }
  return str ;
}
