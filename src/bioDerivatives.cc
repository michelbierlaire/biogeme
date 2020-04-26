//-*-c++-*------------------------------------------------------------
//
// File name : bioDerivatives.h
// @date   Fri Apr 13 10:31:21 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include <iostream>
#include "bioDerivatives.h"

/**
 Constructor .cc
 @param n number of variables
*/
bioDerivatives::bioDerivatives(bioUInt n) : g(n), h(n,std::vector<bioReal>(n)) {

}

bioUInt bioDerivatives::getSize() const {
  return g.size();
}

void bioDerivatives::setDerivativesToZero() {
  std::fill(g.begin(),g.end(),0.0) ;
  std::fill(h.begin(),h.end(),g) ;
}

void bioDerivatives::setGradientToZero() {
  std::fill(g.begin(),g.end(),0.0) ;
}

std::ostream& operator<<(std::ostream &str, const bioDerivatives& x) {
  str << "f = " << x.f << std::endl ;
  str << "g = [" ; 
  for (std::vector<bioReal>::const_iterator i = x.g.begin() ; i != x.g.end() ; ++i) {
    if (i != x.g.begin()) {
      str << ", " ;
    }
    str << *i ;
  }
  str << "]" << std::endl ;
  str << "h = [ " ;
  for (std::vector<std::vector<bioReal> >::const_iterator row = x.h.begin() ; row != x.h.end() ; ++row) {
    if (row != x.h.begin()) {
      str << std::endl ;
    }
    str << " [ " ;
    for (std::vector<bioReal>::const_iterator col = row->begin() ; col != row->end() ; ++col) {
      if (col != row->begin()) {
	str << ", " ;
      }
      str << *col ;
    }
    str << " ] " << std::endl ;
  }
  return str ;
}
