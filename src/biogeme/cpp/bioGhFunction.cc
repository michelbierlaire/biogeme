//-*-c++-*------------------------------------------------------------
//
// File name : bioGhFunction.cc
// Author :    Michel Bierlaire
// Date :      Thu Apr  8 12:02:48 2010
// Modified for biogemepython 3.0: Wed May  9 16:13:51 2018
//
//--------------------------------------------------------------------

#include <cmath>
#include "bioGhFunction.h"

bioGhFunction::bioGhFunction() {

}


std::vector<bioReal> bioGhFunction::getUnweightedValue(bioReal x) {
  if (x*x >= log(bioMaxReal)) {
    return std::vector<bioReal>(getSize(),bioMaxReal) ;
  }
  std::vector<bioReal> result = getValue(x) ;
  for (std::vector<bioReal>::iterator i = result.begin() ;
       i != result.end() ;
       ++i) {
    if (*i != bioMaxReal) {
      (*i) *= exp(x*x) ;
      if (!std::isfinite(*i)) {
	(*i) = bioMaxReal ;
      }
    }
  }
  return result ;
}


