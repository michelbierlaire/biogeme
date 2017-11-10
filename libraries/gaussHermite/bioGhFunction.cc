//-*-c++-*------------------------------------------------------------
//
// File name : bioGhFunction.cc
// Author :    Michel Bierlaire
// Date :      Thu Apr  8 12:02:48 2010
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patMath.h"
#include "bioGhFunction.h"
#include "patDisplay.h"

bioGhFunction::bioGhFunction() {

}

vector<patReal> bioGhFunction::getUnweightedValue(patReal x, patError*& err) {
  if (x*x >= patLogMaxReal::the()) {
    return vector<patReal>(getSize(),patMaxReal) ;
  }
  vector<patReal> result = getValue(x,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return vector<patReal>() ;
  }
  for (vector<patReal>::iterator i = result.begin() ;
       i != result.end() ;
       ++i) {
    if (*i != patMaxReal) {
      (*i) *= exp(x*x) ;
      if (!patFinite(*i)) {
	(*i) = patMaxReal ;
      }
    }
  }
  //DEBUG_MESSAGE("RESULT=" << result) ;
  return result ;
}


