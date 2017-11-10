//-*-c++-*------------------------------------------------------------
//
// File name : patBetaLikeParameter.cc
// Author :    Michel Bierlaire
// Date :      Wed May 16 16:32:52 2001
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <iostream>
#include "patBetaLikeParameter.h"
#include "patConst.h"

patBetaLikeParameter::patBetaLikeParameter() :
  name("???"),
  defaultValue(patMaxReal),
  lowerBound(-patMaxReal),
  upperBound(patMaxReal),
  isFixed(patTRUE),
  hasDiscreteDistribution(patFALSE),
  estimated(patMaxReal),
  index(unsigned(-1)),
  id(unsigned(-1))
{

}


ostream& operator<<(ostream &str, const patBetaLikeParameter& x) {
  str << "x[" ;
  str << x.index << "]=" << x.name << " (" << x.defaultValue << "," 
      <<  x.estimated << ") [" << x.lowerBound << "," << x.upperBound << "] " ;
  if (x.isFixed) {
    str << "FIXED " ;
  }
  else {
    str << "FREE " ;
  }
  str << "ID=" << x.id ;

  return str ;
}

