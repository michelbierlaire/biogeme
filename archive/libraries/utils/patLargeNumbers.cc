//-*-c++-*------------------------------------------------------------
//
// File name : patLargeNumbers.cc
// Author :    Michel Bierlaire
// Date :      Mon Aug 31 21:52:44 2009
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include "patLargeNumbers.h"
#include <cmath>

patLargeNumbers::patLargeNumbers(patULong n) : theNumber(n) {

}
ostream& operator<<(ostream &str, const patLargeNumbers& x) {
  
  if (x.theNumber >= 1000 * 1000* 1000) {
    str << floor(patReal(x.theNumber) / (1000.0 * 1000.0 * 1000.0)) << " billions" ;
  } 
  else if (x.theNumber >= 1000 * 1000) {
    str << floor(patReal(x.theNumber) / (1000.0 * 1000.0)) << " millions" ;
  }
  else if (x.theNumber >= 1000) {
    str << floor(patReal(x.theNumber) / (1000.0)) << " thousands" ;
  }
  else {
    str << x.theNumber ;
  }
  return str ;
  
}

