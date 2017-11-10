//-*-c++-*------------------------------------------------------------
//
// File name : bioCountOperations.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Sun Sep 13 17:53:53 2009
//
//--------------------------------------------------------------------

#include <ostream>
#include "bioCountOperations.h"

ostream& operator<<(ostream &str, const bioCountOperations& x) {

  str << "Function: [before] " << x.fctBeforeSimplification 
      << "\t[after] " << x.fctAfterSimplification 
      << "\t[Savings] " << patULong(100 * (x.fctBeforeSimplification-x.fctAfterSimplification)/x.fctBeforeSimplification) << "%" << endl ;
  str << "Gradient: [before] " << x.gradBeforeSimplification 
      << "\t[after] " << x.gradAfterSimplification  ;
  if (x.gradBeforeSimplification > 0) {
    str << "\t[Savings] " << patULong(100 *(x.gradBeforeSimplification-x.gradAfterSimplification)/x.gradBeforeSimplification ) << "%" << endl ;
  }
  if (x.fctAfterSimplification > 0) {
    str << "Ratio   : [before] " << patReal(x.gradBeforeSimplification)/patReal(x.fctBeforeSimplification) << "\t[after]" << patReal(x.gradAfterSimplification)/patReal(x.fctAfterSimplification) << endl ;
  }
  return str ;
}
