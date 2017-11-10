//-*-c++-*------------------------------------------------------------
//
// File name : patImportance.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Tue Jun  3 09:16:11 2003
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patImportance.h"

patImportance::patImportance(patImportance::type p) : theType(p) {
 
}

patImportance::type patImportance::operator()() const  {
  return theType ;
}

patBoolean operator<(const patImportance& i1, const patImportance& i2) {
  return (i1.theType < i2.theType) ;
}

patBoolean operator<=(const patImportance& i1, const patImportance& i2) {
  return (i1.theType <= i2.theType) ;
}
