//-*-c++-*------------------------------------------------------------
//
// File name : patRandomParameter.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Tue Mar  4 10:23:23 2003
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patConst.h"
#include "patDisplay.h"
#include "patRandomParameter.h"
#include "patBetaLikeParameter.h"

patRandomParameter::patRandomParameter() :
  name("???"),location(NULL),scale(NULL), panel(patFALSE), massAtZero(patFALSE)
{
  
}
ostream& operator<<(ostream &str, const patRandomParameter& x) {
  str << x.name ;
  if (x.location != NULL) {
    str << " Location {" << *(x.location) << "}" ;
  }
  if (x.scale != NULL) {
    str << " StdDev {" << *(x.scale) << "}" ;
  }
  for (vector<patRandomParameter*>::const_iterator i = 
	 x.correlatedParameters.begin() ;
       i != x.correlatedParameters.end() ;
       ++i) {
    if (i == x.correlatedParameters.begin()) {
      str <<" [" ;
    }
    else {
      str << "," ;
    }
    str << (*i)->name ;
  }
  if (!x.correlatedParameters.empty()) {
    str << "]" ;
  }
  return str ;
}

patBoolean operator<(const patRandomParameter& p1, 
		     const patRandomParameter& p2) {
  if ((p1.location == NULL) || p2.location == NULL) {
    WARNING("Null pointer...."); 
    return patFALSE ;
  }
  if ((p1.scale == NULL) || p2.scale == NULL) {
    WARNING("Null pointer...."); 
    return patFALSE ;
  }
  
  if (p1.location->id == p2.location->id) {
    return (p1.scale->id > p2.scale->id) ;
  }
  return (p1.location->id > p2.location->id) ;
}
