//-*-c++-*------------------------------------------------------------
//
// File name : patZhengTest.cc
// Author :    Michel Bierlaire
// Date :      Sun Dec 16 10:48:58 2007
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <sstream>
#include "patZhengTest.h"
#include "patVariables.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"
#include "patNormalPdf.h"

patZhengTest::patZhengTest(patVariables* aVar,
			   patVariables* aResid,
			   patReal bw,
			   patError*& err) :
  theVar(aVar),
  theResid(aResid),
  bandwidth(bw) {
  if (aVar == NULL || aResid == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return ;
  }
  if (theVar->size() != theResid->size()) {
    stringstream str ;
    str << "Incompatible sizes: " << theVar->size() << " variables and " << theResid->size() << " residuals" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
}

patReal patZhengTest::compute(patError*& err) {

  patReal numerator(0.0) ;
  patReal denominator(0.0) ;

  for (patULong i = 0 ; i < theVar->size() ; ++i) {

    patReal xi = (*theVar)[i] ;
    patReal ri = (*theResid)[i] ;
    for (patULong j = 0 ; j < theVar->size() ; ++j) {
      if (i != j) {
	patReal xj = (*theVar)[j] ;
	patReal rj = (*theResid)[j] ;
	
	patReal density = patNormalPdf()((xi - xj)/bandwidth) ;
	numerator += density * ri * rj  ;
	denominator += 2.0 * density * density * ri * ri * rj * rj ;
      }
    }
  }
  return (numerator / sqrt(denominator)) ;
}
