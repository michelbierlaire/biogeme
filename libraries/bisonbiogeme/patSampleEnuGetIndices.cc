//-*-c++-*------------------------------------------------------------
//
// File name : patSampleEnuGetIndices.cc
// Author :    Michel Bierlaire
// Date :      Sun Dec 16 15:59:07 2007
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patDisplay.h"
#include "patSampleEnuGetIndices.h"
#include "patErrOutOfRange.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"

patSampleEnuGetIndices::patSampleEnuGetIndices(patBoolean util, 
					       patULong na, 
					       patULong ne) :

  withUtilities(util),
  nAlt(na),
  nExpressions(ne) {

  DEBUG_MESSAGE("ALT = " << na) ;
  DEBUG_MESSAGE("ZHENG DATA = " << ne) ;
}

patULong patSampleEnuGetIndices::getIndexProba(patULong alt, 
					       patError*& err) const {
  if (alt >= nAlt) {
    err = new patErrOutOfRange<patULong>(alt,0,nAlt-1) ;
    WARNING(err->describe()) ;
    return patBadId ;
  }
  if (withUtilities) {
    return (2 + nAlt + 2 * alt) ;
  }
  else {
    return (2 + 2 * alt) ;
  }
}

patULong patSampleEnuGetIndices::getIndexResid(patULong alt, 
					       patError*& err) const {
  if (alt >= nAlt) {
    err = new patErrOutOfRange<patULong>(alt,0,nAlt-1) ;
    WARNING(err->describe()) ;
    return patBadId;
  }
  if (withUtilities) {
    return (3 + nAlt + 2 * alt) ;
  }
  else {
    return (3 + 2 * alt) ;
  }
  
}

patULong patSampleEnuGetIndices::getIndexUtil(patULong alt, patError*& err) const {
  if (!withUtilities) {
    err = new patErrMiscError("Utilities cannot be accessed in the context of mixtures of models") ;
    WARNING(err->describe()) ;
    return patBadId ;
  }
  if (alt >= nAlt) {
    err = new patErrOutOfRange<patULong>(alt,0,nAlt-1) ;
    WARNING(err->describe()) ;
    return patBadId;
  }
  if (withUtilities) {
    return (2 + alt) ;
  }

  return patBadId ;
}

patULong patSampleEnuGetIndices::getIndexExpr(patULong expr, patError*& err) const {
  if (expr >= nExpressions) {
    err = new patErrOutOfRange<patULong>(expr,0,nExpressions-1) ;
    WARNING(err->describe()) ;
    return patBadId ;
  }
  if (withUtilities) {
    return (3 + 3 * nAlt + expr) ;
  }
  else {
    return (3 + 2 * nAlt + expr) ;
  }

}

patULong patSampleEnuGetIndices::getNbrOfColumns() const {
  if (withUtilities) {
    return (3 + 3 * nAlt + nExpressions) ;
  }
  else {
    return (3 + 2 * nAlt + nExpressions) ;
  }

}

patULong patSampleEnuGetIndices::getIndexZhengFosgerau(patOneZhengFosgerau* t, patError*& err) const {
  if (t == NULL) {
    err = new patErrNullPointer("patOneZhengFosgerau") ;
    WARNING(err->describe()) ;
    return patBadId ;
  }
  if (t->isProbability()) {
    patULong i = getIndexProba(t->getAltInternalId(),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patBadId ;
    }
    return i ;
  }
  else {
    if (t->expressionIndex == patBadId) {
      stringstream str ;
      str << "Index of expression is not initialized: " << *t ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patBadId ;
    }
    patULong i = getIndexExpr(t->expressionIndex,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patBadId ;
    }
    return i ;
  }
  
}
