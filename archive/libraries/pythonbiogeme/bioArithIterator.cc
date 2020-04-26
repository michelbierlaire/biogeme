//-*-c++-*------------------------------------------------------------
//
// File name : bioArithIterator.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Jun 16 12:47:18  2009
//
//--------------------------------------------------------------------

#include <sstream>

#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patError.h"

#include "bioArithIterator.h"
#include "bioIteratorInfoRepository.h"

/*!
*/
bioArithIterator::bioArithIterator(bioExpressionRepository* rep,
				   patULong par,
				   patULong left,
				   patString it,
				   patError*& err) 
  : bioArithUnaryExpression(rep, par,left,err), 
    theIteratorName(it) {

  bioIteratorSpan theSpan(it,0) ;
  setCurrentSpan(theSpan) ;
  theIteratorType = bioIteratorInfoRepository::the()->getType(theIteratorName,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

}

bioArithIterator::~bioArithIterator() {}


patString bioArithIterator::getExpression(patError*& err) const {
  
  stringstream ss ;
  if (isSum()) {
    ss << "Sum(" ;
  }
  else if (isProd()) {
    ss << "Prod(" ;
  }
  patString theInfo =  bioIteratorInfoRepository::the()->getInfo(theIteratorName,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  patString theExpression =  child->getExpression(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  ss << theInfo << "," << theExpression << ")";
  return patString(ss.str()) ;
}



patString bioArithIterator::theIterator() const {
  return theIteratorName ;
}



patBoolean bioArithIterator::containsAnIterator() const {
  return patTRUE ;
}

patBoolean bioArithIterator::containsAnIteratorOnRows() const {
  return theIteratorType == ROW ;
}
