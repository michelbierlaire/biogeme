//-*-c++-*------------------------------------------------------------
//
// File name : bioArithMultinaryExpression.cc
// Author :    Michel Bierlaire
// Date :      Wed Apr 13 19:51:34 2011
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrOutOfRange.h"
#include "bioArithMultinaryExpression.h"
#include "bioArithConstant.h"
#include "bioExpressionRepository.h"
#include "patErrMiscError.h"

bioArithMultinaryExpression::bioArithMultinaryExpression(bioExpressionRepository* rep,
							 patULong par,
							 vector<patULong> l,
							 patError*& err) : 
  bioExpression(rep,par),listOfChildrenIds(l) {

  for (vector<patULong>::iterator i = listOfChildrenIds.begin() ;
       i != listOfChildrenIds.end() ;
       ++i) {
    bioExpression* theExpr = theRepository->getExpression(*i) ;
    if (theExpr == NULL) {
      stringstream str ;
      str << "No expression with ID " << *i ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;

    }
    //theExpr->setParent(getId(),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    listOfChildren.push_back(theExpr) ;
    relatedExpressions.push_back(theExpr) ;
  }
}

bioArithMultinaryExpression::~bioArithMultinaryExpression() {

}

bioExpression* bioArithMultinaryExpression::getChild(patULong index, patError*& err) const {
  if (index >= listOfChildren.size()) {
    err = new patErrOutOfRange<patULong>(index,0,listOfChildren.size()-1) ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  return listOfChildren[index] ;
}


patString bioArithMultinaryExpression::getExpression(patError*& err) const {

  stringstream str ;

  str << endl << getOperatorName() << endl ;
  str << "++++++++++++++++" << endl ;
  for (vector<bioExpression*>::const_iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    str << i-listOfChildren.begin()+1 << ": " ;
      str << (*i)->getExpression(err) << endl ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patString();
    }
  }
  return patString(str.str()) ;
    
}


patULong bioArithMultinaryExpression::getNumberOfOperations() const {
  patULong count = 1 ;
  for (vector<bioExpression*>::const_iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    count += (*i)->getNumberOfOperations() ;
  }
  return (count) ;
}

patBoolean bioArithMultinaryExpression::dependsOf(patULong aLiteralId) const {
  for (vector<bioExpression*>::const_iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    if ((*i)->dependsOf(aLiteralId)) {
      return patTRUE ;
    }
  }
  return patFALSE ;

}



patBoolean bioArithMultinaryExpression::containsAnIterator() const {
  for (vector<bioExpression*>::const_iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    if ((*i)->containsAnIterator()) {
      return patTRUE ;
    }
  }
  return patFALSE ;
}

patBoolean bioArithMultinaryExpression::containsAnIteratorOnRows() const {
  for (vector<bioExpression*>::const_iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    if ((*i)->containsAnIteratorOnRows()) {
      return patTRUE ;
    }
  }
  return patFALSE ;
}

patBoolean bioArithMultinaryExpression::containsAnIntegral() const {
  for (vector<bioExpression*>::const_iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    if ((*i)->containsAnIntegral()) {
      return patTRUE ;
    }
  }
  return patFALSE ;

}

patBoolean bioArithMultinaryExpression::containsASequence() const {
  for (vector<bioExpression*>::const_iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    if ((*i)->containsASequence()) {
      return patTRUE ;
    }
  }
  return patFALSE ;

}



void bioArithMultinaryExpression::simplifyZeros(patError*& err) {
  vector<patULong> newPointersIds(listOfChildren.size(),patBadId) ;
  vector<bioExpression*> newPointers(listOfChildren.size(),NULL) ;
  for (patULong i = 0 ;
       i < listOfChildren.size() ;
       ++i) {
    if (listOfChildren[i]->isStructurallyZero()) {
      newPointers[i] = theRepository->getZero() ;
      newPointersIds[i] = newPointers[i]->getId() ;
    }
    else if (listOfChildren[i]->isConstant()) {
      patReal value = listOfChildren[i]->getValue(patFALSE, patLapForceCompute, err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
        return ;
      }
      newPointers[i]= new bioArithConstant(theRepository,getId(),value) ;
      newPointersIds[i] = newPointers[i]->getId() ;
      
    }
    else {
      listOfChildren[i]->simplifyZeros(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
        return ;
      }
    }
  }

  for (patULong i = 0 ;
       i < listOfChildren.size() ;
       ++i) {
    if (newPointers[i] != NULL) {
      listOfChildren[i] = newPointers[i] ;
      listOfChildrenIds[i] = newPointersIds[i] ;
    }
  }

}





vector<patULong> bioArithMultinaryExpression::getChildrenIds() const {
  return listOfChildrenIds ;
}

void bioArithMultinaryExpression::collectExpressionIds(set<patULong>* s) const {
  s->insert(getId()) ;
  for (vector<bioExpression*>::const_iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    (*i)->collectExpressionIds(s)  ;
  }
  for (vector<bioExpression*>::const_iterator i = relatedExpressions.begin() ;
       i != relatedExpressions.end() ;
       ++i) {
    (*i)->collectExpressionIds(s) ;
  }
}


patString bioArithMultinaryExpression::check(patError*& err) const {
  stringstream str ;
  for (patULong i = 0 ; i < listOfChildren.size() ; ++i) {
    if (listOfChildren[i]->getId() != listOfChildrenIds[i]) {
      str << "Incompatible IDS for children: " << listOfChildren[i]->getId() << " and " << listOfChildrenIds[i] << endl ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patString() ;
    }
  }
  return patString() ;
}

