//-*-c++-*------------------------------------------------------------
//
// File name : bioExpression.cc
// @date   Fri Apr 13 08:47:59 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExpression.h"
#include "bioDebug.h"
#include <sstream>
bioExpression::bioExpression() : parameters(NULL), fixedParameters(NULL), data(NULL), dataMap(NULL), draws(NULL), sampleSize(0), numberOfDraws(0), numberOfDrawVariables(0), rowIndex(NULL), individualIndex(NULL) {
}

bioExpression::~bioExpression() {
  resetDerivatives() ;
}


void bioExpression::resetDerivatives() {
  theDerivatives.clear() ;
}

void bioExpression::setParameters(std::vector<bioReal>* p) {
  parameters = p ;
}

void bioExpression::setFixedParameters(std::vector<bioReal>* p) {
  fixedParameters = p ;
}

void bioExpression::setData(std::vector< std::vector<bioReal> >* d) {
  data = d ;
}

void bioExpression::setDataMap(std::vector< std::vector<bioUInt> >* dm) {
  dataMap = dm ;
}

void bioExpression::setDraws(std::vector< std::vector< std::vector<bioReal> > >* d) {
  draws = d ;
  if (draws != NULL) {
    sampleSize = draws->size() ;
  }
  if (sampleSize > 0) {
    numberOfDraws = (*draws)[0].size() ;
  }
  if (numberOfDraws > 0) {
    numberOfDrawVariables = (*draws)[0][0].size() ;
  }
}

void bioExpression::setRowIndex(bioUInt* d) {
  rowIndex = d ;
  for (std::vector<bioExpression*>::iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    (*i)->setRowIndex(d) ;
  }
}

void bioExpression::setIndividualIndex(bioUInt* d) {
  individualIndex = d ;
  for (std::vector<bioExpression*>::iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    (*i)->setIndividualIndex(d) ;
  }
}

void bioExpression::setMissingData(bioReal md) {
  missingData = md ;
  for (std::vector<bioExpression*>::iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    (*i)->setMissingData(md) ;
  }
}


void bioExpression::setDrawIndex(bioUInt* d) {
  for (std::vector<bioExpression*>::iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    (*i)->setDrawIndex(d) ;
  }
}

void bioExpression::setRandomVariableValuePtr(bioUInt rvId, bioReal* v) {
  for (std::vector<bioExpression*>::iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    (*i)->setRandomVariableValuePtr(rvId, v) ;
  }
}

bioReal bioExpression::getValue() {
  const bioDerivatives* r = getValueAndDerivatives(std::vector<bioUInt>(), false,false) ;
  return r->f ;
}

bioBoolean bioExpression::containsLiterals(std::vector<bioUInt> literalIds) const {
  for (std::vector<bioExpression*>::const_iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    if ((*i)->containsLiterals(literalIds)) {
      return true ;
    }
  }
  return false ;
}

std::map<bioString,bioReal> bioExpression::getAllLiteralValues() {
  std::map<bioString,bioReal>  m ;
  for (std::vector<bioExpression*>::const_iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    std::map<bioString,bioReal> cm = (*i)->getAllLiteralValues() ;
    m.insert(cm.begin(), cm.end());
  }
  return m ;  
}

