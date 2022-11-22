//-*-c++-*------------------------------------------------------------
//
// File name : bioThreadMemoryOneExpression.cc
// @date   Tue Oct 19 14:26:03 2021
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#include "bioDebug.h"
#include "bioThreadMemoryOneExpression.h"
#include "bioExceptions.h"

bioThreadMemoryOneExpression::bioThreadMemoryOneExpression() : inputStructures(1) {
  
}

void bioThreadMemoryOneExpression::resize(bioUInt nThreads, bioUInt dim, bioUInt dataSize) {
  inputStructures.resize(nThreads) ;
}

bioThreadMemoryOneExpression::~bioThreadMemoryOneExpression() {
}


bioThreadArgOneExpression* bioThreadMemoryOneExpression::getInput(bioUInt t) {
  for (bioUInt i = 0 ; i < numberOfThreads() ; ++i) {
    inputStructures[i].threadId = i ;
  }
  
  if (t >= inputStructures.size()) {
    throw bioExceptOutOfRange<bioUInt>(__FILE__,
				       __LINE__,
				       t,
				       0,
				       inputStructures.size() - 1) ;
  }
  
  if (t >= formulasPerThread.size()) {
    throw bioExceptOutOfRange<bioUInt>(__FILE__,
				       __LINE__,
				       t,
				       0,
				       formulasPerThread.size() - 1) ;
  }
  
  inputStructures[t].theFormula = formulasPerThread[t] ;
  
  return &(inputStructures[t]) ;
}


void bioThreadMemoryOneExpression::setFormula(std::vector<bioString> f) {
  formulasPerThread.resize(numberOfThreads()) ;
  for (bioUInt i= 0 ; i < numberOfThreads() ; ++i) {
    formulasPerThread[i].setExpression(f) ;
  }
}

bioUInt bioThreadMemoryOneExpression::numberOfThreads() {
  return inputStructures.size() ;
}
bioUInt bioThreadMemoryOneExpression::dimension() {
  bioUInt n = inputStructures[0].theDerivatives.getSize() ;
  return n ;
}

void bioThreadMemoryOneExpression::setParameters(std::vector<bioReal>* p) {
  for (std::vector<bioFormula>::iterator i = formulasPerThread.begin() ;
       i != formulasPerThread.end() ;
       ++i) {
    i->setParameters(p) ;
  }
}

void bioThreadMemoryOneExpression::setFixedParameters(std::vector<bioReal>* p) {
  for (std::vector<bioFormula>::iterator i = formulasPerThread.begin() ;
       i != formulasPerThread.end() ;
       ++i) {
    i->setFixedParameters(p) ;
  }
}

void bioThreadMemoryOneExpression::setData(std::vector< std::vector<bioReal> >* d) {
  for (std::vector<bioFormula>::iterator i = formulasPerThread.begin() ;
       i != formulasPerThread.end() ;
       ++i) {
    i->setData(d) ;
  }
}

void bioThreadMemoryOneExpression::setMissingData(bioReal md) {
  for (std::vector<bioFormula>::iterator i = formulasPerThread.begin() ;
       i != formulasPerThread.end() ;
       ++i) {
    i->setMissingData(md) ;
  }
}


void bioThreadMemoryOneExpression::setDataMap(std::vector< std::vector<bioUInt> >* dm) {
  for (std::vector<bioFormula>::iterator i = formulasPerThread.begin() ;
       i != formulasPerThread.end() ;
       ++i) {
    i->setDataMap(dm) ;
  }
}

void bioThreadMemoryOneExpression::setDraws(std::vector< std::vector< std::vector<bioReal> > >* d) {
  for (std::vector<bioFormula>::iterator i = formulasPerThread.begin() ;
       i != formulasPerThread.end() ;
       ++i) {
    i->setDraws(d) ;
  }
}

void bioThreadMemoryOneExpression::clear() {
  inputStructures.clear() ;
  formulasPerThread.clear() ;
}
