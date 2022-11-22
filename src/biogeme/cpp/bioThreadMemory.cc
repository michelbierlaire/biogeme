//-*-c++-*------------------------------------------------------------
//
// File name : bioThreadMemory.cc
// @date   Mon Apr 23 09:09:37 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioDebug.h"
#include "bioThreadMemory.h"
#include "bioExceptions.h"

bioThreadMemory::bioThreadMemory() {
  
}

void bioThreadMemory::resize(bioUInt nThreads, bioUInt dim) {
  inputStructures.clear() ;
  inputStructures.resize(nThreads) ;
  for (bioUInt i = 0 ; i < numberOfThreads() ; ++i) {
    inputStructures[i].grad.resize(dim) ;
    inputStructures[i].hessian.resize(dim,inputStructures[i].grad) ;
    inputStructures[i].bhhh.resize(dim,inputStructures[i].grad) ;
  }
}

bioThreadMemory::~bioThreadMemory() {
}


bioThreadArg* bioThreadMemory::getInput(bioUInt t) {
  for (bioUInt i = 0 ; i < numberOfThreads() ; ++i) {
    inputStructures[i].threadId = i ;
  }
  if (t >= inputStructures.size()) {
    throw bioExceptOutOfRange<bioUInt>(__FILE__,__LINE__,t,0,inputStructures.size()  - 1) ;
  }
  if (t >= loglikes.size()) {
    throw bioExceptOutOfRange<bioUInt>(__FILE__,__LINE__,t,0,loglikes.size()  - 1) ;
  }
  inputStructures[t].theLoglike = loglikes[t] ;
  if (!weights.empty()) {
    if (t >= weights.size()) {
      throw bioExceptOutOfRange<bioUInt>(__FILE__,__LINE__,t,0,weights.size()  - 1) ;
    }
    inputStructures[t].theWeight = weights[t] ;
  }
  else {
    inputStructures[t].theWeight.resetExpression() ;
  }
  return &(inputStructures[t]) ;
}


void bioThreadMemory::setLoglike(std::vector<bioString> f) {
  loglikes.resize(numberOfThreads()) ;
  for (bioUInt i= 0 ; i < numberOfThreads() ; ++i) {
    loglikes[i].setExpression(f) ;
  }
}

void bioThreadMemory::setWeight(std::vector<bioString> w) {
  weights.resize(numberOfThreads()) ;
  for (bioUInt i= 0 ; i < numberOfThreads() ; ++i) {
    weights[i].setExpression(w) ;
  }
}



bioUInt bioThreadMemory::numberOfThreads() {
  return inputStructures.size() ;
}
bioUInt bioThreadMemory::dimension() {
  return inputStructures[0].grad.size() ;
}

void bioThreadMemory::setParameters(std::vector<bioReal>* p) {
  for (std::vector<bioFormula>::iterator i = loglikes.begin() ;
       i != loglikes.end() ;
       ++i) {
    i->setParameters(p) ;
  }
  for (std::vector<bioFormula>::iterator i = weights.begin() ;
       i != weights.end() ;
       ++i) {
    i->setParameters(p) ;
  }
}

void bioThreadMemory::setFixedParameters(std::vector<bioReal>* p) {
  for (std::vector<bioFormula>::iterator i = loglikes.begin() ;
       i != loglikes.end() ;
       ++i) {
    i->setFixedParameters(p) ;
  }
  for (std::vector<bioFormula>::iterator i = weights.begin() ;
       i != weights.end() ;
       ++i) {
    i->setFixedParameters(p) ;
  }

}

void bioThreadMemory::setData(std::vector< std::vector<bioReal> >* d) {
  for (std::vector<bioFormula>::iterator i = loglikes.begin() ;
       i != loglikes.end() ;
       ++i) {
    i->setData(d) ;
  }
  for (std::vector<bioFormula>::iterator i = weights.begin() ;
       i != weights.end() ;
       ++i) {
    i->setData(d) ;
  }

}

void bioThreadMemory::setMissingData(bioReal md) {
  for (std::vector<bioFormula>::iterator i = loglikes.begin() ;
       i != loglikes.end() ;
       ++i) {
    i->setMissingData(md) ;
  }
  for (std::vector<bioFormula>::iterator i = weights.begin() ;
       i != weights.end() ;
       ++i) {
    i->setMissingData(md) ;
  }
}


void bioThreadMemory::setDataMap(std::vector< std::vector<bioUInt> >* dm) {
  for (std::vector<bioFormula>::iterator i = loglikes.begin() ;
       i != loglikes.end() ;
       ++i) {
    i->setDataMap(dm) ;
  }
  for (std::vector<bioFormula>::iterator i = weights.begin() ;
       i != weights.end() ;
       ++i) {
    i->setDataMap(dm) ;
  }
}

void bioThreadMemory::setDraws(std::vector< std::vector< std::vector<bioReal> > >* d) {
  for (std::vector<bioFormula>::iterator i = loglikes.begin() ;
       i != loglikes.end() ;
       ++i) {
    i->setDraws(d) ;
  }
  for (std::vector<bioFormula>::iterator i = weights.begin() ;
       i != weights.end() ;
       ++i) {
    i->setDraws(d) ;
  }
  
}
