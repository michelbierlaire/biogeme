//-*-c++-*------------------------------------------------------------
//
// File name : bioThreadMemorySimul.cc
// @date   Mon Mar  8 14:33:35 2021
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioDebug.h"
#include "bioThreadMemorySimul.h"
#include "bioExceptions.h"
#include "bioSeveralExpressions.h"
bioThreadMemorySimul::bioThreadMemorySimul() {
  
}

void bioThreadMemorySimul::resize(bioUInt nThreads) {
  inputStructures.resize(nThreads) ;
  theFormulas.resize(nThreads) ;
  for (bioUInt i = 0 ; i < nThreads ; ++i) {
    inputStructures[i].results.erase(inputStructures[i].results.begin(),
				     inputStructures[i].results.end()) ;
  }
}

bioThreadMemorySimul::~bioThreadMemorySimul() {
}


bioThreadArgSimul* bioThreadMemorySimul::getInput(bioUInt t) {
  if (t >= inputStructures.size()) {
    throw bioExceptOutOfRange<bioUInt>(__FILE__,__LINE__,t,0,inputStructures.size()  - 1) ;
  }
  if (t >= theFormulas.size()) {
    throw bioExceptOutOfRange<bioUInt>(__FILE__,__LINE__,t,0,theFormulas.size()  - 1) ;
  }
  for (bioUInt i = 0 ; i < numberOfThreads() ; ++i) {
    inputStructures[i].threadId = i ;
  }
  inputStructures[t].theFormulas = theFormulas[t] ;
  bioThreadArgSimul* ptr = &(inputStructures[t]) ;
  return ptr ;
}


void bioThreadMemorySimul::setFormulas(std::vector<std::vector<bioString> > vectOfExpressionsStrings) {
  theFormulas.resize(numberOfThreads()) ;
  for (bioUInt i = 0 ; i < numberOfThreads() ; ++i) {
    theFormulas[i].setExpressions(vectOfExpressionsStrings) ;
  }
}

bioUInt bioThreadMemorySimul::numberOfThreads() {
  return inputStructures.size() ;
}
bioUInt bioThreadMemorySimul::dimension() {
  return inputStructures[0].results.size() ;
}

void bioThreadMemorySimul::setParameters(std::vector<bioReal>* p) {
  for (std::vector<bioSeveralFormulas>::iterator i = theFormulas.begin() ;
       i != theFormulas.end() ;
       ++i) {
    i->setParameters(p) ;
  }
}

void bioThreadMemorySimul::setFixedParameters(std::vector<bioReal>* p) {
  for (std::vector<bioSeveralFormulas>::iterator i = theFormulas.begin() ;
       i != theFormulas.end() ;
       ++i) {
    i->setFixedParameters(p) ;
  }
}

void bioThreadMemorySimul::setData(std::vector< std::vector<bioReal> >* d) {
  for (std::vector<bioSeveralFormulas>::iterator i = theFormulas.begin() ;
       i != theFormulas.end() ;
       ++i) {
    i->setData(d) ;
  }
}

void bioThreadMemorySimul::setMissingData(bioReal md) {
  for (std::vector<bioSeveralFormulas>::iterator i = theFormulas.begin() ;
       i != theFormulas.end() ;
       ++i) {
    i->setMissingData(md) ;
  }
}


void bioThreadMemorySimul::setDataMap(std::vector< std::vector<bioUInt> >* dm) {
  for (std::vector<bioSeveralFormulas>::iterator i = theFormulas.begin() ;
       i != theFormulas.end() ;
       ++i) {
    i->setDataMap(dm) ;
  }
}

void bioThreadMemorySimul::setDraws(std::vector< std::vector< std::vector<bioReal> > >* d) {
  for (std::vector<bioSeveralFormulas>::iterator i = theFormulas.begin() ;
       i != theFormulas.end() ;
       ++i) {
    i->setDraws(d) ;
  }
}
