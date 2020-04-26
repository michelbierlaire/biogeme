//-*-c++-*------------------------------------------------------------
//
// File name : panelbiogeme.cc
// @date   Sun May 20 18:54:53 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "panelbiogeme.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <pthread.h>
#include "bioExceptions.h"
#include "bioDebug.h"
#include "bioThreadMemory.h"
#include "bioExpression.h"

// Dealing with exceptions across threads
static std::exception_ptr theExceptionPtr = nullptr ;

void *computePanelFunctionForThread( void *ptr );

panelbiogeme::panelbiogeme(): nbrOfThreads(1),calculateHessian(false),calculateBhhh(false),theThreadMemory(NULL) {
}

panelbiogeme::~panelbiogeme() {
  if (theThreadMemory != NULL) {
    delete(theThreadMemory) ;
  }
}
  

bioReal panelbiogeme::calculateLikelihood(std::vector<bioReal>& betas,
					  std::vector<bioReal>& fixedBetas) {

  
  theThreadMemory->setParameters(&betas) ;
  theThreadMemory->setFixedParameters(&fixedBetas) ;
  bioReal result = applyTheFormula() ;
  return result ;
}


bioReal panelbiogeme::applyTheFormula(  std::vector<bioReal>* g,
					std::vector< std::vector<bioReal> >* h,
					std::vector< std::vector<bioReal> >* bh) {
  if ( g != NULL) {
    if (g->size() != theThreadMemory->dimension()) {
      std::stringstream str ;
      str << "Gradient: inconsistent dimensions " << g->size() << " and " << theThreadMemory->dimension() ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
  }
  if ( h != NULL) {
    if (h->size() != theThreadMemory->dimension()) {
      std::stringstream str ;
      str << "Hessian: inconsistent dimensions " << h->size() << " and " << theThreadMemory->dimension() ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
  }
  if ( bh != NULL) {
    if (bh->size() != theThreadMemory->dimension()) {
      std::stringstream str ;
      str << "BHHH: inconsistent dimensions " << bh->size() << " and " << theThreadMemory->dimension() ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
  }
  
  // Calculate the size of the block  of data to be sent to each thread
  bioUInt sizeOfEachBlock = ceil(bioReal(theDataMap.size())/bioReal(nbrOfThreads)) ;
  std::vector<bioThreadArg*> theInput(nbrOfThreads) ;
  std::vector<pthread_t> theThreads(nbrOfThreads) ;
  if (theThreadMemory == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"thread memory") ;
  }
  for (bioUInt thread = 0 ; thread < nbrOfThreads ; ++thread) {
    theInput[thread] = theThreadMemory->getInput(thread) ;
    if (theInput[thread] == NULL) {
      throw bioExceptNullPointer(__FILE__,__LINE__,"thread memory") ;
    }
    theInput[thread]->calcGradient = (g != NULL) ;
    theInput[thread]->calcHessian = (h != NULL) ;
    theInput[thread]->calcBhhh = (bh != NULL) ;
      
    bioUInt diagnostic = pthread_create(&(theThreads[thread]),
					NULL,
					computePanelFunctionForThread,
					(void*) theInput[thread]) ;
    if (diagnostic != 0) {
      std::stringstream str ;
      str << "Error " << diagnostic << " in creating thread " << thread << "/" << nbrOfThreads ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
  }

  if (theThreadMemory->dimension() < literalIds.size()) {
    std::stringstream str ;
    str << " Memory should be reserved for dimension " << literalIds.size() << " and not " << theThreadMemory->dimension() ;
    throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  }
  
  bioReal result(0.0) ;
  if (g != NULL) {
    std::fill(g->begin(),g->end(),0.0) ;
    if (h != NULL) {
      std::fill(h->begin(),h->end(),*g) ;
    }
    if (bh != NULL) {
      std::fill(bh->begin(),bh->end(),*g) ;
    }
  }
  for (bioUInt thread = 0 ; thread < nbrOfThreads ; ++thread) {
    pthread_join( theThreads[thread], NULL);
    if (theExceptionPtr != nullptr) {
      std::rethrow_exception(theExceptionPtr);
    }
    result += theInput[thread]->result ;
    if (g != NULL) {
      for (bioUInt i = 0 ; i < g->size() ; ++i) {
	(*g)[i] += (theInput[thread]->grad)[i] ;
	if ( h != NULL) {
	  for (bioUInt j = i ; j < g->size() ; ++j) {
	    (*h)[i][j] += (theInput[thread]->hessian)[i][j] ;
	  }
	}
	if (bh != NULL) {
	  for (bioUInt j = i ; j < g->size() ; ++j) {
	    (*bh)[i][j] += (theInput[thread]->bhhh)[i][j] ;
	  }
	}
      }
    }
  }

  // Fill the symmetric part of the matrices
  if (h != NULL) {
    for (bioUInt i = 0 ; i < g->size() ; ++i) {
      for (bioUInt j = i+1 ; j < g->size() ; ++j) {
	(*h)[j][i] = (*h)[i][j] ;
      }
    }
  }
  if (bh != NULL) {
    for (bioUInt i = 0 ; i < g->size() ; ++i) {
      for (bioUInt j = i+1 ; j < g->size() ; ++j) {
	(*bh)[j][i] = (*bh)[i][j] ;
      }
    }
  }
  
  return result ;
}


bioReal panelbiogeme::calculateLikeAndDerivatives(std::vector<bioReal>& betas,
					     std::vector<bioReal>& fixedBetas,
					     std::vector<bioUInt>& betaIds,
					     std::vector<bioReal>& g,
					     std::vector< std::vector<bioReal> >& h,
					     std::vector< std::vector<bioReal> >& bh,
					     bioBoolean hessian,
					     bioBoolean bhhh) {


  literalIds = betaIds ;
  if (theThreadMemory->dimension() != literalIds.size()) {
    prepareData() ;
  }
  calculateHessian = hessian ;
  calculateBhhh = bhhh ;

  if (theThreadMemory == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"thread memory") ;
  }
  theThreadMemory->setParameters(&betas) ;
  theThreadMemory->setFixedParameters(&fixedBetas) ;

  std::vector< std::vector<bioReal> >* hptr = (calculateHessian) ? &h : NULL ;
  std::vector< std::vector<bioReal> >* bhptr = (calculateBhhh) ? &bh : NULL ;

  
  bioReal r = applyTheFormula(&g,hptr,bhptr) ;
  return r ;


}


void panelbiogeme::setExpressions(std::vector<bioString> ll,
			     std::vector<bioString> w,
			     bioUInt t) {
  

  theLoglike = ll ;
  theWeight = w ;
  nbrOfThreads = t ;
  prepareData() ;
  

}

void *computePanelFunctionForThread(void* fctPtr) {
  try {
    bioThreadArg *input = (bioThreadArg *) fctPtr;
    bioReal w ;
    input->result = 0.0 ;
    if (input->calcGradient) {
      std::fill(input->grad.begin(),input->grad.end(),0.0) ;
      if (input->calcHessian) {
	std::fill(input->hessian.begin(),input->hessian.end(),input->grad) ;
      }
      if (input->calcBhhh) {
	std::fill(input->bhhh.begin(),input->bhhh.end(),input->grad) ;
      }
    }
    bioUInt individual ;
    bioExpression* theLoglike = input->theLoglike->getExpression() ;
    theLoglike->setData(input->data) ;
    theLoglike->setDataMap(input->dataMap) ;
    theLoglike->setMissingData(input->missingData) ;
    theLoglike->setIndividualIndex(&individual) ;
    if (input->theWeight != NULL) {
      input->theWeight->setData(input->data) ;
      input->theWeight->setDataMap(input->dataMap) ;
      input->theWeight->setMissingData(input->missingData) ;
      input->theWeight->setIndividualIndex(&individual) ;
    }
    for (individual = input->startData ;
	 individual < input->endData ;
	 ++individual) {
      if (input->theWeight != NULL) {
	w = input->theWeight->getExpression()->getValue() ;
      }
      
      bioDerivatives* fgh = theLoglike->getValueAndDerivatives(*input->literalIds,
							       input->calcGradient,
							       input->calcHessian) ;
      
      

      if (input->theWeight == NULL) {
	input->result += fgh->f ;
	for (bioUInt i = 0 ; i < input->grad.size() ; ++i) {
	  (input->grad)[i] += fgh->g[i] ;
	  if (input->calcHessian) {
	    for (bioUInt j = 0 ; j < input->grad.size() ; ++j) {
	      (input->hessian)[i][j] += fgh->h[i][j] ;
	    }
	  }
	  if (input->calcBhhh) {
	    for (bioUInt j = i ; j < input->grad.size() ; ++j) {
	      (input->bhhh)[i][j] += fgh->g[i] * fgh->g[j] ;
	    }
	  }
	}
      }
      else {
	input->result += w * fgh->f ;
	for (bioUInt i = 0 ; i < input->grad.size() ; ++i) {
	  (input->grad)[i] += w * fgh->g[i] ;
	  if (input->calcHessian) {
	    for (bioUInt j = 0 ; j < input->grad.size() ; ++j) {
	      (input->hessian)[i][j] += w * fgh->h[i][j] ;
	    }
	  }
	  if (input->calcBhhh) {
	    for (bioUInt j = i ; j < input->grad.size() ; ++j) {
	      (input->bhhh)[i][j] += w * fgh->g[i] * fgh->g[j] ;
	    }
	  }
	}
      }
    }
    input->theLoglike->setIndividualIndex(NULL) ;
    if (input->theWeight != NULL) {
      input->theWeight->setIndividualIndex(NULL) ;
    }
  }
  catch(...)  {
    theExceptionPtr = std::current_exception() ;
  }

  return NULL ;
}

void panelbiogeme::prepareMemoryForThreads(bioBoolean force) {
  if (theThreadMemory != NULL) {
    if (theThreadMemory->numberOfThreads() != nbrOfThreads ||
	theThreadMemory->dimension() < literalIds.size() ||
	force) {
      delete(theThreadMemory) ;
      theThreadMemory = NULL ;
    }
  }
  if (theThreadMemory == NULL) {
    theThreadMemory = new bioThreadMemory(nbrOfThreads,literalIds.size()) ;
    theThreadMemory->setLoglike(theLoglike) ;
    if (!theWeight.empty()) {
      theThreadMemory->setWeight(theWeight) ;
    }
  }
}

void panelbiogeme::simulateFormula(std::vector<bioString> formula,
			      std::vector<bioReal>& beta,
			      std::vector<bioReal>& fixedBeta,
			     std::vector< std::vector<bioReal> >& data,
			     std::vector<bioReal>& results) {
  std::stringstream str ;
  str << "Simulation not yet implemented for panel data" ;
  throw bioExceptions(__FILE__,__LINE__,str.str()) ;

  bioFormula theFormula(formula) ;
  theFormula.setParameters(&beta) ;
  theFormula.setFixedParameters(&fixedBeta) ;

  bioUInt N = data.size() ;
  results.resize(N) ;
  theFormula.setData(&data) ;
  theFormula.setMissingData(missingData) ;
  bioUInt row ;
  theFormula.setRowIndex(&row) ;
  for (row = 0 ;
       row < N ;
       ++row) {
     // DEBUG_MESSAGE("Set row " << row) ;
     // for (bioUInt i = 0 ; i < data[row].size() ; ++i) {
     //   DEBUG_MESSAGE(data[row][i]) ;
     // }
    results[row] = theFormula.getExpression()->getValue() ;
  }
  theFormula.setRowIndex(NULL) ;
  return ;
}

void panelbiogeme::setData(std::vector< std::vector<bioReal> >& d) {
  theData = d ;
}

void panelbiogeme::setDataMap(std::vector< std::vector<bioUInt> >& dm) {
  theDataMap = dm ;
}

void panelbiogeme::setMissingData(bioReal md) {
  missingData = md ;
}


void panelbiogeme::setDraws(std::vector< std::vector< std::vector<bioReal> > >& draws) {
  theDraws = draws ;
}


void panelbiogeme::prepareData() {
  prepareMemoryForThreads() ;
  if (theThreadMemory == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"thread memory") ;
  }
  theThreadMemory->setData(&theData) ;
  theThreadMemory->setDataMap(&theDataMap) ;
  theThreadMemory->setMissingData(missingData) ;
  if (!theDraws.empty()) {
    theThreadMemory->setDraws(&theDraws) ;
  }

  if (theThreadMemory->dimension() < literalIds.size()) {
    std::stringstream str ;
    str << " Memory should be reserved for dimension " << literalIds.size() << " and not " << theThreadMemory->dimension() ;
    throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  }
  
  // Prepare the input for the threads

  // Calculate the size of the block  of data to be sent to each thread
  bioUInt sizeOfEachBlock = ceil(bioReal(theDataMap.size())/bioReal(nbrOfThreads)) ;
  // For small data sets, there may be more threads than number of blocks.
  bioUInt numberOfBlocks = ceil(bioReal(theDataMap.size()) / bioReal(sizeOfEachBlock)) ;
  if (numberOfBlocks < nbrOfThreads) {
    nbrOfThreads = numberOfBlocks ;
  }

  theInput.resize(nbrOfThreads,NULL) ;
  for (bioUInt thread = 0 ; thread < nbrOfThreads ; ++thread) {
    theInput[thread] = theThreadMemory->getInput(thread) ;
    if (theInput[thread] == NULL) {
      throw bioExceptNullPointer(__FILE__,__LINE__,"thread memory") ;
    }
    theInput[thread]->data = &theData ;
    theInput[thread]->dataMap = &theDataMap ;
    theInput[thread]->startData = thread * sizeOfEachBlock ;
    theInput[thread]->endData = (thread == nbrOfThreads-1) ? theDataMap.size() : (thread+1) * sizeOfEachBlock ;
    theInput[thread]->missingData = missingData ;
    theInput[thread]->literalIds = &literalIds ;
  }  
  
}
