//-*-c++-*------------------------------------------------------------
//
// File name : biogeme.cc
// @date   Wed Apr  4 18:11:29 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "biogeme.h"
#include "bioTypes.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <pthread.h>
#include "bioMemoryManagement.h"
#include "bioExceptions.h"
#include "bioDebug.h"
#include "bioThreadMemory.h"
#include "bioThreadMemorySimul.h"
#include "bioExpression.h"
#include "bioSeveralExpressions.h"
//#include "bioCfsqp.h"

// Dealing with exceptions across threads
static std::exception_ptr theExceptionPtr = nullptr ;

void *computeFunctionForThread( void *ptr );

void *simulFunctionForThread( void *ptr );

biogeme::biogeme(): nbrOfThreads(1),
		    fixedBetasDefined(false),
		    calculateHessian(false),
		    calculateBhhh(false),
		    panel(false),
		    forceDataPreparation(true) {
}

biogeme::~biogeme() {
  //DEBUG_MESSAGE("********* CALL DESTRUCTOR **************") ;
  //bioMemoryManagement::the()->releaseAllMemory() ;
}

void biogeme::setPanel(bioBoolean p) {
  panel = p ;
}

bioReal biogeme::calculateLikelihood(std::vector<bioReal> betas,
				     std::vector<bioReal> fixedBetas) {

  ++nbrFctEvaluations ;
  if (forceDataPreparation || (theThreadMemory.dimension() != literalIds.size())) {
    prepareData() ;
    forceDataPreparation = false ;
  }
  theThreadMemory.setParameters(&betas) ;
  theThreadMemory.setFixedParameters(&fixedBetas) ;
  bioReal result = applyTheFormula() ;
  return result ;
}


bioReal biogeme::applyTheFormula(  std::vector<bioReal>* g,
				   std::vector< std::vector<bioReal> >* h,
				   std::vector< std::vector<bioReal> >* bh) {

  if ( g != NULL) {
    if (g->size() != theThreadMemory.dimension()) {
      std::stringstream str ;
      str << "Gradient: inconsistent dimensions " << g->size() << " and " << theThreadMemory.dimension() ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
  }
  if ( h != NULL) {
    if (h->size() != theThreadMemory.dimension()) {
      std::stringstream str ;
      str << "Hessian: inconsistent dimensions " << h->size() << " and " << theThreadMemory.dimension() ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
  }
  if ( bh != NULL) {
    if (bh->size() != theThreadMemory.dimension()) {
      std::stringstream str ;
      str << "BHHH: inconsistent dimensions " << bh->size() << " and " << theThreadMemory.dimension() ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
  }

  
  //  std::vector<bioThreadArg*> theInput(nbrOfThreads) ;
  std::vector<pthread_t> theThreads(nbrOfThreads) ;
  for (bioUInt thread = 0 ; thread < nbrOfThreads ; ++thread) {
    if (theInput[thread] == NULL) {
      throw bioExceptNullPointer(__FILE__,__LINE__,"thread") ;
    }
    theInput[thread]->calcGradient = (g != NULL) ;
    theInput[thread]->calcHessian = (h != NULL) ;
    theInput[thread]->calcBhhh = (bh != NULL) ;

    bioUInt diagnostic = pthread_create(&(theThreads[thread]),
					NULL,
					computeFunctionForThread,
					(void*) theInput[thread]) ;

    if (diagnostic != 0) {
      std::stringstream str ;
      str << "Error " << diagnostic << " in creating thread " << thread << "/" << nbrOfThreads ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
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

  if (!std::isfinite(result)) {
    result = -std::numeric_limits<bioReal>::max() ;
  }

  if (g != NULL) {
    for (bioUInt i = 0 ; i < g->size() ; ++i) {
      if (!std::isfinite((*g)[i])) {
	(*g)[i] = -std::numeric_limits<bioReal>::max() ;
      }
      if ( h != NULL) {
	for (bioUInt j = i ; j < g->size() ; ++j) {
	  if (!std::isfinite((*h)[i][j])) {
	    (*h)[i][j] = -std::numeric_limits<bioReal>::max() ;
	  }
	}
      }
      if ( bh != NULL) {
	for (bioUInt j = i ; j < g->size() ; ++j) {
	  if (!std::isfinite((*bh)[i][j])) {
	    (*bh)[i][j] = -std::numeric_limits<bioReal>::max() ;
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


bioReal biogeme::calculateLikeAndDerivatives(std::vector<bioReal> betas,
 					     std::vector<bioReal> fixedBetas,
 					     std::vector<bioUInt> betaIds,
 					     bioReal* g,
 					     bioReal* h,
 					     bioReal* bh,
 					     bioBoolean hessian,
 					     bioBoolean bhhh) {
  
  int n = betas.size() ;

   ++nbrFctEvaluations ;
  literalIds = betaIds ;
  if (forceDataPreparation || (theThreadMemory.dimension() != literalIds.size())) {
    prepareData() ;
    forceDataPreparation = false ;
  }
  calculateHessian = hessian ;
  calculateBhhh = bhhh ;

  theThreadMemory.setParameters(&betas) ;
  theThreadMemory.setFixedParameters(&fixedBetas) ;


  // This is inefficient. It is first implemented to check that the
  // memory leak is fixed. Afterwards, everything should be
  // implemented as linear array.

  std::vector<bioReal> gmem(n) ;
  std::vector< std::vector<bioReal> > hmem(n, std::vector<bioReal>(n)) ;
  std::vector< std::vector<bioReal> > bhmem(n, std::vector<bioReal>(n)) ;

  std::vector< std::vector<bioReal> >* hptr = (calculateHessian) ? &hmem : NULL ;
  std::vector< std::vector<bioReal> >* bhptr = (calculateBhhh) ? &bhmem : NULL ;

  bioReal r = applyTheFormula(&gmem,hptr,bhptr) ;
  
  for (int i = 0 ; i < n ; ++i) {
    g[i] = gmem[i] ;
    if (calculateHessian || calculateBhhh) {
      for (int j = i ; j < n ; j++) {
	if (calculateHessian) {
	  h[i*n+j] = h[j*n+i] = hmem[i][j] ;
	}
	if (calculateBhhh) {
	  bh[i*n+j] = bh[j*n+i] = bhmem[i][j] ;
	}
      }
    }
  }

  return r ;


}

// bioReal biogeme::calculateLikeAndDerivatives(std::vector<bioReal>& betas,
// 					     std::vector<bioReal>& fixedBetas,
// 					     std::vector<bioUInt>& betaIds,
// 					     std::vector<bioReal>& g,
// 					     std::vector< std::vector<bioReal> >& h,
// 					     std::vector< std::vector<bioReal> >& bh,
// 					     bioBoolean hessian,
// 					     bioBoolean bhhh) {


//   ++nbrFctEvaluations ;
//   literalIds = betaIds ;
//   if (forceDataPreparation || (theThreadMemory.dimension() != literalIds.size())) {
//     prepareData() ;
//     forceDataPreparation = false ;
//   }
//   calculateHessian = hessian ;
//   calculateBhhh = bhhh ;

//   theThreadMemory.setParameters(&betas) ;
//   theThreadMemory.setFixedParameters(&fixedBetas) ;


//   std::vector< std::vector<bioReal> >* hptr = (calculateHessian) ? &h : NULL ;
//   std::vector< std::vector<bioReal> >* bhptr = (calculateBhhh) ? &bh : NULL ;

  
//   bioReal r = applyTheFormula(&g,hptr,bhptr) ;
//   return r ;


// }


void biogeme::setExpressions(std::vector<bioString> ll,
			     std::vector<bioString> w,
			     bioUInt t) {

  theLoglikeString.erase(theLoglikeString.begin(),theLoglikeString.end()) ;
  for (bioUInt i = 0 ; i < ll.size() ; ++i) {
    if (std::find(theLoglikeString.begin(),theLoglikeString.end(),ll[i]) == theLoglikeString.end()) {
      theLoglikeString.push_back(ll[i]) ;
    }
  }

  theWeightString.erase(theWeightString.begin(),theWeightString.end()) ;
  for (bioUInt i = 0 ; i < w.size() ; ++i) {
    if (std::find(theWeightString.begin(),theWeightString.end(),w[i]) == theWeightString.end()) {
      theWeightString.push_back(w[i]) ;
    }
  }
  theLoglikeString = ll ;
  theWeightString = w ;
  nbrOfThreads = t ;
  prepareData() ;

}

void *computeFunctionForThread(void* fctPtr) {
  try {
    bioThreadArg *input = (bioThreadArg *) fctPtr;
    bioReal w(1.0) ;
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

    bioExpression* myLoglike = input->theLoglike.getExpression() ;
    if (input->panel) {
      // Panel data
      bioUInt individual ;
      myLoglike->setIndividualIndex(&individual) ;
      for (individual = input->startData ;
	   individual < input->endData ;
	   ++individual) {
	if (input->theWeight.isDefined()) {
	  w = input->theWeight.getExpression()->getValue() ;
	}
      
	const bioDerivatives* fgh = myLoglike->getValueAndDerivatives(*input->literalIds,
								      input->calcGradient,
								      input->calcHessian) ;
      
	if (!input->theWeight.isDefined()) {
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
      
    }
    else {
      // No panel data
      bioUInt row ;
      if (myLoglike == NULL) {
	throw bioExceptNullPointer(__FILE__,__LINE__,"thread memory") ;
      }
      myLoglike->setIndividualIndex(&row) ;
      myLoglike->setRowIndex(&row) ;
      if (input->theWeight.isDefined()) {
	input->theWeight.setIndividualIndex(&row) ;
	input->theWeight.setRowIndex(&row) ;
      }

      for (row = input->startData ;
	   row < input->endData ;
	   ++row) {

	// if (row % 100 == 0) {
	//   DEBUG_MESSAGE("Row " << row) ;
	// }
	try {
	  if (input->theWeight.isDefined()) {
	    w = input->theWeight.getExpression()->getValue() ;
	  }

	  //	  DEBUG_MESSAGE("getValueAndDerivatives") ;
	  const bioDerivatives* fgh = myLoglike->getValueAndDerivatives(*input->literalIds,
						  input->calcGradient,
						  input->calcHessian) ;

	  //	  DEBUG_MESSAGE("getValueAndDerivatives: done") ;
	  if (!input->theWeight.isDefined()) {
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
	catch(bioExceptions& e) {
	  std::stringstream str ;
	  str << "Error for data entry " << row << " : " << e.what() ;
	  throw bioExceptions(__FILE__,__LINE__,str.str()) ;
	}
      }
    }
    //DEBUG_MESSAGE("End of loop on rows") ;
    input->theLoglike.setRowIndex(NULL) ;
    input->theLoglike.setIndividualIndex(NULL) ;
    if (input->theWeight.isDefined()) {
      input->theWeight.setRowIndex(NULL) ;
      input->theWeight.setIndividualIndex(NULL) ;
    }
  }
  catch(...)  {
    theExceptionPtr = std::current_exception() ;
  }

  //DEBUG_MESSAGE("RETURN") ;
  return NULL ;
}


void *simulFunctionForThread(void* fctPtr) {
  try {
    bioThreadArgSimul *input = (bioThreadArgSimul *) fctPtr;
    bioSeveralExpressions* expressions = input->theFormulas.getExpressions() ;
    if (input->panel) {
      bioUInt individual ;
      expressions->setIndividualIndex(&individual) ;
      for (individual = input->startData ;
	   individual < input->endData ;
	   ++individual) {
	try {
	  std::vector<bioReal > res = expressions->getValues() ;
	  input->results.push_back(res) ;
	}
	catch(bioExceptions& e) {
	  std::stringstream str ;
	  str << "Error in simulating panel data: " << e.what() ;
	  throw bioExceptions(__FILE__,__LINE__,str.str()) ;
	}
      }
    }
    else {
      bioUInt row ;
      if (expressions == NULL) {
	throw bioExceptNullPointer(__FILE__,__LINE__,"thread memory") ;
      }
      expressions->setIndividualIndex(&row) ;
      expressions->setRowIndex(&row) ;
      
      for (row = input->startData ;
	   row < input->endData ;
	   ++row) {
	try {
	  std::vector<bioReal > res = expressions->getValues() ;
	  input->results.push_back(res) ;
	}
	catch(bioExceptions& e) {
	  std::stringstream str ;
	  str << "Error for data entry " << row << " : " << e.what() ;
	  throw bioExceptions(__FILE__,__LINE__,str.str()) ;
	}
      }
    }
    input->theFormulas.setRowIndex(NULL) ;
    input->theFormulas.setIndividualIndex(NULL) ;
  }
  catch(...)  {
    theExceptionPtr = std::current_exception() ;
  }

  return NULL ;
}

void biogeme::prepareMemoryForThreads(bioBoolean force) {
  theThreadMemory.resize(nbrOfThreads,literalIds.size()) ;
  theThreadMemory.setLoglike(theLoglikeString) ;
  if (!theWeightString.empty()) {
    theThreadMemory.setWeight(theWeightString) ;
  }
  
}

void biogeme::simulateFormula(std::vector<bioString> formula,
			      std::vector<bioReal> beta,
			      std::vector<bioReal> fixedBeta,
			      std::vector< std::vector<bioReal> > data,
			      bioReal* results) {

  bioFormula theFormula ;
  theFormula.setExpression(formula) ;
  theFormula.setParameters(&beta) ;
  theFormula.setFixedParameters(&fixedBeta) ;
  if (!theDraws.empty()) {
    theFormula.setDraws(&theDraws) ;
  }  

  bioUInt N = data.size() ;
  theFormula.setData(&data) ;
  theFormula.setMissingData(missingData) ;
  bioUInt row ;
  theFormula.setRowIndex(&row) ;
  theFormula.setIndividualIndex(&row) ;
  for (row = 0 ;
       row < N ;
       ++row) {
    results[row] = theFormula.getExpression()->getValue() ;
  }
  return ;
}

bioReal biogeme::simulateSimpleFormula(std::vector<bioString> formula,
				       std::vector<bioReal> beta,
				       std::vector<bioReal> fixedBeta,
				       bioBoolean gradient,
				       bioBoolean hessian,
				       bioReal* g,
				       bioReal* h) {

  bioFormula theFormula ;
  theFormula.setExpression(formula) ;
  theFormula.setParameters(&beta) ;
  theFormula.setFixedParameters(&fixedBeta) ;
  if (!theDraws.empty()) {
    theFormula.setDraws(&theDraws) ;
  }  

  if (!gradient && !hessian) {
    return theFormula.getExpression()->getValue() ;
  }
  int n = beta.size() ;
  // The ids of th eliterals are from 0 to n-1, corresponding to the free parameters.
  std::vector<bioUInt> literalIds(n) ;
  for (int i = 0 ; i < n ; ++i) {
    literalIds[i] = i ;
  }
  const bioDerivatives* results =
    theFormula.getExpression()->getValueAndDerivatives(literalIds,
						       gradient,
						       hessian) ;

  for (int i = 0 ; i < n ; ++i) {
    g[i] = results->g[i] ;
    if (hessian) {
      for (int j = i ; j < n ; j++) {
	h[i*n+j] = h[j*n+i] = results->h[i][j] ;
      }
    }
  }
  return results->f ;
}


void biogeme::simulateSeveralFormulas(std::vector<std::vector<bioString> > formulas,
				      std::vector<bioReal> betas,
				      std::vector<bioReal> fixedBetas,
				      bioUInt t,
				      std::vector< std::vector<bioReal> > data,
				      bioReal* results) {
  nbrOfThreads = t ;
  theThreadMemorySimul.resize(nbrOfThreads) ;
  theThreadMemorySimul.setFormulas(formulas) ;
  prepareDataSimul() ;
  theThreadMemorySimul.setParameters(&betas) ;
  theThreadMemorySimul.setFixedParameters(&fixedBetas) ;
  
  std::vector<pthread_t> theThreads(nbrOfThreads) ;
  for (bioUInt thread = 0 ; thread < nbrOfThreads ; ++thread) {
    bioSeveralExpressions* theFormulas = theSimulInput[thread]->theFormulas.getExpressions() ;
    theFormulas->setData(theSimulInput[thread]->data) ;
    if (panel) {
      theFormulas->setDataMap(theSimulInput[thread]->dataMap) ;
    }
    theFormulas->setMissingData(theSimulInput[thread]->missingData) ;
    if (theSimulInput[thread] == NULL) {
      throw bioExceptNullPointer(__FILE__,__LINE__,"thread") ;
    }
    bioUInt diagnostic = pthread_create(&(theThreads[thread]),
					NULL,
					simulFunctionForThread,
					(void*) theSimulInput[thread]) ;
    
    if (diagnostic != 0) {
      std::stringstream str ;
      str << "Error " << diagnostic << " in creating thread " << thread << "/" << nbrOfThreads ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
  }

  bioUInt N = data.size() ;

  for (bioUInt thread = 0 ; thread < nbrOfThreads ; ++thread) {
    pthread_join( theThreads[thread], NULL);
    if (theExceptionPtr != nullptr) {
      std::rethrow_exception(theExceptionPtr);
    }
    if (theSimulInput[thread]->results.size() !=
	theSimulInput[thread]->endData - theSimulInput[thread]->startData) {
      std::stringstream str ;
      str << "Inconsistent dimensions: "
	  << theSimulInput[thread]->results.size()
	  << " and "
	  << theSimulInput[thread]->endData - theSimulInput[thread]->startData ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    for (bioUInt i = 0 ; i < theSimulInput[thread]->results.size() ; ++i) {
      bioUInt row = theSimulInput[thread]->startData + i ;
      for (bioUInt j = 0 ; j < theSimulInput[thread]->results[i].size() ; ++j) {
	results[j * N + row] = theSimulInput[thread]->results[i][j] ;
      }
    }
  }
  return ;
}

// void biogeme::simulateSeveralFormulas(std::vector<std::vector<bioString> > formulas,
// 				      std::vector<bioReal> beta,
// 				      std::vector<bioReal> fixedBeta,
// 				      std::vector< std::vector<bioReal> > data,
// 				      bioReal* results) {


//   bioSeveralFormulas theFormulas ;
//   theFormulas.setExpressions(formulas) ;
//   theFormulas.setParameters(&beta) ;
//   theFormulas.setFixedParameters(&fixedBeta) ;
//   if (!theDraws.empty()) {
//     theFormulas.setDraws(&theDraws) ;
//   }  

//   bioUInt N = data.size() ;
//   theFormulas.setData(&data) ;
//   theFormulas.setMissingData(missingData) ;
//   bioUInt row ;
//   theFormulas.setRowIndex(&row) ;
//   theFormulas.setIndividualIndex(&row) ;
  
//   for (row = 0 ;
//        row < N ;
//        ++row) {
//     std::vector<bioReal > res = theFormulas.getExpressions()->getValues() ;
//     for (bioUInt i = 0 ; i < res.size() ; ++i) {
//       results[i * N + row] = res[i] ;
//     }
//   }
//   return ;
// }



void biogeme::setData(std::vector< std::vector<bioReal> >& d) {
  theData = d ;
  forceDataPreparation = true ;
}

void biogeme::setDataMap(std::vector< std::vector<bioUInt> >& dm) {
  theDataMap = dm ;
  forceDataPreparation = true ;
  panel = true ;
}

void biogeme::setMissingData(bioReal md) {
  missingData = md ;
  forceDataPreparation = true ;
}

void biogeme::setDraws(std::vector< std::vector< std::vector<bioReal> > >& draws) {
  theDraws = draws ;
  forceDataPreparation = true ;
}

void biogeme::prepareData() {

  // Here, we prepare the data that do not vary from one call of the
  // functions to the next.

  prepareMemoryForThreads() ;
  theThreadMemory.setData(&theData) ;
  if (panel) {
    theThreadMemory.setDataMap(&theDataMap) ;
  }
  theThreadMemory.setMissingData(missingData) ;
  if (!theDraws.empty()) {
    theThreadMemory.setDraws(&theDraws) ;
  }

  if (theThreadMemory.dimension() < literalIds.size()) {
    std::stringstream str ;
    str << " Memory should be reserved for dimension " << literalIds.size() << " and not " << theThreadMemory.dimension() ;
    throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  }
  
  // Prepare the input for the threads

  // Calculate the size of the block  of data to be sent to each thread
  bioUInt sizeOfEachBlock ;
  bioUInt numberOfBlocks ;
  if (panel) {
    sizeOfEachBlock = ceil(bioReal(theDataMap.size())/bioReal(nbrOfThreads)) ;
    numberOfBlocks = ceil(bioReal(theDataMap.size()) / bioReal(sizeOfEachBlock)) ;
  }
  else {
    sizeOfEachBlock = ceil(bioReal(theData.size())/bioReal(nbrOfThreads)) ;
    numberOfBlocks = ceil(bioReal(theData.size()) / bioReal(sizeOfEachBlock)) ;
  }
  // For small data sets, there may be more threads than number of blocks.
  if (numberOfBlocks < nbrOfThreads) {
    nbrOfThreads = numberOfBlocks ;
  }

  theInput.resize(nbrOfThreads,NULL) ;

  for (bioUInt thread = 0 ; thread < nbrOfThreads ; ++thread) {
    theInput[thread] = theThreadMemory.getInput(thread) ;
    theInput[thread]->panel = panel ;
    if (theInput[thread] == NULL) {
      throw bioExceptNullPointer(__FILE__,__LINE__,"thread memory") ;
    }
    theInput[thread]->data = &theData ;
    if (panel) {
      theInput[thread]->dataMap = &theDataMap ;
    }
    theInput[thread]->missingData = missingData ;
    theInput[thread]->startData = thread * sizeOfEachBlock ;
    if (panel) {
      theInput[thread]->endData = (thread == nbrOfThreads-1) ? theDataMap.size() : (thread+1) * sizeOfEachBlock ;
    }
    else {
      theInput[thread]->endData = (thread == nbrOfThreads-1) ? theData.size() : (thread+1) * sizeOfEachBlock ;
    }
    theInput[thread]->literalIds = &literalIds ;
    bioExpression* theLoglike = theInput[thread]->theLoglike.getExpression() ;
    theLoglike->setData(theInput[thread]->data) ;
    if (panel) {
      theLoglike->setDataMap(theInput[thread]->dataMap) ;
    }
    theLoglike->setMissingData(theInput[thread]->missingData) ;
    if (theInput[thread]->theWeight.isDefined()) {
      theInput[thread]->theWeight.setData(theInput[thread]->data) ;
      if (panel) {
	theInput[thread]->theWeight.setDataMap(theInput[thread]->dataMap) ;
      }
      theInput[thread]->theWeight.setMissingData(theInput[thread]->missingData) ;
    }
  }

}

void biogeme::prepareDataSimul() {

  theThreadMemorySimul.setData(&theData) ;
  if (panel) {
    theThreadMemorySimul.setDataMap(&theDataMap) ;
  }
  theThreadMemorySimul.setMissingData(missingData) ;
  if (!theDraws.empty()) {
    theThreadMemorySimul.setDraws(&theDraws) ;
  }

  // Prepare the input for the threads

  // Calculate the size of the block  of data to be sent to each thread
  bioUInt sizeOfEachBlock ;
  bioUInt numberOfBlocks ;
  if (panel) {
    sizeOfEachBlock = ceil(bioReal(theDataMap.size())/bioReal(nbrOfThreads)) ;
    numberOfBlocks = ceil(bioReal(theDataMap.size()) / bioReal(sizeOfEachBlock)) ;
  }
  else {
    sizeOfEachBlock = ceil(bioReal(theData.size())/bioReal(nbrOfThreads)) ;
    numberOfBlocks = ceil(bioReal(theData.size()) / bioReal(sizeOfEachBlock)) ;
  }
  // For small data sets, there may be more threads than number of blocks.
  if (numberOfBlocks < nbrOfThreads) {
    nbrOfThreads = numberOfBlocks ;
  }

  theSimulInput.resize(nbrOfThreads, NULL) ;

  for (bioUInt thread = 0 ; thread < nbrOfThreads ; ++thread) {
    bioThreadArgSimul* ptr = theThreadMemorySimul.getInput(thread) ;
    if (ptr == NULL) {
      throw bioExceptNullPointer(__FILE__,__LINE__,"thread memory") ;
    }
    theSimulInput[thread] = ptr ;
    theSimulInput[thread]->panel = panel ;
    theSimulInput[thread]->data = &theData ;
    if (panel) {
      theSimulInput[thread]->dataMap = &theDataMap ;
    }
    theSimulInput[thread]->missingData = missingData ;
    theSimulInput[thread]->startData = thread * sizeOfEachBlock ;
    if (panel) {
      theSimulInput[thread]->endData = (thread == nbrOfThreads-1) ? theDataMap.size() : (thread+1) * sizeOfEachBlock ;
    }
    else {
      theSimulInput[thread]->endData = (thread == nbrOfThreads-1) ? theData.size() : (thread+1) * sizeOfEachBlock ;
    }
    bioSeveralExpressions* theFormulas = theSimulInput[thread]->theFormulas.getExpressions() ;
    if (theFormulas == NULL) {
      throw bioExceptNullPointer(__FILE__,__LINE__,"bioSeveralExpressions") ;
    }
    theFormulas->setData(theSimulInput[thread]->data) ;
    if (panel) {
      theFormulas->setDataMap(theSimulInput[thread]->dataMap) ;
    }
    theFormulas->setMissingData(theSimulInput[thread]->missingData) ;
  }
}

bioUInt biogeme::getDimension() const {
  return literalIds.size() ;
}

// // Used only by CFSQP
// bioReal biogeme::repeatedCalcLikeAndDerivatives(std::vector<bioReal>& beta,
// 						std::vector<bioReal>& g,
// 						std::vector< std::vector<bioReal> >& h,
// 						std::vector< std::vector<bioReal> >& bh,
// 						bioBoolean hessian,
// 						bioBoolean bhhh) {

//   if (!fixedBetasDefined) {
//     std::stringstream str ;
//     str << "The function setFixedBetas must be called first" ;
//     throw bioExceptions(__FILE__,__LINE__,str.str()) ;
//   }
//   return calculateLikeAndDerivatives(beta,
// 				     theFixedBetas,
// 				     literalIds,
// 				     g,
// 				     h,
// 				     bh,
// 				     hessian,
// 				     bhhh) ;
  
// }

void biogeme::setFixedBetas(std::vector<bioReal>& fb,
			    std::vector<bioUInt>& betaIds) {
  theFixedBetas = fb ;
  literalIds = betaIds ;
  fixedBetasDefined = true ;
}

// // Used only by CFSQP
// bioReal biogeme::repeatedCalculateLikelihood(std::vector<bioReal>& beta) {
//   return calculateLikelihood(beta,theFixedBetas) ;
// }


// bioString biogeme::cfsqp(std::vector<bioReal>& beta,
// 			 std::vector<bioReal>& fixedBeta,
// 			 std::vector<bioUInt>& betaIds,
// 			 bioUInt& nit,
// 			 bioUInt& nf,
// 			 bioUInt& mode,
// 			 bioUInt& iprint,
// 			 bioUInt& miter,
// 			 bioReal eps) {

//   setFixedBetas(fixedBeta,betaIds) ;
//   bioCfsqp theCfsqp(this) ;
//   theCfsqp.setParameters(mode,iprint,miter,eps,eps,0) ;
//   theCfsqp.defineStartingPoint(beta) ;
//   bioString d = theCfsqp.run() ;
//   std::vector<bioReal> xstar = theCfsqp.getSolution() ;
//   std::copy(xstar.begin(),xstar.end(),beta.begin()) ;
//   nit = theCfsqp.nbrIter() ;
//   nf = nbrFctEvaluations ;
//   return d ;
// }

void biogeme::setBounds(std::vector<bioReal> lb, std::vector<bioReal> ub) {
  lowerBounds = lb ;
  upperBounds = ub ;
}


std::vector<bioReal> biogeme::getLowerBounds() {
  return lowerBounds ;

}

std::vector<bioReal> biogeme::getUpperBounds() {
  return upperBounds ;
}

void biogeme::resetFunctionEvaluations() {
  nbrFctEvaluations = 0 ;
}

bioReal biogeme::getNumericalEpsilon() {
  return bioEpsilon ;
}

bioReal biogeme::getNumericalMin() {
  return bioMinReal ;
}

bioReal biogeme::getNumericalMax() {
  return bioMaxReal ;
}
