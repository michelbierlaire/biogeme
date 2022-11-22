//-*-c++-*------------------------------------------------------------
//
// File name : evaluateExpressions.cc
// @date   Thu Oct 14 13:49:38 2021
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#include <sstream>
#include "evaluateExpressions.h"
#include "bioExpression.h"
#include "bioFormula.h"
#include "bioDebug.h"
#include "bioExceptions.h"

// Dealing with exceptions across threads
static std::exception_ptr theExceptionPtr = nullptr ;

void *computeFunctionForThreadExpression( void *ptr );

evaluateOneExpression::evaluateOneExpression():
  with_data(false),
  panel(false),
  nbrOfThreads(4) {
}

void evaluateOneExpression::setFreeBetas(std::vector<bioReal> freeBetas) {
  theFreeBetas = freeBetas ;
  literalIds.resize(theFreeBetas.size()) ;
  for (bioUInt i = 0 ;
       i < literalIds.size() ;
       ++i) {
    literalIds[i] = i ;
  }
}

bioUInt evaluateOneExpression::getDimension() const {
  return literalIds.size() ;
}

bioUInt evaluateOneExpression::getSampleSize() const {
  if (!with_data || aggregation) {
    return 1 ;
  }
  if (panel) {
    return theDataMap.size() ;
  }
  return theData.size() ;
}

void evaluateOneExpression::setFixedBetas(std::vector<bioReal> fixedBetas) {
  theFixedBetas = fixedBetas ;
}

void evaluateOneExpression::setExpression(std::vector<bioString> f) {
  expression = f ;
}

void evaluateOneExpression::calculate(bioBoolean gradient,
				      bioBoolean hessian,
				      bioBoolean bhhh,
				      bioBoolean aggr) {

  results.clear() ;
  with_g = gradient ;
  with_h = hessian ;
  with_bhhh = bhhh ;
  results.set_with_g(with_g) ;
  results.set_with_h(with_h) ;
  results.set_with_bhhh(with_bhhh) ;
  aggregation = aggr ;
  if (with_data) {
    prepareData() ;
    applyTheFormula() ;
  }
  else {
    // When there is no database, no need to use parallel computing
    bioFormula theFormula ;
    theFormula.setExpression(expression) ;
    theFormula.setParameters(&theFreeBetas) ;
    theFormula.setFixedParameters(&theFixedBetas) ;
    theFormula.setMissingData(missingData) ;
    bioDerivatives d = *theFormula.getExpression()->getValueAndDerivatives(literalIds,
									  gradient,
									  hessian) ;
    results.aggregate(d) ;
  }
  gradientCalculated = gradient ;
  hessianCalculated = hessian ;
  bhhhCalculated = bhhh ;
}

void evaluateOneExpression::getResults(bioReal* f,
				       bioReal* g,
				       bioReal* h,
				       bioReal* bhhh) {
  bioUInt n = getDimension() ;
  bioUInt m = getSampleSize() ;
  for (bioUInt k = 0 ; k < m ; ++k) {
    f[k] = results.theDerivatives[k].f ;
    if (gradientCalculated) {
      for (bioUInt i = 0 ; i < n ; ++i) {
	g[k * n + i] = results.theDerivatives[k].g[i] ;
	if (hessianCalculated) {
	  for (bioUInt j = i ; j < n ; ++j) {
	    h[k * n * n + i * n + j] =
	      h[k * n * n + j * n + i] =
	      results.theDerivatives[k].h[i][j] ;
	  }
	}
	if (bhhhCalculated) {
	  for (bioUInt j = i ; j < n ; ++j) {
	    bhhh[k * n * n + i * n + j] =
	      bhhh[k * n * n + j * n + i] =
	      results.theDerivatives[k].bhhh[i][j] ;
	  }
	}
      }
    }
  }
  return ;
}

void evaluateOneExpression::setData(std::vector< std::vector<bioReal> >& d) {
  theData = d ;
  with_data = true ;
}

void evaluateOneExpression::setDataMap(std::vector< std::vector<bioUInt> >& dm) {
  theDataMap = dm ;
  panel = true ;
}

void evaluateOneExpression::setMissingData(bioReal md) {
  missingData = md ;
}

void evaluateOneExpression::setDraws(std::vector< std::vector< std::vector<bioReal> > >& draws) {
  theDraws = draws ;
}

void evaluateOneExpression::prepareData() {
  theThreadMemory.clear() ;
  if (aggregation) {
    theThreadMemory.resize(nbrOfThreads, getDimension(), 1) ;
  }
  else {
    if (panel) {
      theThreadMemory.resize(nbrOfThreads, getDimension(), theDataMap.size()) ;
    }
    else {
      theThreadMemory.resize(nbrOfThreads, getDimension(), theData.size()) ;
    }
  }
  theThreadMemory.setFormula(expression) ;
  theThreadMemory.setParameters(&theFreeBetas) ;
  theThreadMemory.setFixedParameters(&theFixedBetas) ;
  theThreadMemory.setMissingData(missingData) ;
  if (with_data) {
    theThreadMemory.setData(&theData) ;
    if (panel) {
      theThreadMemory.setDataMap(&theDataMap) ;
    }
  }

  theThreadMemory.setMissingData(missingData) ;

  if (!theDraws.empty()) {
    theThreadMemory.setDraws(&theDraws) ;
  }

  // if (theThreadMemory.dimension() < literalIds.size()) {
  //   std::stringstream str ;
  //   str << " Memory should be reserved for dimension "
  // 	<< literalIds.size()
  // 	<< " and not "
  // 	<< theThreadMemory.dimension() ;
  //   throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  // }
  
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
      theInput[thread]->endData =
	(thread == nbrOfThreads-1)
	? theDataMap.size()
	: (thread+1) * sizeOfEachBlock ;
    }
    else {
      theInput[thread]->endData =
	(thread == nbrOfThreads-1)
	? theData.size()
	: (thread+1) * sizeOfEachBlock ;
    }
    theInput[thread]->literalIds = &literalIds ;
    bioExpression* theExpression = theInput[thread]->theFormula.getExpression() ;
    theExpression->setData(theInput[thread]->data) ;
    if (panel) {
      theExpression->setDataMap(theInput[thread]->dataMap) ;
    }
    theExpression->setMissingData(theInput[thread]->missingData) ;
  }
}

void evaluateOneExpression::setNumberOfThreads(bioUInt n) {
  nbrOfThreads = n ;
}

void evaluateOneExpression::applyTheFormula() {
  std::vector<pthread_t> theThreads(nbrOfThreads) ;
  for (bioUInt thread = 0 ; thread < nbrOfThreads ; ++thread) {
    if (theInput[thread] == NULL) {
      throw bioExceptNullPointer(__FILE__,__LINE__,"thread") ;
    }

    theInput[thread]->calcGradient = with_g ;
    theInput[thread]->calcHessian = with_h ;
    theInput[thread]->calcBhhh = with_bhhh ;

    bioUInt diagnostic = pthread_create(&(theThreads[thread]),
					NULL,
					computeFunctionForThreadExpression,
					(void*) theInput[thread]) ;
    
    if (diagnostic != 0) {
      std::stringstream str ;
      str << "Error " << diagnostic << " in creating thread " << thread << "/" << nbrOfThreads ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
  }

  results.clear() ;
  for (bioUInt thread = 0 ; thread < nbrOfThreads ; ++thread) {
    pthread_join( theThreads[thread], NULL);
    if (theExceptionPtr != nullptr) {
      std::rethrow_exception(theExceptionPtr);
    }
    if (aggregation) {
      results.aggregate(theInput[thread]->theDerivatives) ;
    }
    else {
      results.disaggregate(theInput[thread]->theDerivatives) ;
    }
  }
  results.dealWithNumericalIssues() ;
  return ;
}



void *computeFunctionForThreadExpression(void* fctPtr) {
  try {
    bioThreadArgOneExpression *input = (bioThreadArgOneExpression *) fctPtr;
    input->theDerivatives.clear() ;
    bioExpression* myExpression = input->theFormula.getExpression() ;
    if (input->panel) {
      // Panel data
      bioUInt individual ;
      myExpression->setIndividualIndex(&individual) ;
      for (individual = input->startData ;
	   individual < input->endData ;
	   ++individual) {
      
	const bioDerivatives* fgh = myExpression->getValueAndDerivatives(*input->literalIds,
									 input->calcGradient,
									 input->calcHessian) ;

	if (input->aggregation) {
	  input->theDerivatives.aggregate(*fgh) ;
	}
	else {
	  input->theDerivatives.disaggregate(*fgh) ;
	}
      }
    }
    else {
      // No panel data
      bioUInt row ;
      if (myExpression == NULL) {
	throw bioExceptNullPointer(__FILE__,__LINE__,"thread memory") ;
      }
      myExpression->setIndividualIndex(&row) ;
      myExpression->setRowIndex(&row) ;
      for (row = input->startData ;
	   row < input->endData ;
	   ++row) {
	
	try {

	  const bioDerivatives* fgh = myExpression->getValueAndDerivatives(*input->literalIds,
									   input->calcGradient,
									   input->calcHessian) ;
	  if (input->aggregation) {
	    input->theDerivatives.aggregate(*fgh) ;
	  }
	  else {
	    input->theDerivatives.disaggregate(*fgh) ;
	  }
	}
	catch(bioExceptions& e) {
	  std::stringstream str ;
	  str << "Error for data entry " << row << " : " << e.what() ;
	  throw bioExceptions(__FILE__,__LINE__,str.str()) ;
	}
      }
    }
    input->theDerivatives.set_with_g(input->calcGradient) ;
    input->theDerivatives.set_with_h(input->calcHessian) ;
    input->theDerivatives.set_with_bhhh(input->calcBhhh) ;
    input->theFormula.setRowIndex(NULL) ;
    input->theFormula.setIndividualIndex(NULL) ;
  }
  catch(...)  {
    theExceptionPtr = std::current_exception() ;
  }
  return NULL ;
}
