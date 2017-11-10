//-*-c++-*------------------------------------------------------------
//
// File name : bioArithPrint.cc
// Author :    Michel Bierlaire
// Date :      Tue May 11 07:10:16 2010
//
//--------------------------------------------------------------------


#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patError.h"
#include "patFileNames.h"
#include "bioVersion.h"
#include "bioRowIterator.h"
#include "bioArithPrint.h"
#include "bioArithBinaryPlus.h"
#include "bioLiteralRepository.h"
#include "bioIteratorInfoRepository.h"
#include "bioExpressionRepository.h"
#include "bioSample.h"
#include "patHybridMatrix.h"
#include "patMultivariateNormal.h"
#include "bioParameters.h"
#include "bioRandomDraws.h"
#include "patNormalWichura.h"
#include "bioSensitivityAnalysis.h"

/*!
*/
bioArithPrint::bioArithPrint(bioExpressionRepository* rep,
			     patULong par,
			     map<patString,patULong> theTerms,
			     patString it,patError*& err) 
  : bioExpression(rep, par),
    theIteratorName(it),
    theExpressionsId(theTerms), 
    theRandomDraws(NULL) {
  
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  theIteratorType = bioIteratorInfoRepository::the()->getType(theIteratorName,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  for (map<patString,patULong>::iterator i = theExpressionsId.begin() ;
       i != theExpressionsId.end() ;
       ++i) {
    theExpressions[i->first] = theRepository->getExpression(i->second) ;
    relatedExpressions.push_back(theExpressions[i->first]) ;
  }
  bioIteratorSpan theSpan(it,0) ;
  setCurrentSpan(theSpan) ;

}

bioArithPrint::~bioArithPrint() {}

patString bioArithPrint::getOperatorName() const {
  return patString("Print") ;
}

bioExpression* bioArithPrint::getDerivative(patULong aLiteralId, patError*& err) const {

  err = new patErrMiscError("No derivative for the print expression") ;
  WARNING(err->describe()) ;
  return NULL;
}


bioArithPrint* bioArithPrint::getDeepCopy(bioExpressionRepository* rep,
					  patError*& err) const {

  map<patString,patULong> newDict ;
  for (map<patString,bioExpression*>::const_iterator i = theExpressions.begin() ;
       i != theExpressions.end() ;
       ++i) {
    bioExpression* clone = i->second->getDeepCopy(rep,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    newDict[i->first] = clone->getId() ;
  }

  bioArithPrint* theNode = 
    new bioArithPrint(rep,patBadId,newDict,theIteratorName,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  
  return theNode ;
}

bioArithPrint* bioArithPrint::getShallowCopy(bioExpressionRepository* rep,
					  patError*& err) const {

  bioArithPrint* theNode = 
    new bioArithPrint(rep,patBadId,theExpressionsId,theIteratorName,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  
  return theNode ;
}



patBoolean bioArithPrint::isSum() const {
  return patFALSE ;
}

patBoolean bioArithPrint::isProd() const {
  return patFALSE ;
}


patString bioArithPrint::getExpression(patError*& err) const {
  stringstream ss ;
  ss << bioIteratorInfoRepository::the()->getInfo(theIteratorName,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  ss << "[" ;
  for (map<patString,bioExpression*>::const_iterator i = theExpressions.begin() ;
       i != theExpressions.end() ;
       ++i) {
    if (i != theExpressions.begin()) {
      ss << ", " ;
    }
    ss << i->first << "=" << *(i->second) ;
  }
  ss << "]" ;

  return patString(ss.str()) ;
  
}

patULong bioArithPrint::getNumberOfOperations() const {
  patULong count = 0 ;
  for (map<patString,bioExpression*>::const_iterator i = theExpressions.begin() ;
       i != theExpressions.end() ;
       ++i) {
    count += i->second->getNumberOfOperations() ;
  }
  return(count) ;

}


patReal bioArithPrint::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  {
  err = new patErrMiscError("Irrelevant with this object") ;
  WARNING(err->describe()) ;
  return patReal() ;
}


bioFunctionAndDerivatives* bioArithPrint::getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian, patBoolean debugDeriv, patError*& err) {
  stringstream str ;
  str << "Not implemented for expression " << getOperatorName() ;
  err = new patErrMiscError(str.str()) ;
  WARNING(err->describe()) ;
  return NULL ;
  
}




patBoolean bioArithPrint::dependsOf(patULong aLiteralId) const {
  for (map<patString,bioExpression*>::const_iterator i = theExpressions.begin() ;
       i != theExpressions.end() ;
       ++i) {
    if (i->second->dependsOf(aLiteralId)) {
      return patTRUE ;
    }
  }
  return patFALSE ;

}


patBoolean bioArithPrint::containsAnIterator() const {
  return patTRUE ;
}

patBoolean bioArithPrint::containsAnIteratorOnRows() const {
  return theIteratorType == ROW ;
}


patBoolean bioArithPrint::containsAnIntegral() const {
  for (map<patString,bioExpression*>::const_iterator i = theExpressions.begin() ;
       i != theExpressions.end() ;
       ++i) {
    if (i->second->containsAnIntegral()) {
      return patTRUE ;
    }
  }
  return patFALSE ;
}

patBoolean bioArithPrint::containsASequence() const {
  for (map<patString,bioExpression*>::const_iterator i = theExpressions.begin() ;
       i != theExpressions.end() ;
       ++i) {
    if (i->second->containsASequence()) {
      return patTRUE ;
    }
  }
  return patFALSE ;

}


patString bioArithPrint::getIncludeStatements(patString prefix) const {
  return patString("Not implemented") ;
}

void bioArithPrint::simplifyZeros(patError*& err) {
  return ;
}

void bioArithPrint::collectExpressionIds(set<patULong>* s) const {
  s->insert(getId()) ;
  for (map<patString,bioExpression*>::const_iterator i = theExpressions.begin() ;
       i != theExpressions.end() ;
       ++i) {
    i->second->collectExpressionIds(s) ;
  }
}

patString bioArithPrint::getExpressionString() const {
  stringstream str ;
  str << "$Print{" ;
  for (map<patString,bioExpression*>::const_iterator i = theExpressions.begin() ;
       i != theExpressions.end() ;
       ++i) {
    if (i != theExpressions.begin()) {
      str << "," ;
    }
    str << i->second->getExpressionString() ;
  }
  str << "}" ;
  return patString(str.str()) ;
}

patBoolean bioArithPrint::isSimulator() const {
  return patTRUE ;
}

void bioArithPrint::simulate(patHybridMatrix* varCovar, 
			     vector<patString> betaNames, 
			     bioExpression* weight,
			     patError*& err) {

  if (theIteratorType != ROW) {
    err = new patErrMiscError("Simulation only on row iterators") ;
    WARNING(err->describe()) ;
    return  ;
  }

  // Initialize the database

  bioRowIterator* theIter = theSample->createRowIterator(theCurrentSpan,
							 theThreadSpan,
							 patTRUE,
							 err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  patULong dim = theIter->nbrOfItems() ;
  patULong nDraws(0) ;
  if (varCovar != NULL) {
    nDraws = bioParameters::the()->getValueInt("NbrOfDrawsForSensitivityAnalysis",err)  ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }

  // Include the weight if present

  patString weightHeader = bioParameters::the()->getValueString("HeaderForWeightInSimulatedResults",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return  ;
  }
  if (weight != NULL) {
    simulatedValues[weightHeader].resize(dim,0) ;
  }

  // Possibly include the data
  patBoolean includeData = 
    (bioParameters::the()->getValueInt("includeDataWhenSimulate",err) != 0);
  if (err != NULL) {
    WARNING(err->describe()) ;
    return  ;
  }
  vector<patString> theHeaders = theSample->getHeaders() ;
  if (includeData) {
    for (patULong i = 0 ; i < theHeaders.size() ; ++i) {
      stringstream str ; 
      str << "__DATA_" << theHeaders[i] ;
      simulatedValues[str.str()].resize(dim,0) ;
    }
  }

  // Values requested by the user
  for (map<patString,bioExpression*>::iterator i = theExpressions.begin() ;
       i != theExpressions.end() ;
       ++i) {
    simulatedValues[i->first].resize(dim,nDraws) ; 
  }

  // Prepare the sensitivity analysis

  vector<patVariables> simulatedBetas ;

  patVariables mean ;
  if (varCovar != NULL) {
    DEBUG_MESSAGE("Prepare sensitivy analysis for " << betaNames.size() << " parameters") ;
    simulatedParam.resize(betaNames.size()) ;
    for (patULong b = 0 ; b < betaNames.size()  ; ++b) {
      patString name = betaNames[b] ;
      patReal literalId = bioLiteralRepository::the()->getLiteralId(name,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      if (literalId != patBadId) {
	simulatedParam[b] = patTRUE ;
	patReal v = bioLiteralRepository::the()->getBetaValue(name,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
	mean.push_back(v) ;
      }
      else {
	simulatedParam[b] = patFALSE ;
	mean.push_back(0.0) ;
	stringstream str ;
	str << "Parameter " << name << " is not used in the formula to simulate. It is unnecesary to compute the sensitivity with respect to it." ;
	WARNING(str.str()) ;
      }
    }
    
    patNormalWichura* theNormal = new patNormalWichura() ;
    theNormal->setUniform(bioRandomDraws::the()->getUniformGenerator()) ;
    
    ofstream debugFile("beta.draws") ;
    for (vector<patString>::iterator b = betaNames.begin() ;
	 b != betaNames.end() ;
	 ++b) {
      debugFile << *b << '\t' ;
    }
    debugFile << endl ;
    debugFile << mean << endl ;
    debugFile << "+++++++++++++++" << endl ;

    theRandomDraws = new patMultivariateNormal(&mean,
					       varCovar,
					       theNormal,
					       err) ;
      
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    for (patULong r = 0 ; r < nDraws ; ++r) {
      patVariables newBeta = theRandomDraws->getNextDraw(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      debugFile << newBeta << endl ;
      simulatedBetas.push_back(newBeta) ;
    }
    debugFile.close() ;
  }
  
  patULong currentRow(0) ;
  for (theIter->first() ;
       !theIter->isDone() ;
       theIter->next()) {
    // First, the weight, if asked
      
    if (weight != NULL) {
      weight->setVariables(theIter->currentItem()) ;
      patReal w = weight->getValue(patFALSE, patLapForceCompute, err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return  ;
      }
      
      simulatedValues[weightHeader].setNominalValue(currentRow,w) ;
      
    }
      
    // Include data

    if (includeData) {
      if (theHeaders.size() != theIter->currentItem()->size()) {
	stringstream str ;
	str << "Incompatible sizes: " << theHeaders.size() << " and " << theIter->currentItem()->size() ; 
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return ;
      }
      for (patULong i = 0 ; i < theHeaders.size() ; ++i) {
	patULong j =  theSample->getIndexFromHeader(theHeaders[i],err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return  ;
	}
	stringstream str ; 
	str << "__DATA_" << theHeaders[i] ;
	simulatedValues[str.str()].setNominalValue(currentRow,(*(theIter->currentItem()))[j]) ;
      }
    }
    
    bioSensitivityAnalysis theAnalysis ;
    for (map<patString,bioExpression*>::iterator i = theExpressions.begin() ;
	 i != theExpressions.end() ;
	 ++i) {
      if (varCovar != NULL) {
	for (patULong k = 0 ; k < betaNames.size() ; ++k) {
	  if (simulatedParam[k]) {
	    bioLiteralRepository::the()->setBetaValue(betaNames[k],mean[k],err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	  }
	}
      }

      i->second->setVariables(theIter->currentItem()) ;
      patReal res = i->second->getValue(patFALSE, patLapForceCompute, err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      simulatedValues[i->first].setNominalValue(currentRow,res) ;
      
      if (varCovar != NULL) {
	// Sensitivity analysis
	for (patULong r = 0 ; r < nDraws ; ++r) { 
	  for (patULong k = 0 ; k < simulatedBetas[r].size() ; ++k) {
	    if (simulatedParam[k]) {
	      bioLiteralRepository::the()->setBetaValue(betaNames[k],simulatedBetas[r][k],err) ;
	      if (err != NULL) {
		WARNING(err->describe()) ;
		return ;
	      }
	    }
	  }
	  patReal res = i->second->getValue(patFALSE, patLapForceCompute, err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return ;
	  }
	  stringstream str ;
	  simulatedValues[i->first].setSimulatedValue(currentRow,r,res) ;
	}
      }
    }
    ++currentRow;
  }
  
  
  DELETE_PTR(theIter) ;
  
}

map<patString,bioSimulatedValues >* bioArithPrint::getSimulationResults()  {
  return &simulatedValues ;
}

patString bioArithPrint::check(patError*& err) const {
  stringstream str ;
  for (map<patString,bioExpression*>::const_iterator  i = theExpressions.begin();
       i != theExpressions.end() ;
       ++i) {
    map<patString,patULong>::const_iterator found = theExpressionsId.find(i->first) ;
    if (found == theExpressionsId.end()) {
      str << "Incompatible data structure for bioArithPrint: expression " << i->first << " does not have an associated ID" << endl ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patString() ;
    }
    if (i->second->getId() != found->second) {
      str << "Incompatible IDS for children: " << i->second->getId() << " and " << found->second << endl ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patString() ;
    }
  }
  return patString(str.str()) ;
}

