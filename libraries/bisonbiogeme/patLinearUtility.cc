//-*-c++-*------------------------------------------------------------
//
// File name : patLinearUtility.cc
// Author :    Michel Bierlaire
// Date :      Wed Jan 10 14:06:04 2001
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include "patModelSpec.h"
#include "patLinearUtility.h"
#include "patOutputFiles.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patValueVariables.h"

patLinearUtility::patLinearUtility(patBoolean m) : 
  oldDetermUtilities(patModelSpec::the()->getNbrAlternatives()),
  previousObservation(patModelSpec::the()->getNbrAlternatives(),patBadId),
  previousDraw(patModelSpec::the()->getNbrAlternatives(),patBadId),
  previousDerivObservation(patModelSpec::the()->getNbrAlternatives(),patBadId),
  utilities(patModelSpec::the()->getNbrAlternatives(),NULL),
  startRandom(patModelSpec::the()->getNbrAlternatives())
{

}

patLinearUtility::~patLinearUtility() {

}

patReal patLinearUtility::computeLinearFunction(unsigned long observationId,
						unsigned long drawNumber,
						unsigned long altId,
						patVariables* beta,
						vector<patObservationData::patAttributes>* x,
						patError*& err) {

  
  
  completeCalculation = 
    (previousObservation[altId] != observationId || 1+previousDraw[altId] != drawNumber) ;
  
  //  DEBUG_MESSAGE("-------------------------") ;
  if (!completeCalculation) {
    outputVariable = oldDetermUtilities[altId] ;
  }
  else {
    outputVariable = 0.0 ;
  }

  if (utilities[altId] == NULL) {
    
    userId = patModelSpec::the()->getAltId(altId,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    utilities[altId]= patModelSpec::the()->getUtil(userId,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    initPointers(altId) ;
  }
  theUtility = utilities[altId] ;


  start = 
    (completeCalculation)
    ? theUtility->begin() 
    : startRandom[altId] ;

  determ = 0.0 ;

  for (i = start ;
       i != theUtility->end() ;
       ++i) {

    drawValue = -1 ;
    patBoolean computeTerm = patTRUE ;
    if (i->massAtZero) {
      	drawValue = patValueVariables::the()->getRandomDrawValue(i->rndIndex,err) ;
	if (drawValue == -1.0) {
	  computeTerm = patFALSE ;
	}
    }

    if (computeTerm) {

      if ((*x)[i->xIndex].value == patParameters::the()->getgevMissingValue()) {
	stringstream str ;
	str << "Missing value used in the computation of utility function for " << i->beta ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return patReal() ;
      }
      
      if (i->random) {
	if (drawValue == -1) {
	  
	  drawValue = patValueVariables::the()->getRandomDrawValue(i->rndIndex,err) ;
	}
	theAttr = (*x)[i->xIndex].value * drawValue ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return patReal() ;
	}
      }
      else {
	theAttr = (*x)[i->xIndex].value ;
	if (completeCalculation) {
	  determ += theAttr * (*beta)[i->betaIndex]  ;
	}
      }
      
      outputVariable +=  theAttr * (*beta)[i->betaIndex]  ;
    }
  }

  if (completeCalculation) {
    oldDetermUtilities[altId] = determ ;
    previousObservation[altId] = observationId ;
    previousDraw[altId] = drawNumber ;
  }

  
  return outputVariable ;
}

patReal patLinearUtility::computeFunction(unsigned long observationId,
					  unsigned long drawNumber,
					  unsigned long altId,
					  patVariables* beta,
					  vector<patObservationData::patAttributes>* x,
					  patError*& err) {
  
  patReal theLinearUtility = computeLinearFunction(observationId,
						   drawNumber,
						   altId,
						   beta,
						   x,
						   err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  
  patBetaLikeParameter* betaMGEV = patModelSpec::the()->getMGEVParameter() ;
  if (betaMGEV != NULL) {
    patReal beta = betaMGEV->estimated;
    if (beta == 0.0) {
      return theLinearUtility ;
    }
    patReal term = log(1.0 + beta * theLinearUtility) / beta ;
    if (isfinite(term) == 0) {
      stringstream str ;
      str << "Error with the MGEV model. Cannot compute log(1+" 
	  << beta 
	  << "*" 
	  << theLinearUtility
	  << ")/" 
	  << beta ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patReal(0) ;
    }
    else {
      return term ;
    }
  }
  else {
    return theLinearUtility ;
  }
}
  


patVariables*
patLinearUtility::computeBetaDerivative(unsigned long observationId,
					unsigned long drawNumber,
					unsigned long altId,
					patVariables* beta,
					vector<patObservationData::patAttributes>* x,
					patVariables* betaDerivatives,
					patError*& err)  {

  if (betaDerivatives == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return betaDerivatives  ;
  }

  if (beta == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return betaDerivatives  ;
  }
  if (x == NULL) {
    err = new patErrNullPointer("vector<patObservationData::patAttributes>") ;
    WARNING(err->describe()) ;
    return betaDerivatives ;
  }

  if (betaDerivatives->size() != beta->size()) {
    stringstream str ;
    str << "Size of betaDerivatives should be " << beta->size() 
	<< " and not " << betaDerivatives->size() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return betaDerivatives  ;
  }

  if (utilities[altId] == NULL) {
    userId = patModelSpec::the()->getAltId(altId,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return betaDerivatives ;
    }
    utilities[altId] = patModelSpec::the()->getUtil(userId,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return betaDerivatives ;
    }
  }
  theUtility = utilities[altId] ;

  fill(betaDerivatives->begin(),betaDerivatives->end(),0.0) ;
  for (i = theUtility->begin() ;
       i != theUtility->end() ;
       ++i) {
    if (i->random) {
      theAttr = (*x)[i->xIndex].value * 
	patValueVariables::the()->getRandomDrawValue(i->rndIndex,err) ;
    }
    else {
      theAttr = (*x)[i->xIndex].value ;
    }

    (*betaDerivatives)[i->betaIndex] += theAttr ;
  }

  patBetaLikeParameter* betaMGEV = patModelSpec::the()->getMGEVParameter() ;
  if (betaMGEV != NULL) {
    (*betaDerivatives)[betaMGEV->id] = 0.0 ;
    patReal theValue = betaMGEV->estimated;
    patReal theLinearUtility = computeLinearFunction(observationId,
						     drawNumber,
						     altId,
						     beta,
						     x,
						     err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return betaDerivatives ;
    }
    
    
    patReal t1 = 1.0 + theValue * theLinearUtility ;
    patReal term = ((theValue * theLinearUtility / t1) - log(t1)) / (theValue * theValue) ;
    if (isfinite(term) == 0) {
      stringstream str ;
      str << "Error with the MGEV model. Cannot compute (("
	  << theValue
	  << " * "
	  << theLinearUtility
	  << " / "
	  << t1
	  << ") - log(" 
	  << t1 
	  << ")) / (" 
	  << theValue 
	  << " * " 
	  << theValue
	  <<")" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return betaDerivatives ;
    }
    
    
    (*betaDerivatives)[betaMGEV->id] = term ;
  }
  return betaDerivatives ;
}

patVariables patLinearUtility::computeCharDerivative(unsigned long observationId,
						     unsigned long drawNumber,
						     unsigned long altId,
						     patVariables* beta,
						     vector<patObservationData::patAttributes>* x,
						     patError*& err)  {
  if (beta == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return patVariables()  ;
  }
  if (x == NULL) {
    err = new patErrNullPointer("vector<patObservationData::patAttributes>") ;
    WARNING(err->describe()) ;
    return patVariables() ;
  }

  err = new patErrMiscError("Not yet implemented...") ;
  WARNING(err->describe()) ;
  return patVariables() ;
  
}
  

void patLinearUtility::initPointers(unsigned long altId) {
  // Compute the startRandom pointers

  if (utilities[altId] == NULL) {
    WARNING("Utility " << altId << " is not defined") ;
  }
  else{
    
    patBoolean found(patFALSE) ;
    for (term = utilities[altId]->begin() ;
	 term != utilities[altId]->end() && !found ;
	 ++term) {
      if (term->random) {
	found = patTRUE ;
	startRandom[altId] = term ;
      }
    }
    if (!found) {
      startRandom[altId] = utilities[altId]->end() ;
    }
  }
}

void patLinearUtility::printFileForDenis(unsigned long nbeta) {

  patError* err(NULL) ;

  DEBUG_MESSAGE("Nbr of utilities = " << utilities.size()) ;
  
  vector< vector < int > > table(utilities.size(),vector<int>(nbeta,0)) ;
  for (unsigned long i = 0 ; i < utilities.size() ;  ++i) {

    DEBUG_MESSAGE("--> DENIS --> Alt " << i) ;
    unsigned long userId = patModelSpec::the()->getAltId(i,err) ;
    utilities[i]= patModelSpec::the()->getUtil(userId,err) ;
    for (patUtilFunction::iterator term = utilities[i]->begin() ;
	 term != utilities[i]->end() ;
	 ++term) {
      DEBUG_MESSAGE(term->beta << "[" << term->betaIndex << "]*" << term->x << "[" << term->xIndex+1 << "]") ;

      table[i][term->betaIndex] = term->xIndex+1;
    }
  }

  patString fileName("util.lis") ;
  ofstream denis(fileName.c_str()) ;

  for (vector< vector < int > >::iterator i = table.begin() ;
       i != table.end() ;
       ++i) {
    for (vector<int>::iterator j = i->begin() ;
	 j != i->end() ;
	 ++j) {
      denis << *j << " " ;
    }
    denis << endl ;
  }
  denis.close() ;
  patOutputFiles::the()->addDebugFile(fileName,"List of utility functions");
  
}


void patLinearUtility::generateCppCode(ostream& cppFile,
				       unsigned long altId,
				       patError*& err) {

  if (utilities[altId] == NULL) {
    
    userId = patModelSpec::the()->getAltId(altId,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    utilities[altId]= patModelSpec::the()->getUtil(userId,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    initPointers(altId) ;
  }
  theUtility = utilities[altId] ;
  
  patBoolean firstTerm(patTRUE) ;
  for (list<patUtilTerm>::iterator i = theUtility->begin() ;
       i != theUtility->end() ;
       ++i) {
    patBoolean found ;
    patBetaLikeParameter beta = patModelSpec::the()->getBeta(i->beta,&found) ;
    if (!found) {
      stringstream str ;
      str << "Unknown parameter " << i->beta ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }

    if (i->random) {
      if (beta.isFixed) {
	if (beta.defaultValue != 0.0) {
	  if (!firstTerm) {
	    cppFile << " + " ;
	  }
	  cppFile << beta.defaultValue ;
	  cppFile << " * observation->attributes[" << i->xIndex << "].value * observation->draws[drawNumber-1][" << i->rndIndex << "]" ;
	  firstTerm = patFALSE ;
	}
      }
      else {
	if (!firstTerm) {
	  cppFile << " + " ;
	}
	cppFile << "(*x)[" << beta.index << "]" ;
	cppFile << " * observation->attributes[" << i->xIndex << "].value * observation->draws[drawNumber-1][" << i->rndIndex << "]" ;
	firstTerm = patFALSE ;
      }
    }
    else {
      if (beta.isFixed) {
	if (!firstTerm) {
	  cppFile << " + " ;
	}
	if (beta.defaultValue != 0.0) {
	  cppFile << beta.defaultValue ;
	  cppFile << " * observation->attributes[" << i->xIndex << "].value" ;
	}
	else {
	  cppFile << "0.0" ;
	}
	firstTerm = patFALSE ;
      }
      else {
	if (!firstTerm) {
	  cppFile << " + " ;
	}
	cppFile << "(*x)[" << beta.index << "]" ;
	cppFile << " * observation->attributes[" << i->xIndex << "].value" ;
	firstTerm = patFALSE ;
      }
    }
  }
  if (firstTerm) {
    cppFile << "0.0" ;
  }
  
}

void patLinearUtility::generateCppDerivativeCode(ostream& cppFile,
						 unsigned long altId,
						 unsigned long betaId,
						 patError*& err) {

  if (utilities[altId] == NULL) {
    
    userId = patModelSpec::the()->getAltId(altId,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    utilities[altId]= patModelSpec::the()->getUtil(userId,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    initPointers(altId) ;
  }
  theUtility = utilities[altId] ;
  
  patBoolean something(patFALSE) ;
  for (list<patUtilTerm>::iterator i = theUtility->begin() ;
       i != theUtility->end() ;
       ++i) {
    patBoolean found ;
    patBetaLikeParameter beta = patModelSpec::the()->getBeta(i->beta,&found) ;
    if (!found) {
      stringstream str ;
      str << "Unknown parameter " << i->beta ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }

    if (i->random) {
      if (!beta.isFixed) {
	if (beta.index == betaId) {
	  cppFile << "observation->attributes[" << i->xIndex << "].value * observation->draws[drawNumber-1][" << i->rndIndex << "]" ;
	  something = patTRUE ;
	}
      }
    }
    else {
      if (!beta.isFixed) {
	if (beta.index == betaId) {
	  cppFile << " observation->attributes[" << i->xIndex << "].value"  ;
	  something = patTRUE ;
	}
      }
    }
  }
  if (!something) {
    cppFile << "0.0" ;
  }
  
}
