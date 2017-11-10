//-*-c++-*------------------------------------------------------------
//
// File name : patSampleEnumeration.cc
// Author :    Michel Bierlaire
// Date :      Thu Jul 18 10:01:15 2002
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <fstream>
#include <sstream>
#include <iomanip>
#include "patModelSpec.h"
#include "patSampleEnumeration.h"
#include "patOutputFiles.h"
#include "patProbaModel.h"
#include "patSample.h" 
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "patValueVariables.h"
#include "patUtility.h"
#include "patDiscreteDistribution.h"
#include "patDiscreteParameterProba.h"
#include "patLoopTime.h"
#include "patSampleEnuGetIndices.h"

patSampleEnumeration::patSampleEnumeration(patString file, 
					   patPythonReal** arrayResult,
					   unsigned long rr,
					   unsigned long rc,
					   patSample* s,
					   patProbaModel* m,
					   patSampleEnuGetIndices* ei,
					   patUniform* rng) :
  sampleEnumerationFile(file),
  model(m),
  sample(s),
  randomNumbersGenerator(rng),
  theDiscreteParamModel(NULL),
  resultArray(arrayResult),
  resRow(rr),
  resCol(rc),
  theIndices(ei)
 {
  
  if (patModelSpec::the()->containsDiscreteParameters()) { // contains discrete paramters
    theDiscreteParamModel = new patDiscreteParameterProba(m,NULL) ;
    
  }
}

patSampleEnumeration::~patSampleEnumeration() {

}
void patSampleEnumeration::enumerate(patError*& err) {

  patBoolean saveInDataStructure = (resultArray != NULL) ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  vector<map<unsigned long,unsigned long> > 
    stats(patModelSpec::the()->nbrSimulatedChoiceInSampleEnumeration()) ;

  DEBUG_MESSAGE("Start sample enumeration...") ;
  patModelSpec::the()->copyParametersValue(err) ;
  if (err != NULL) {
    WARNING(err->describe());
    return ;
  }
  ofstream file(sampleEnumerationFile.c_str()) ;


  file << "Choice_Id\tP_choice" ;
  unsigned long nAlt = patModelSpec::the()->getNbrAlternatives() ;

  if (theIndices == NULL) {
    err = new patErrNullPointer("patSampleEnuGetIndices") ;
    WARNING(err->describe());
    return ;
  }

  unsigned long nbrOfColumns = theIndices->getNbrOfColumns() ;

  if (resultArray != NULL && nbrOfColumns > resCol) {
    GENERAL_MESSAGE("Data structure from external program contains only " << resCol << " columns when " << nbrOfColumns << " are needed") ;
    GENERAL_MESSAGE("Simulated data will not be saved in the data structure") ;
    saveInDataStructure = patFALSE ;
  }

  if (includeUtilities()) {

    for (unsigned long i = 0 ; i < nAlt ; ++i) {
      unsigned long userId = patModelSpec::the()->getAltId(i,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }
      patString name = patModelSpec::the()->getAltName(userId,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }

      file << '\t' << "V_" << name ;
    }
  }
  for (unsigned long i = 0 ; i < nAlt ; ++i) {
    unsigned long userId = patModelSpec::the()->getAltId(i,err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return ;
    }
    patString name = patModelSpec::the()->getAltName(userId,err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return ;
    }
    file << '\t' << "P_" << name  ;
    file << '\t' << "Residual_" << name  ;
  }
  file << '\t' << "Total" ;

  unsigned short nbrOfZf = patModelSpec::the()->numberOfZhengFosgerau(NULL) ;

  if (nbrOfZf > 0) {
    vector<patOneZhengFosgerau>* zf = patModelSpec::the()->getZhengFosgerauVariables() ;
    for (vector<patOneZhengFosgerau>::iterator ii = zf->begin() ;
	 ii != zf->end() ;
	 ++ii) {
      if (!ii->isProbability()) {
	patModelSpec::the()->setVariablesBetaWithinExpression(ii->expression) ;
	file << '\t' << ii->getDefaultExpressionName() ;
      }
    }
  }


  for (short i = 1 ;
       i <= patModelSpec::the()->nbrSimulatedChoiceInSampleEnumeration() ; 
       ++i) {
    file << "\tSimul" << i ;
  }

  for (map<patString, list<long> >::iterator i = patModelSpec::the()->iiaTests.begin() ;
       i != patModelSpec::the()->iiaTests.end() ;
       ++i) {
    
    for (list<long>::iterator k = i->second.begin() ;
	 k != i->second.end() ;
	 ++k) {
      stringstream str ;
      str << i->first << "_Alt_" << *k ;
      file << '\t' << str.str() ;
    }
  }

  file << endl ;
  file << setprecision(7) << setiosflags(ios::scientific|ios::showpos) ;  

  patVariables* betaParameters = 
    patModelSpec::the()->getPtrBetaParameters() ;
  patReal* mu = 
    patModelSpec::the()->getPtrMu() ;
  patVariables* modelParameters = 
    patModelSpec::the()->getPtrModelParameters() ;
  patVariables* scaleParameters = 
    patModelSpec::the()->getPtrScaleParameters() ;
    
  // Loop on all observations in the sample
  unsigned long observationCounter = 0 ;

  patBoolean success(patTRUE) ;

  file << setprecision(7) << setiosflags(ios::scientific) ;

  if (model != NULL) {
    if (patModelSpec::the()->isAggregateObserved()) {
      err = new patErrMiscError("Biosim does not handle aggregate observations") ;
      WARNING(err->describe()) ;
      return ;
    }

    patIterator<patObservationData*>* obsIterator = sample->createObsIterator() ;
    unsigned long step = patParameters::the()->getgevDataFileDisplayStep() ;

    patLoopTime loopTime(sample->getSampleSize()) ;



    for (obsIterator->first() ;
	 !obsIterator->isDone() ;
	 obsIterator->next()) {
      ++observationCounter ;
      loopTime.setIteration(observationCounter) ;
      if ((observationCounter % step) == 0) {
	GENERAL_MESSAGE(patULong(100.0 * observationCounter / sample->getSampleSize()) << "%\t" << loopTime) ;
      }
      unsigned long colNumber = 0 ;

      if (observationCounter > resRow && saveInDataStructure) {
	GENERAL_MESSAGE("There are only " << resRow << " rows in external data structures") ;
	GENERAL_MESSAGE("Results from observation " << observationCounter << " will be skipped") ;
	saveInDataStructure = patFALSE ;
      }
      
      patObservationData* observation = obsIterator->currentItem() ;
      
      patULong chosenAlt = observation->choice ;

      // Construct the utilities
      
      
      unsigned long groupIndex = sample->groupIndex(observation->group) ;
      if (groupIndex >= scaleParameters->size()) {
	stringstream str ;
	str << "Attempts to access group index " << groupIndex << " out of " 
	    << scaleParameters->size() ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return ;
      }
      patReal scale = (*scaleParameters)[groupIndex] ;
      
      vector<patBoolean> dummy ;

      patString name = patModelSpec::the()->getAltName(observation->choice,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }

      patValueVariables::the()->setAttributes(&(observation->attributes)) ;
      
      file << observation->choice ;
      if (saveInDataStructure) {
	resultArray[observationCounter-1][colNumber] = observation->choice ;
	++colNumber ;
      }

      if (patModelSpec::the()->containsDiscreteParameters()) { // contains discrete paramters

	tmp = theDiscreteParamModel->evalProbaLog(observation,
					    NULL,
					    betaParameters,
					    modelParameters,
					    scale,
					    patBadId,
					    patTRUE,
					    dummy,
					    dummy,
					    patFALSE,
					    patFALSE,
					    NULL,
					    NULL,
					    NULL,
					    NULL,
					    mu,
					    NULL,
					    &success,
					    NULL,
					    patFALSE,
					    NULL,
					    err) ;    
	
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
	
	
      }      
      else { // else
	
	tmp = model->evalProbaLog(observation,
				  NULL,
				  betaParameters,
				  modelParameters,
				  scale,
				  patBadId,
				  patTRUE,
				  dummy,
				  dummy,
				  patFALSE,
				  patFALSE,
				  NULL,
				  NULL,
				  NULL,
				  NULL,
				  mu,
				  NULL,
				  &success,
				  NULL,
				  patFALSE,
				  NULL,
				  err) ;    
	
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
	
      }
      if (success) {
	file << '\t' << exp(tmp) ;
	if (saveInDataStructure) {
	  resultArray[observationCounter-1][colNumber] = exp(tmp) ;
	  ++colNumber ;
	}
      }
      else {
	file << '\t' << "NaN" ;
	if (saveInDataStructure) {
	  resultArray[observationCounter-1][colNumber] = 9999.99 ;
	  ++colNumber ;
	}
      }
      vector<patReal> probab(nAlt,0.0) ;
      vector<patReal> utilities(nAlt,0.0) ;
      patReal total(0.0) ;
      
      patUtility* theUtility = model->getUtility() ;
      
      if (includeUtilities()) {
	for (unsigned long i = 0 ; i < nAlt ; ++i) {
	  patULong iii = theIndices->getIndexUtil(i,err) ;
	  if (err != NULL) {
	    WARNING(err->describe());
	    return ;
	  }
	  if (saveInDataStructure && iii != colNumber) {
	    stringstream str ;
	    str << "Inconsistent column number for U" << i << ": " << colNumber << " instead of " << iii ;
	    err = new patErrMiscError(str.str()) ;
	    WARNING(err->describe()) ;
	    return ;
	  }
	  
	  if (observation->availability[i]) {
	    //	    unsigned long userId = patModelSpec::the()->getAltId(i,err) ;
	    if (err != NULL) {
	      WARNING(err->describe());
	      return ;
	    }
	    tmp = theUtility->computeFunction(observation->id,
						      1,
						      i,
						      betaParameters,
						      &(observation->attributes),
						      err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return  ;
	    }
	    
	    utilities[i] = tmp ;
	    file << '\t' << tmp ;
	    if (saveInDataStructure) {
	      resultArray[observationCounter-1][colNumber] = tmp ;
	      ++colNumber ;
	    }
	    
	  }
	  else {
	    utilities[i] = -9999.99 ;
	    file << "\t-9999.99" ;
	    if (saveInDataStructure) {
	      resultArray[observationCounter-1][colNumber] = -9999.99 ;
	      ++colNumber ;
	    }
	  }
	}
      }
      
      for (unsigned long i = 0 ; i < nAlt ; ++i) {
	patULong iii = theIndices->getIndexProba(i,err) ;
	if (saveInDataStructure && iii != colNumber) {
	  stringstream str ;
	  str << "Inconsistent column number for P" << i << ": " << colNumber << " instead of " << iii ;
	  err = new patErrMiscError(str.str()) ;
	  WARNING(err->describe()) ;
	  return ;
	}
	if (observation->availability[i]) {
	  unsigned long userId = patModelSpec::the()->getAltId(i,err) ;
	  if (err != NULL) {
	    WARNING(err->describe());
	    return ;
	  }
	  observation->choice = userId ;
	  if (patModelSpec::the()->containsDiscreteParameters()) { // contains discrete paramters
	    tmp = theDiscreteParamModel->evalProbaLog(observation,
						NULL,
						betaParameters,
						modelParameters,
						scale,
						patBadId,
						patTRUE,
						dummy,
						dummy,
						patFALSE,
						patFALSE,
						NULL,
						NULL,
						NULL,
						NULL,
						mu,
						NULL,
						&success,
						NULL,
						patFALSE,
						NULL,
						err) ;    
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	  }
	  else { // no discrete distribution
	    tmp = model->evalProbaLog(observation,
				      NULL,
				      betaParameters,
				      modelParameters,
				      scale,
				      patBadId,
				      patTRUE,
				      dummy,
				      dummy,
				      patFALSE,
				      patFALSE,
				      NULL,
				      NULL,
				      NULL,
				      NULL,
				      mu,
				      NULL,
				      &success,
				      NULL,
				      patFALSE,
				      NULL,
				      err) ;    
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	  }
	
	  if (success) {
	    file << '\t' << exp(tmp) ;
	    if (saveInDataStructure) {
	      resultArray[observationCounter-1][colNumber] = exp(tmp) ;
	      ++colNumber ;
	    }
	    // Residual
	    if (observation->choice == chosenAlt) {
	      file << '\t' << 1.0 - exp(tmp) ;
	      if (saveInDataStructure) {
		resultArray[observationCounter-1][colNumber] = 1.0-exp(tmp) ;
		++colNumber ;
	      }
	    }
	    else {
	      file << '\t' << - exp(tmp) ;
	      if (saveInDataStructure) {
		resultArray[observationCounter-1][colNumber] = -exp(tmp) ;
		++colNumber ;
	      }

	    }
	    probab[i] = exp(tmp) ;
	    total += probab[i] ;
	  }
	  else {
	    file << "\t9999.99\t9999.99" ;
	    if (saveInDataStructure) {
	      resultArray[observationCounter-1][colNumber] = 9999.99 ;
	      ++colNumber ;
	      // Residual
	      resultArray[observationCounter-1][colNumber] = 9999.99 ;
	      ++colNumber ;
	    }

	  }
	}
	else {
	  file << "\t" << 0.0 ;
	  if (saveInDataStructure) {
	    resultArray[observationCounter-1][colNumber] = 0.0 ;
	    ++colNumber ;
	  }
	  // Residual
	  if (observation->choice == chosenAlt) {
	    file << '\t' << 1.0 ;
	    if (saveInDataStructure) {
	      resultArray[observationCounter-1][colNumber] = 1.0 ;
	      ++colNumber ;
	    }
	  }
	  else {
	    file << '\t' << 0.0 ;
	    if (saveInDataStructure) {
	      resultArray[observationCounter-1][colNumber] = 0.0 ;
	      ++colNumber ;
	    }
	    
	  }
	}
      }
      file << '\t' << total ;
      if (saveInDataStructure) {
	resultArray[observationCounter-1][colNumber] = total ;
	++colNumber ;
      }

      // Add columns for the Zheng-Fosgerau test
      
      unsigned short nbrOfZf = patModelSpec::the()->numberOfZhengFosgerau(NULL) ;
      if (nbrOfZf > 0) {
	patValueVariables::the()->setVariables(patModelSpec::the()->getPtrBetaParameters()) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
	vector<patOneZhengFosgerau>* zf = 
	  patModelSpec::the()->getZhengFosgerauVariables() ;
	for (vector<patOneZhengFosgerau>::iterator ii = zf->begin() ;
	     ii != zf->end() ;
	     ++ii) {
	  if (!ii->isProbability()) {
	    patReal theExpr = ii->expression->getValue(err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	    file << '\t' << theExpr ;
	    if (saveInDataStructure) {
	      resultArray[observationCounter-1][colNumber] = theExpr ;
	      ++colNumber ;
	    }
	  }
	}
      }

      patReal ttt(0.0) ;
      for (unsigned short i = 0 ;
	   i < probab.size() ;
	   ++i) {
	ttt += probab[i] ;
      }

      for (short i = 1 ;
	   i <= patModelSpec::the()->nbrSimulatedChoiceInSampleEnumeration() ; 
	   ++i) {
	unsigned long chosenId = 
	  patDiscreteDistribution(&probab,randomNumbersGenerator)() ;
	stats[i-1][chosenId] += 1 ;
	unsigned long userId = patModelSpec::the()->getAltId(chosenId,err) ;
	file << "\t" << userId ;
	if (saveInDataStructure) {
	  resultArray[observationCounter-1][colNumber] = userId ;
	  ++colNumber ;
	}
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}

      }
      
      for (map<patString, list<long> >::iterator i = patModelSpec::the()->iiaTests.begin() ;
	   i != patModelSpec::the()->iiaTests.end() ;
	   ++i) {
	
	list<long> internalIds ;
	for (list<long>::iterator j = i->second.begin() ;
	     j != i->second.end(); 
	     ++j) {
	  patULong theId = patModelSpec::the()->getAltInternalId(*j,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return ;
	  }
	  internalIds.push_back(theId) ;
	}

	patReal numerator(0.0) ;
	patReal denominator(0.0) ;
	for (list<long>::iterator k = internalIds.begin() ;
	     k != internalIds.end() ;
	     ++k) {
	  numerator += probab[*k] * utilities[*k] ;
	  denominator += probab[*k] ;
	}

	for (list<long>::iterator kk = internalIds.begin() ;
	     kk != internalIds.end() ;
	     ++kk) {
	  file << '\t' << utilities[*kk] - numerator/denominator ;
	}
	
      }
	
      file << endl ;
      
    }


    
    DELETE_PTR(obsIterator) ;
  }


  file.close() ;
  patOutputFiles::the()->addCriticalFile(sampleEnumerationFile,"Results the sample enumeration.");
  DEBUG_MESSAGE("Done.") ;
  for (short i = 1 ;
       i <= patModelSpec::the()->nbrSimulatedChoiceInSampleEnumeration() ; 
       ++i) {
    DEBUG_MESSAGE("Simulation #" << i) ;
    DEBUG_MESSAGE("###############" << i) ;
    for (map<unsigned long,unsigned long>::iterator iter = stats[i-1].begin() ;
	 iter != stats[i-1].end() ;
	 ++iter) {
      DEBUG_MESSAGE("Alt " << iter->first << ":\t" << iter->second) ;
    }
  }
}

patBoolean patSampleEnumeration::includeUtilities() const {

  return patModelSpec::the()->includeUtilitiesInSimulation() ;
}
