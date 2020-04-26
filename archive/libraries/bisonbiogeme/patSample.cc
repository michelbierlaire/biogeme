//-*-c++-*------------------------------------------------------------
//
// File name : patSample.cc
// Author :    Michel Bierlaire
// Date :      Thu Jul 13 15:43:29 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <iomanip>
#include <algorithm>
#include "patUnixUniform.h"
#include "patMath.h"
#include "patFileNames.h"
#include "patParameters.h"
#include "patOutputFiles.h"
#include "patSample.h"
#include "patErrFileNotFound.h"
#include "patErrOutOfRange.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patValueVariables.h"
#include "patArithNode.h"
#include "patLikelihood.h"
#include "patModelSpec.h"
#include "patRandomNumberGenerator.h"
#include "patDataMemory.h"
#include "patDataFile.h"
#include "patIndividualData.h"
#include "patSequenceIterator.h"
#include "patFileIterator.h"
#include "patArrayIterator.h"

patSample::patSample()  :
  nAttributes(patModelSpec::the()->getNbrAttributes()), 
  nAlternatives(patModelSpec::the()->getNbrAlternatives()),
  cases(0),
  nDataRowsInFile(0),
  numberOfObservations(0),
  numberOfAggregateObservations(0),
  dataArrayProvidedByUser(NULL),
  theObsPtr(NULL) {

  if (patParameters::the()->getgevStoreDataOnFile()) {
    DEBUG_MESSAGE("Store data on file") ;
    warehouse = new patDataFile() ;
  }
  else {
    DEBUG_MESSAGE("DO NOT store data on file") ;
    warehouse = new patDataMemory() ;
  }


  for (patULong i = 1 ; i <= nAttributes ; ++i) {
    for (patULong j = 1 ; j <= nAlternatives ; ++j) {
      stringstream str ;
      str << '\t' << "par" << j << "_" << i  ;
      headers.push_back(patString(str.str())) ;
    }
  }
  for (patULong j = 1 ; j <= nAlternatives ; ++j) {
    stringstream str ;
    str << "avail" << j ;
    headers.push_back(patString(str.str())) ;
  }
  headers.push_back("choice") ;
  headers.push_back("group") ;
  
}

patSample::~patSample() {
  DELETE_PTR(warehouse) ;
}


void patSample::processData(patRandomNumberGenerator* normalRndNumbers ,
			    patRandomNumberGenerator* unifRndNumbers   ,
			    patIterator<pair<vector<patString>*        ,
			    vector<patReal>* > >* theIterator          ,  
			    patString dataName                         ,
			    patError*& err                             ) {

  DEBUG_MESSAGE("PROCESS DATA") ;
  if (err != NULL) {
    WARNING(err->describe());
    return;
  }

  if (patModelSpec::the()->isMixedLogit() &&
      patModelSpec::the()->getNumberOfDraws() > 0 && 
      (normalRndNumbers == NULL || unifRndNumbers == NULL) ) {
    err = new patErrNullPointer("patRandomNumberGenerator") ;
    WARNING(err->describe()) ;
    return ;
  }

  levelOfMagnitude.resize(patModelSpec::the()->getNbrUsedAttributes()) ;
  fill(levelOfMagnitude.begin(),
       levelOfMagnitude.end(),
       1.0) ;

  numberOfAttributes.erase(numberOfAttributes.begin(),numberOfAttributes.end()) ;
  meanOfAttributes.erase(meanOfAttributes.begin(),meanOfAttributes.end()) ;
  minOfAttributes.erase(minOfAttributes.begin(),minOfAttributes.end()) ;
  maxOfAttributes.erase(maxOfAttributes.begin(),maxOfAttributes.end()) ;
  
  chosenAlt.erase(chosenAlt.begin(),chosenAlt.end()) ;
  availableAlt.erase(availableAlt.begin(),availableAlt.end()) ;
  weightedChosenAlt.erase(weightedChosenAlt.begin(),weightedChosenAlt.end()) ;
  
  group.erase(group.begin(),group.end()) ;
  weightedGroup.erase(weightedGroup.begin(),weightedGroup.end()) ;

  totalWeight = 0.0 ;

  patString staFile = patFileNames::the()->getStaFile(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  ofstream logFile(staFile.c_str()) ;
  patAbsTime time ;
  time.setTimeOfDay() ;
  logFile << "++++++++ " << time.getTimeString(patTsfFULL) 
	  << " ++++++++" << endl ;

  unsigned long obsExcluded = 0 ;
  unsigned long totalObs = 0 ;

  if (theIterator == NULL) {
    err = new patErrNullPointer("patIterator<vector<patReal>* >") ;
    WARNING(err->describe()) ;
    return ;
  }
  

  ofstream* outputSampleFile(NULL) ;

  if (patParameters::the()->getgevGenerateActualSample()) {
    outputSampleFile = new ofstream(patParameters::the()->getgevOutputActualSample().c_str()) ;

    WARNING("HEADERS ARE NOT COPIED. MUST BE FIXED") ;

  }
  
  patReal readValue ;
  unsigned long rowNumber = 1 ;
  size_t sizeOfMemory = 0.0 ;
    
    
  // Loop on rows
  unsigned long step = patParameters::the()->getgevDataFileDisplayStep() ;
  //  unsigned long currentDraw(0) ;
    
  patReal n(0.0) ;
  map<unsigned long, vector<patVariables> >  individualDraws ;
  map<unsigned long, vector<patVariables> >  individualSnpTerms ;
  
  unsigned long previousPersonId = patBadId ;
  unsigned long panelId = patBadId  ;
  patIndividualData currentPerson ;
  patBoolean newPerson ;
  
  // Used for the distributions with a mass at zero
  patUnixUniform aRandomNumberGenerator(patParameters::the()->getgevSeed()) ;

  unsigned long nRandomParam = 
    patModelSpec::the()->getNbrDrawAttributesPerObservation() ;
  unsigned long nDraws = (nRandomParam > 0) 
    ? patModelSpec::the()->getNumberOfDraws()
    : 0 ;

  
  patObservationData currentRow(patModelSpec::the()->getNbrAlternatives(),
				patModelSpec::the()->getNbrUsedAttributes(),
				nRandomParam,
				patModelSpec::the()->numberOfSnpTerms(),
				nDraws) ;

  patAggregateObservationData currentAggregateObs ;
  patBoolean aggregateObservations = patModelSpec::the()->isAggregateObserved() ;
  

  //  patBoolean applySnp = patModelSpec::the()->applySnpTransform() ;
  patBoolean found ;
  unsigned short indexOfSnpBaseDistribution = 
    patModelSpec::the()->getIdSnpAttribute(&found) ;
  unsigned short nbrOfSnpTerms = patModelSpec::the()->numberOfSnpTerms() ;
  DEBUG_MESSAGE("Index of SNP attribute: " << indexOfSnpBaseDistribution) ;
  for (theIterator->first() ;
       !theIterator->isDone() ;
       theIterator->next()) { // theIterator
    vector<patString>* theHeadersPtr =  theIterator->currentItem().first ;
    if (theHeadersPtr == NULL) {
      err = new patErrNullPointer("vector<patString>") ;
      WARNING("Row in data file: " << rowNumber) ;
      WARNING(err->describe()) ;
      return ;
    }
    vector<patReal>* theRowPtr = theIterator->currentItem().second ;
    
    if (theRowPtr == NULL) {
      err = new patErrNullPointer("vector<patReal>") ;
      WARNING("Row in data file: " << rowNumber) ;
      WARNING(err->describe()) ;
      return ;
    }

    if (theRowPtr->empty()) {
      stringstream str ;
      str << "Row " << rowNumber << " is empty" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }

    if (theRowPtr->size() != theHeadersPtr->size()) {
      err = new patErrMiscError("Incompatibility between row size and nhumber of headers") ;
      WARNING("Row in data file: " << rowNumber) ;
      WARNING(err->describe()) ;
      return ;
    }

    // Default values for the variables
    patModelSpec::the()->resetHeadersValues() ;
    
    // Read the current row
    ++rowNumber ;
    patBoolean printDebugInfo(patFALSE) ;
    if (rowNumber >= patParameters::the()->getgevDebugDataFirstRow() &&
	rowNumber <= patParameters::the()->getgevDebugDataLastRow()) {
      printDebugInfo = patTRUE;
    }
    
    if ((rowNumber % step) == 0) {
      stringstream str ;
      str << "Data " << dataName << "... line " << rowNumber ;
      if (sizeOfMemory >= 1024 * 1024 * 1024) {
	str << "\tMemory: " << floor(patReal(sizeOfMemory) / (1024.0 * 1024.0 * 1024.0)) << " Gb" ;
      }
      else if (sizeOfMemory >= 1024 * 1024) {
	str << "\tMemory: " << floor(patReal(sizeOfMemory) / (1024.0 * 1024.0)) << " Mb" ;
      }
      else if (sizeOfMemory >= 1024) {
	str << "\tMemory: " << floor(patReal(sizeOfMemory) / 1024.0) << " Kb" ;
      }
      GENERAL_MESSAGE(str.str()) ;
    }
    if (printDebugInfo) {
      DEBUG_MESSAGE("******************************") ;
      DEBUG_MESSAGE("Details on row " << rowNumber) ;
      DEBUG_MESSAGE("******************************") ;
    }
    
    stringstream reconstRow ;
    
    //    patBoolean debugFlag = patFALSE ;
    
    ++nDataRowsInFile ;
    for (unsigned long i = 0 ; 
	 i < theRowPtr->size() ;
	 ++i) {

      readValue = (*theRowPtr)[i] ;
      reconstRow << readValue << '\t' ;
      
      patValueVariables::the()->setValue((*theHeadersPtr)[i],readValue) ;
      
      if (printDebugInfo) {
	DEBUG_MESSAGE( (*theHeadersPtr)[i]
		       << "=" << readValue) ;
      }
      if (err != NULL) {
	WARNING("Row in data file: " << rowNumber) ;
	WARNING(err->describe());
	return;
      }
      
    }

    // Check if the observation must be excluded
    
    patArithNode* expr = patModelSpec::the()->getExcludeExpr(err) ;
    if (err != NULL) {
      WARNING("Row in data file: " << rowNumber) ;
      WARNING(err->describe()) ;
      return ;
    }
    patBoolean exclude = (expr == NULL) 
      ? patFALSE 
      : (expr->getValue(err) != 0) ;
    if (err != NULL) {
      WARNING("Row in data file: " << rowNumber) ;
      WARNING(err->describe()) ;
      return ;
    }
    
    if (patParameters::the()->getgevGenerateActualSample() && !exclude) {
      *outputSampleFile << reconstRow.str() << endl ;
    }
    
    if (printDebugInfo && expr != NULL) {
      DEBUG_MESSAGE("Exclusion: " << *expr << "=" << expr->getValue(err)) ;
    }
    
    if (!exclude || printDebugInfo) {
      
      n += 1.0 ;
     
      if (!exclude) {
	expr = patModelSpec::the()->getPanelExpr(err) ;
	if (err != NULL) {
	  WARNING("Row in data file: " << rowNumber) ;
	  WARNING(err->describe()) ;
	  return ;
	}
	if (expr != NULL) {
	  panelDataFeature = patTRUE ;
	  previousPersonId = panelId ;
	  panelId = patULong(expr->getValue(err));
	  if (err != NULL) {
	    WARNING("Row in data file: " << rowNumber) ;
	    WARNING(err->describe()) ;
	    return ;
	  }
	
	}
	else {
	  panelDataFeature = patFALSE ;
	}
      
	map<unsigned long, unsigned long>::iterator foundPerson =
	  nbrOfObsPerIndividual.find(panelId) ;


	if (panelDataFeature && foundPerson != nbrOfObsPerIndividual.end() && 
	    panelId != previousPersonId) {
	  if (foundPerson != nbrOfObsPerIndividual.end()) {
	    DEBUG_MESSAGE("Found person: " << foundPerson->first << ";" << 
			  foundPerson->second) ;
	  }
	  DEBUG_MESSAGE("previous = " << previousPersonId) ;
	  DEBUG_MESSAGE("panelId = " << panelId) ;
	  stringstream str ;
	  str << "It is mandatory to sort the data in order to use the panel data feature." << endl ;
	  str << "Observations for person " << panelId << " are not contiguous" ;
	  err = new patErrMiscError(str.str()) ;
	  WARNING(err->describe()) ;
	  return ;
	}
	//       if ((!aggregateObservations) || currentRow.isLast) {
	// 	nbrOfObsPerIndividual[panelId] = 
	// 	  nbrOfObsPerIndividual[panelId]+ 1 ; 
	//       };
	currentPerson.panelId = panelId ;
	
	// Fill in the random numbers first
	
	// map: first panelId, second draws for a given individual 
	map<unsigned long, vector<patVariables> >::iterator found = 
	  individualDraws.find(panelId) ;
	if (found == individualDraws.end()) {
	  individualDraws[panelId] = 
	    vector<patVariables>(nDraws,patVariables(nRandomParam)) ;
	}
	
	map<unsigned long, vector<patVariables> >::iterator foundZeroOneDrawsForSnp = 
	  individualSnpTerms.find(panelId) ;
	if (foundZeroOneDrawsForSnp == individualSnpTerms.end()) {
	  individualSnpTerms[panelId] = 
	    vector<patVariables>(nDraws,patVariables(nbrOfSnpTerms)) ;
	}
	
	for (unsigned long i = 0 ;
	     i < patModelSpec::the()->getNbrDrawAttributesPerObservation() ;
	     ++i) {
	  patBoolean isPanel = patModelSpec::the()->isDrawPanel(i,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return ;
	  }
	  
	  patReal massAtZero = patModelSpec::the()->getMassAtZero(i,err) ;
	  
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return ;
	  }
	  
	  patBoolean recordDrawsForSnp(i == indexOfSnpBaseDistribution) ;
	  
	  
	  for (unsigned long j = 0 ;
	       j < nDraws ;
	       ++j) {
	    patReal value ;
	    patDistribType theType = 
	      patModelSpec::the()->getDistributionOfDraw(i,err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	    if (isPanel && found != individualDraws.end()) {
	      currentRow.draws[j][i] = (found->second)[j][i]  ;
	      currentRow.unifDrawsForSnpPolynomial[j] = (foundZeroOneDrawsForSnp->second)[j] ;
	      newPerson = patFALSE ;
	    }
	    else {
	      patReal zeroOneDraw ;
	      pair<patReal,patReal> theDraw ;
	      switch (theType) {
	      case NORMAL_DIST:
		theDraw = normalRndNumbers->getNextValue(err) ;
		value = theDraw.first ;
		if (err != NULL) {
		  WARNING(err->describe()) ;
		  return ;
		}
		zeroOneDraw = theDraw.second ;
		if (err != NULL) {
		  WARNING(err->describe()) ;
		  return ;
		}
		break ;
	      case UNIF_DIST:
		theDraw = unifRndNumbers->getNextValue(err) ;
		value = theDraw.first ;
		if (err != NULL) {
		  WARNING(err->describe()) ;
		  return ;
		}
		zeroOneDraw = theDraw.second ;
		if (err != NULL) {
		  WARNING(err->describe()) ;
		  return ;
		}
		break ;
	      default:
		stringstream str ;
		str << "Unknown distribution for draw number " << i ;
		err = new patErrMiscError(str.str()) ;
		WARNING(err->describe()) ;
		return ;
	      }
	      
	      if (massAtZero > 0) {
		patReal tir = aRandomNumberGenerator.getUniform(err) ;
		if (err != NULL) {
		  WARNING(err->describe()) ;
		  return ;
		}
		if (tir <= massAtZero) {
		  value = -1.0 ;
		}
	      }
	      
	      
	      currentRow.draws[j][i] = value ;
	      (individualDraws[panelId])[j][i] = value ;
	      if (recordDrawsForSnp) {
		// Compute the Legendre polynomials
		for (unsigned short term = 0 ; 
		     term < nbrOfSnpTerms ; 
		     ++term) {
		  unsigned short order = patModelSpec::the()->orderOfSnpTerm(term,err) ;
		  if (err != NULL) {
		    WARNING(err->describe()) ;
		    return ;
		  }
		  currentRow.unifDrawsForSnpPolynomial[j][term] = 
		    (individualSnpTerms[panelId])[j][term] = 
		    legendrePolynomials.evaluate(order,zeroOneDraw) ;
		}
	      }
	      newPerson = patTRUE ;
	    }
	  }
	  
	}
	
	if (printDebugInfo) {
	  DEBUG_MESSAGE("------ Calculated values ------") ;
	}
      }


      // Then the attributes computed from the data file
	
      patIterator<pair<patString,unsigned long> >* iter =
	patModelSpec::the()->createUsedAttributeNamesIterator() ;
	
      for (iter->first() ;
	   !iter->isDone() ;
	   iter->next()) {
	  
	pair<patString,unsigned long> attr = iter->currentItem() ;
	  
	patArithNode* expr = 
	  patModelSpec::the()->getVariableExpr(attr.first,
					       err) ;
	  
	if (err != NULL) {
	  WARNING("Row in data file: " << rowNumber) ;
	  WARNING("Attribute:        " << attr.first) ;
	  WARNING(err->describe()) ;
	  return ;
	}
	if (expr == NULL) {
	  err = new patErrNullPointer("patArithNode") ;
	  WARNING("Row in data file: " << rowNumber) ;
	  WARNING("Attribute:        " << attr.first) ;
	  WARNING(err->describe()) ;
	  return ;
	}
	patReal tmp = expr->getValue(err) ;
	if (expr == NULL) {
	  err = new patErrNullPointer("patArithNode") ;
	  WARNING("Row in data file: " << rowNumber) ;
	  WARNING("Attribute:        " << attr.first) ;
	  WARNING(err->describe()) ;
	  return ;
	}
	if (printDebugInfo) {
	  DEBUG_MESSAGE(*expr << "=" << tmp) ;
	}
	
	/// 
	  ///    CHECK WHAT IS THE USE OF THE FILE ID IN THE DATA
	  ///
	  //	  currentRow.fileId = fileId ;
	  if (!exclude) {
	    currentRow.fileId = patBadId ;
	    currentRow.attributes[attr.second].name = attr.first ;
	    currentRow.attributes[attr.second].value = tmp ;
	  
	    // Collect statistics
	    
	    if (tmp != patParameters::the()->getgevMissingValue()) {
	      map<patString,patReal>::iterator jter = 
		meanOfAttributes.find(attr.first) ;
	      if (jter == meanOfAttributes.end()) {
		meanOfAttributes[attr.first] = tmp ;
		minOfAttributes[attr.first] = tmp ;
		maxOfAttributes[attr.first] = tmp ;
		numberOfAttributes[attr.first] = 1 ;
	      }
	      else {
		++numberOfAttributes[attr.first] ;
		patReal oldValue = meanOfAttributes[attr.first] ;
		meanOfAttributes[attr.first] = 
		  (n * oldValue + tmp) / (n+1.0) ;
		if (tmp < minOfAttributes[attr.first]) {
		  minOfAttributes[attr.first] = tmp ;
		}
		if (tmp > maxOfAttributes[attr.first]) {
		  maxOfAttributes[attr.first] = tmp ;
		}
	      }
	    }
	  }
      }
	
      DELETE_PTR(iter) ;

	
#ifndef GIANLUCA
      for (unsigned long alt = 0 ; 
	   alt < patModelSpec::the()->getNbrAlternatives() ; 
	   ++alt) {  
	
	unsigned long userId = patModelSpec::the()->getAltId(alt,err) ;
	if (err != NULL) {
	  WARNING("Row in data file: " << rowNumber) ;
	  WARNING(err->describe()) ;
	  return ;
	}
	
	// Availability
	  
	patArithNode* expr = 
	  patModelSpec::the()->
	  getAvailExpr(patModelSpec::the()->getAltId(alt,err),
		       err) ;
	  
	if (err != NULL) {
	  WARNING("Row in data file: " << rowNumber) ;
	  WARNING(err->describe()) ;
	  return ;
	}
	  
	if (expr == NULL) {
	    
	  if (err != NULL) {
	    WARNING("Row in data file: " << rowNumber) ;
	    WARNING(err->describe()) ;
	    return ;
	  }
	  DEBUG_MESSAGE("Row " << rowNumber << ": No availability defined for alt. " << userId << ". Assumed not available") ;
	  currentRow.availability[alt] = patFALSE ;
	}
	else {
	  patReal availValue = expr->getValue(err) ;
	  if (err != NULL) {
	    WARNING("Row in data file: " << rowNumber) ;
	    WARNING(err->describe()) ;
	    return ;
	  }
	  currentRow.availability[alt] = 
	    ( availValue != 0.0 && 
	      availValue != patParameters::the()->getgevMissingValue() ) ;
	}
	if (printDebugInfo) {
	  if (expr != NULL) {
	    DEBUG_MESSAGE("Availability of alt. "
			  << patModelSpec::the()->getAltId(alt,err) 
			  <<" : " << *expr << "=" 
			  << expr->getValue(err)) ;
	  }
	}
	if (!exclude) {
	  if (currentRow.availability[alt]) {
	    ++availableAlt[userId] ; 
	    ++cases ;
	  }
	  else {
	    patModelSpec::the()->allAlternativesAlwaysAvail = patFALSE ;
	  }
	  
	  if (err != NULL) {
	    WARNING("Row in data file: " << rowNumber) ;
	    WARNING(err->describe());
	    return;
	  }
	}
      }

      expr = patModelSpec::the()->getChoiceExpr(err) ;
      if (err != NULL) {
	WARNING("Row in data file: " << rowNumber) ;
	WARNING(err->describe()) ;
	return ;
      }
      if (expr == NULL) {
	err = new patErrMiscError("Choice is not defined properly") ;
	WARNING("Row in data file: " << rowNumber) ;
	WARNING(err->describe()) ;
	return ;
      }
      if (!exclude) {
	currentRow.choice = unsigned(expr->getValue(err)) ;
	if (err != NULL) {
	  WARNING("Row in data file: " << rowNumber) ;
	  WARNING(err->describe());
	  return;
	}
	
	
	if (currentRow.choice == patParameters::the()->getgevMissingValue()) {
	  WARNING("Row in data file: " << rowNumber) ;
	  stringstream str ;
	  str << "Choice is defined by the missing data value: " << patParameters::the()->getgevMissingValue() ;
	  err = new patErrMiscError(str.str()) ;
	  WARNING(err->describe()) ;
	  return ;
	}
      }
      if (printDebugInfo) {
	DEBUG_MESSAGE("[Choice] " << *expr << "=" 
		      << unsigned(expr->getValue(err))) ;
      }
	
      expr = patModelSpec::the()->getWeightExpr(err) ;
      if (err != NULL) {
	WARNING("Row in data file: " << rowNumber) ;
	WARNING(err->describe()) ;
	return ;
      }
      if (expr != NULL) {
	if (!exclude) {
	  currentRow.weight = patReal(expr->getValue(err)) ;
	  if (currentRow.weight == patParameters::the()->getgevMissingValue()) {
	    WARNING("Row in data file: " << rowNumber) ;
	    stringstream str ;
	    str << "Weight is defined by the missing data value: " << patParameters::the()->getgevMissingValue() ;
	    err = new patErrMiscError(str.str()) ;
	    WARNING(err->describe()) ;
	    return ;
	  }
	  totalWeight += currentRow.weight ;
	  if (err != NULL) {
	    WARNING("Row in data file: " << rowNumber) ;
	    WARNING(err->describe());
	    return;
	  }
	}
	if (printDebugInfo) {
	  DEBUG_MESSAGE("[Weight]: " << *expr 
			<< "=" << patReal(expr->getValue(err))) ;
	}
	if (currentRow.weight <= 0.0 && !exclude) {
	  logFile << "WARNING (row " << rowNumber 
		  << "): Weight is not strictly positive: " 
		  << currentRow.weight << endl ; 
	  WARNING("(row " << rowNumber 
		  << "): Weight is not strictly positive" 
		  << currentRow.weight) ; 
	}
      }
	


      expr = patModelSpec::the()->getAggLastExpr(err) ;
      if (err != NULL) {
	WARNING("Row in data file: " << rowNumber) ;
	WARNING(err->describe()) ;
	return ;
      }
      if (expr != NULL) {
	currentRow.isLast = (patReal(expr->getValue(err)) != 0) ;
	if (err != NULL) {
	  WARNING("Row in data file: " << rowNumber) ;
	  WARNING(err->describe());
	  return;
	}
	if (printDebugInfo) {
	  DEBUG_MESSAGE("[AggregateLast]: " << *expr 
			<< "=" << patReal(expr->getValue(err))) ;
	}
      }
	
      expr = patModelSpec::the()->getAggWeightExpr(err) ;
      if (err != NULL) {
	WARNING("Row in data file: " << rowNumber) ;
	WARNING(err->describe()) ;
	return ;
      }
      if (expr != NULL) {
	if (!exclude) {
	  currentRow.aggWeight = patReal(expr->getValue(err)) ;
	  if (err != NULL) {
	    WARNING("Row in data file: " << rowNumber) ;
	    WARNING(err->describe());
	    return;
	  }
	  if ( currentRow.aggWeight == 0) {
	    stringstream str ;
	    str << "Aggregate weight is zero on row " << rowNumber ;
	    err = new patErrMiscError(str.str()) ;
	    WARNING(err->describe());
	    return ;
	  }
	  
	  if (currentRow.aggWeight == patParameters::the()->getgevMissingValue()) {
	    WARNING("Row in data file: " << rowNumber) ;
	    stringstream str ;
	    str << "Aggregate weight is defined by the missing data value: " << patParameters::the()->getgevMissingValue() ;
	    err = new patErrMiscError(str.str()) ;
	    WARNING(err->describe()) ;
	    return ;
	  }
	}
	if (printDebugInfo) {
	  DEBUG_MESSAGE("[AggregateWeight]: " << *expr 
			<< "=" << patReal(expr->getValue(err))) ;
	}
      }

      
      unsigned long choiceID = 
	patModelSpec::the()->getAltInternalId(currentRow.choice,err) ;
      if (err != NULL) {
	logFile << "WARNING (row " << rowNumber << ") " ; 
	logFile << err->describe() << endl ;
	DELETE_PTR(err) ;
	err = NULL ;
	++obsExcluded ;
	exclude = patTRUE ;
      }
      else {
	
	// Warning: for the ordinal   logit model, choiceID is 0
	if (!currentRow.availability[choiceID]) {
	  DEBUG_MESSAGE("(row " << rowNumber << "). Chosen alt. " 
			<<  currentRow.choice << " not available. ") ;
	  logFile << "WARNING (row " << rowNumber << "). Chosen alt. " 
		  <<  currentRow.choice << " not available. "  ;
	  logFile << "Obs. excluded." << endl ;
	  ++obsExcluded ;
	  exclude = patTRUE ;
	    
	}
	else {
	  expr = patModelSpec::the()->getGroupExpr(err) ;
	  if (err != NULL) {
	    WARNING("Row in data file: " << rowNumber) ;
	    WARNING(err->describe()) ;
	    return ;
	  }
	  if (expr == NULL) {
	    err = new patErrNullPointer("patArithNode") ;
	    WARNING("Row in data file: " << rowNumber) ;
	    WARNING(err->describe()) ;
	    return ;
	  }
	  currentRow.group = unsigned(expr->getValue(err)) ;  
	  if (err != NULL) {
	    WARNING("Row in data file: " << rowNumber) ;
	    WARNING(err->describe());
	    return;
	  }
	  if (currentRow.group == patParameters::the()->getgevMissingValue()) {
	    WARNING("Row in data file: " << rowNumber) ;
	    stringstream str ;
	    str << "Group is defined by the missing data value: " << patParameters::the()->getgevMissingValue() ;
	    err = new patErrMiscError(str.str()) ;
	    WARNING(err->describe()) ;
	    return ;
	  }

	  if (printDebugInfo) {
	    DEBUG_MESSAGE("[Group] " << *expr << "=" 
			  << unsigned(expr->getValue(err))) ;
	  }
	  currentRow.id = rowNumber-1 ;
	    
	  ++chosenAlt[currentRow.choice] ;
	  ++group[currentRow.group] ;
	  weightedChosenAlt[currentRow.choice] += currentRow.weight ;
	  weightedGroup[currentRow.group] += currentRow.weight ;
	}
      }

#endif
      if (!exclude) {
	if (!panelDataFeature ||  panelId != previousPersonId) {
	  if (aggregateObservations) {
	    if (!currentPerson.theAggregateObservations.empty()){
	      warehouse->push_back(&currentPerson) ;
	      currentPerson.theAggregateObservations.erase(currentPerson.theAggregateObservations.begin(),
							   currentPerson.theAggregateObservations.end()) ;
	    }

	  }
	  else {
	    if (!currentPerson.theObservations.empty()){
	      warehouse->push_back(&currentPerson) ;
	      currentPerson.theObservations.erase(currentPerson.theObservations.begin(),
						  currentPerson.theObservations.end()) ;
	    }
	  }
	}

	if (panelDataFeature) {
	  if (!currentPerson.theObservations.empty() && 
	      currentRow.weight != currentPerson.getWeight()) {
	    stringstream str ;
	    str << "Error for Ind. " << currentPerson.panelId 
		<< ": Weights must be constant across observations of a same individual" << endl ;
	    err = new patErrMiscError(str.str()) ;
	    WARNING(err->describe()) ;
	    return ;
	      
	  }
	  if (!currentPerson.theObservations.empty() && 
	      currentRow.group != currentPerson.getGroup()) {
	    stringstream str ;
	    str << "Error for Ind. " << currentPerson.panelId 
		<< ": Groups must be constant across observations of a same individual (" <<  currentRow.group << "<>" <<currentPerson.getGroup() << ")" << endl ;
	    str << currentPerson.theObservations.size() << " obs. " << endl ;
	    err = new patErrMiscError(str.str()) ;
	    WARNING(err->describe()) ;
	    return ;
	  }
	}


	currentPerson.panelId = panelId ;

	//
	if (currentPerson.theObservations.empty()) {
	  // First observation
	}
	    
	if (aggregateObservations) {
	  currentAggregateObs.theObservations.push_back(currentRow) ;
	  if (currentRow.isLast) {
	    ++numberOfAggregateObservations ;
	    currentPerson.theAggregateObservations.push_back(currentAggregateObs) ;
	    currentAggregateObs.theObservations.
	      erase(currentAggregateObs.theObservations.begin(),
		    currentAggregateObs.theObservations.end()) ;
	  }
	}
	else {
	  currentPerson.theObservations.push_back(currentRow) ;
	  ++nbrOfObsPerIndividual[panelId] ;
	}
	    
	sizeOfMemory += currentRow.memory ;
	//cout << "[" << getSampleSize() << "]" ;
	//cout << "*" ;
      }
    }
    else {
      ++obsExcluded ;
    }
	
  }

  if (outputSampleFile != NULL) {
    outputSampleFile->close() ;
  patOutputFiles::the()->addUsefulFile(patParameters::the()->getgevOutputActualSample(),"Sample file that was actually used for estimation");
    DELETE_PTR(outputSampleFile) ;
  }
  if (aggregateObservations) {
    if (!currentPerson.theAggregateObservations.empty()){
      warehouse->push_back(&currentPerson) ;
    }
  }
  else {
    if (!currentPerson.theObservations.empty()){
      warehouse->push_back(&currentPerson) ;
    }
  }
  GENERAL_MESSAGE("Total obs.:   " << rowNumber-2) ;
  GENERAL_MESSAGE("Total memory: " << patReal(sizeOfMemory) / 1024 << " Kb") ;
  if (patModelSpec::the()->isAggregateObserved()) {
    GENERAL_MESSAGE("Aggregate observations: " << numberOfAggregateObservations) ;
  }
  totalObs += rowNumber-2 ;
  

  warehouse->finalize() ;
  patIterator<pair<patString,unsigned long> >* iter =
    patModelSpec::the()->createUsedAttributeNamesIterator() ;
  
  for (iter->first() ;
       !iter->isDone() ;
       iter->next()) {
    pair<patString,unsigned long> attr = iter->currentItem() ;
    patReal l = 
      patMax(patZero,ceil(log10(patAbs(meanOfAttributes[attr.first])))) ; 
    levelOfMagnitude[attr.second] = pow(10.0,l) ;
  }

  DELETE_PTR(iter) ;

  // unsigned long total = getSampleSize() *  
  //   patModelSpec::the()->getNbrAlternatives() * 
  //   patModelSpec::the()->getNbrOrigBeta() ;

  logFile << "Sample size=" << getSampleSize() << endl ;
  if (patModelSpec::the()->getWeightExpr(err) != NULL) {
    logFile << "Total weight=" << totalWeight << endl ;
    if (getSampleSize() != totalWeight) {
      logFile << "     --> It is recommended to multiply all weights by " 
	      << setiosflags(ios::scientific)
	      << patReal(getSampleSize())/totalWeight 
	      << resetiosflags(ios::scientific)
	      << endl ;
    }
  }
  logFile << "Excluded Obs.:       " << obsExcluded << endl ;
  logFile << "Total obs. in files: " << totalObs << endl ;
  logFile << "Number of cases:     " << getCases() << endl ;
  logFile << "Statistics of attributes" << endl ;
  logFile << "++++++++++++++++++++++++" << endl ;
  logFile << "Name\tNbr\tMean\tMin\tMax\tRecommended/Conservative upper bounds for Box-Cox" << endl ;
  for (map<patString,patReal>::iterator it = meanOfAttributes.begin() ;
       it != meanOfAttributes.end() ;
       ++it) {
    logFile << it->first <<'\t' 
	    << numberOfAttributes[it->first] << '\t'
	    << it->second << '\t' 
	    << minOfAttributes[it->first] << '\t' 
	    << maxOfAttributes[it->first] << '\t' ;
//     if (minOfAttributes[it->first] >= 0) {
//       logFile << log(patLogMaxReal::the()) / log(it->second) << '\t'
// 	      << log(patLogMaxReal::the()) / log(maxOfAttributes[it->first]) ;
//     }
//     else {
//       logFile << log(patLogMaxReal::the()) / log(it->second-minOfAttributes[it->first]) << '\t'
// 	      << log(patLogMaxReal::the()) / log(maxOfAttributes[it->first]-minOfAttributes[it->first]) << " [Box-Tukey mandatory]";
//     }
    logFile << endl ;
    
  }
  logFile << "Nbr of times alternatives are available" << endl ; 
  logFile << "Alt\t#" << endl ;
  for (map<unsigned long, unsigned long>::iterator it = availableAlt.begin() ;
       it != availableAlt.end() ;
       ++it) {
    logFile << it->first << '\t' << it->second << endl ;
  }
  logFile << "Nbr of times alternatives are chosen" << endl ; 
  logFile << "Alt\t#" << endl ;
  for (map<unsigned long, unsigned long>::iterator it = chosenAlt.begin() ;
       it != chosenAlt.end() ;
       ++it) {
    logFile << it->first << '\t' << it->second << endl ;
  }
  if (patModelSpec::the()->isSampleWeighted()) {
    logFile << "Nbr of times alternatives are chosen [weighted]" << endl ; 
    logFile << "Alt\t#" << endl ;
    for (map<unsigned long, patReal>::iterator it = weightedChosenAlt.begin() ;
	 it != weightedChosenAlt.end() ;
	 ++it) {
      logFile << it->first << '\t' << it->second << endl ;
    }
  }

  logFile << "Group membership" << endl ;
  logFile << "Group\t#" << endl ;
  for (map< long,  long>::iterator it = group.begin() ;
       it != group.end() ;
       ++it) {
    logFile << it->first << '\t' << it->second << endl ;
  }
  if (patModelSpec::the()->isSampleWeighted()) {
    logFile << "Group membership [weighted]" << endl ;
    logFile << "Group\t#" << endl ;
    for (map< long,  patReal>::iterator it = weightedGroup.begin() ;
	 it != weightedGroup.end() ;
	 ++it) {
      logFile << it->first << '\t' << it->second << endl ;
    }
  }
  if (panelDataFeature) {
    logFile << endl ;
    logFile << "Description of the panel data" << endl ;
    logFile << "Individual\t#Observations" << endl ;
    for (map<unsigned long, unsigned long>::const_iterator i =
	   nbrOfObsPerIndividual.begin() ;
	 i != nbrOfObsPerIndividual.end() ;
	 ++i) {
      logFile << i->first << '\t' << i->second << endl ;
    }
  }
  logFile.close() ;
  patOutputFiles::the()->addUsefulFile(staFile,"Statistics about the sample file");

//  DEBUG_MESSAGE("Sample size=" << getSampleSize()) ;

  //DEBUG_MESSAGE("There are " << numberOfGroups() << " groups") ;
  DETAILED_MESSAGE("Detailed info in " << patFileNames::the()->getStaFile(err)) ;

}
void patSample::readDataFile(patRandomNumberGenerator* normalRndNumbers,
			     patRandomNumberGenerator* unifRndNumbers,
			     patError*& err) {


  if (dataArrayProvidedByUser != NULL) {

    GENERAL_MESSAGE("Read data from external data structure...") ;
    vector<patString> headers = patModelSpec::the()->getHeaders() ;
    patIterator<pair<vector<patString>*,vector<patReal>* > >* theIterator =
      new patArrayIterator(dataArrayProvidedByUser,
			   nRows,
			   nColumns,
			   &headers) ;
    if (theIterator == NULL) {
      err = new patErrNullPointer("patIterator<vector<patReal>* >") ;
      WARNING(err->describe()) ;
      return ;
    }
    
    processData(normalRndNumbers,
		unifRndNumbers,
		theIterator,
		patString(" data structure "),
		err) ;
    
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    DELETE_PTR(theIterator) ;
  }
  else {
    
    unsigned short nbrSampleFiles = patFileNames::the()->getNbrSampleFiles() ;
    patSequenceIterator<pair<vector<patString>*,vector<patReal>* > > 
      iteratorOnAllFiles  ;
    
    for (unsigned short fileId = 0 ; fileId < nbrSampleFiles ; ++fileId) {
      patString fileName = patFileNames::the()->getSamFile(fileId,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return  ;
      }
      GENERAL_MESSAGE("Opening file " << fileName) ;
      
      
      patIterator<pair<vector<patString>*,vector<patReal>* > >* theIterator = createFileIterator(fileName,
												 patModelSpec::the()->getNbrDataPerRow(fileId),
												 patModelSpec::the()->getHeaders(fileId)) ;
      
    
      if (theIterator == NULL) {
	err = new patErrNullPointer("patIterator<vector<patReal>* >") ;
	WARNING(err->describe()) ;
	return ;
      }
    
      iteratorOnAllFiles.addIterator(theIterator) ;
    
    
    }

    processData(normalRndNumbers,
		unifRndNumbers,
		&iteratorOnAllFiles,
		patString(" file"),
		err) ;

    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }

    

  }
}
  
unsigned long patSample::getSampleSize() const {
  return warehouse->getSize() ;
}

void patSample::empty() {
  warehouse->erase() ;
}

patIterator<patObservationData*>* patSample::createObsIterator() {
  return warehouse->createObsIterator() ;
}

vector<patIterator<patObservationData*>*> 
patSample::createObsIteratorThread(unsigned int nbrThreads, 
				   patError*& err) {
  return warehouse->createObsIteratorThread(nbrThreads,err) ;
}


vector<patIterator<patIndividualData*>*> 
patSample::createIndIteratorThread(unsigned int nbrThreads, 
				   patError*& err) {
  return warehouse->createIndIteratorThread(nbrThreads,err) ;
}


patIterator<patAggregateObservationData*>* patSample::createAggObsIterator() {
  return warehouse->createAggObsIterator() ;
}


patIterator<patIndividualData*>* patSample::createIndIterator() {
  return warehouse->createIndIterator() ;
}

unsigned long patSample::numberOfGroups() const {
  return group.size() ;
}

unsigned long patSample::groupIndex(unsigned long groupLabel)  {
  if (grIndex.empty()) {
    grIndex.resize(patModelSpec::the()->getLargestGroupUserId()+1,patBadId) ;
  }
  if (grIndex[groupLabel] == patBadId) {
    patError* err = NULL ;
    patBetaLikeParameter theScale = patModelSpec::the()->getScale(groupLabel,err) ;
    if (err != NULL) {
      WARNING("***********************") ;
      WARNING(err->describe()) 
      WARNING("***********************") ;
      return patBadId ;
    }
    grIndex[groupLabel] = theScale.id ;
    return theScale.id ;
  }
  else {
    return grIndex[groupLabel] ;
  }
}

void patSample::generateSimulatedData(patLikelihood* like, 
				      patError*& err) {

  if (err != NULL) {
    WARNING(err->describe());
    return;
  }

  WARNING("Must be reimplemented") ;


}


unsigned long patSample::getCases() const {
  return (cases-getSampleSize()) ;
}

unsigned long patSample::getNbrOfDataRowsInFile() {
  return nDataRowsInFile ;
}


patVariables* patSample::getLevelOfMagnitude(patError*& err) {
  return &levelOfMagnitude ;
}

void patSample::scaleAttributes(patVariables* scales,
				patError*& err) {
  if (scales == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return ;
  }

  patReal beforeMean = 0.0 ;
  patReal afterMean = 0.0 ;

  patIterator<patObservationData*>* obsIterator = createObsIterator() ;

  DEBUG_MESSAGE("Start the loop here") ;
  for (obsIterator->first() ;
       !obsIterator->isDone() ;
       obsIterator->next()) {

    static patBoolean first = patTRUE ;
    patObservationData* obs = obsIterator->currentItem() ;
    
    if (first) {
      first = patFALSE ;
    }
    if (obs->attributes.size() != levelOfMagnitude.size()) {
      stringstream str ;
      str << "Inconsistent sizes: " << obs->attributes.size() 
	  << " and " << levelOfMagnitude.size() ;
      err = new patErrMiscError(
str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
    for (unsigned long i = 0 ; i < obs->attributes.size() ; ++i) {
      beforeMean += obs->attributes[i].value ;
      if (i >= scales->size()) {
	err =  new patErrOutOfRange<unsigned long>(i,0,scales->size()) ;
	WARNING(err->describe()) ;
	return ;
      }
      obs->attributes[i].value /= (*scales)[i] ;
      afterMean += obs->attributes[i].value ;
    }
  }
}


// void patSample::generateDataForDenis(patError*& err) {

//   DEBUG_MESSAGE("--- FILE FOR DENIS --- ") ;
//   DEBUG_MESSAGE(sample.begin()->attributes.size() << " x's") ;
//   DEBUG_MESSAGE(sample.begin()->availability.size() << " alts") ;

//   ofstream denis("denis.dat") ;
//   for (iter = sample.begin() ;
//        iter != sample.end() ;
//        ++iter) {
//       patString name = patModelSpec::the()->getAltName(iter->choice,err) ;
//       denis << 1+patModelSpec::the()->getAltInternalId(name) << " " ;
//     for (unsigned long i = 0 ; i < iter->attributes.size() ; ++i) {
//       denis << iter->attributes[i].value << " " ;
//     }
//     for (unsigned long i = 0 ; i < iter->availability.size() ; ++i) {
//       if (iter->availability[i]) {
// 	denis << "1 " ;
//       }
//       else {
// 	denis << "0 " ;
//       }
//     }
//     denis << endl ;
//   }
//   denis.close() ;
// }

void patSample::shuffleSample() {
  if (patParameters::the()->getgevStoreDataOnFile()) {
    WARNING("Cannot shuffle data on file") ;
    return ;
  }
  patDataMemory* thedata = (patDataMemory*)warehouse ;

  
  thedata->shuffleSample() ;

}

unsigned long patSample::nbrObsForPerson(unsigned long i) {
  map<unsigned long, unsigned long>::iterator found =
    nbrOfObsPerIndividual.find(i) ;
  if (found == nbrOfObsPerIndividual.end()) {
    return 0 ;
  }
  else {
    return found->second ;
  }
}

unsigned long patSample::getNumberOfIndividuals() const {
  return getSampleSize() ;
}

unsigned long patSample::getNumberOfObservations() {
  if (numberOfObservations == 0) {
    for (map<unsigned long, unsigned long>::const_iterator i = 
	   nbrOfObsPerIndividual.begin() ;
	 i != nbrOfObsPerIndividual.end() ;
	 ++i) {
      numberOfObservations += i->second ;
    }
  }
  return numberOfObservations ;
}

unsigned long patSample::getNumberOfAggregateObservations() {
  if (numberOfAggregateObservations == 0) {
    for (map<unsigned long, unsigned long>::const_iterator i = 
	   nbrOfObsPerIndividual.begin() ;
	 i != nbrOfObsPerIndividual.end() ;
	 ++i) {
      numberOfObservations += i->second ;
    }
  }
  return numberOfAggregateObservations ;
}

patIterator<pair<vector<patString>*,vector<patReal>  * > >* 
  patSample::createFileIterator(patString fileName, 
				unsigned long dataPerRow,
				vector<patString>* header) {
  return new patFileIterator(fileName,
			     dataPerRow,
			     header) ;
 }
 
void patSample::externalDataStructure(patPythonReal** array, 
				      unsigned long nr, 
				      unsigned nc) {
  dataArrayProvidedByUser = array ;
  nRows = nr ;
  nColumns = nc ;
}

patReal patSample::computeLogLikeWithCte() {

  patReal logLike = 0.0 ;
  patReal total = 0.0 ;
  for (map<unsigned long, unsigned long>::iterator it = chosenAlt.begin() ;
       it != chosenAlt.end() ;
       ++it) {
    patReal nj = patReal(it->second) ;
    logLike += nj * log(nj) ;
    total += nj ;
  }
  logLike -= total * log(total) ;
  return logLike ;
}

