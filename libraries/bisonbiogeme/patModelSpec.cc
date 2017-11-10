//-*-c++-*------------------------------------------------------------
//
// File name : patModelSpec.cc
// Author :    Michel Bierlaire
// Date :      Tue Nov  7 16:27:33 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <algorithm>
#include <set>
#include <iomanip>
#include "patMath.h"
#include "patNL.h"
#include "patFileNames.h"
#include "patOutputFiles.h"
#include "patValueVariables.h"
#include "patLinearUtility.h"
#include "patGeneralizedUtility.h"
#include "patAdditiveUtility.h"
#include "patAttributeNamesIterator.h"
#include "patUsedAttributeNamesIterator.h"
#include "patArithAttribute.h"
#include "patFileExists.h"
#include "patPValue.h"
#include "patPythonResults.h"
#include "patOutputTable.h"
#include "patBisonSingletonFactory.h"
// #ifndef patNO_MTL
// #include "patMtl.h"
// #endif

#include "patType.h"
#include "patParameters.h"
#include "patModelSpec.h"
#include "patSpecParser.hh"
//#include "modelParser.h"
#include "patErrMiscError.h"
#include "patErrOutOfRange.h"
#include "patErrNullPointer.h"
#include "patVersion.h"
#include "patBetaLikeIterator.h"
#include "patCorrelation.h"
#include "patCompareCorrelation.h"
#include "patNlNestIterator.h"
#include "patCnlAlphaIterator.h"
#include "patFullCnlAlphaIterator.h"
#include "patStlVectorIterator.h"
#include "patConstraintNestIterator.h"
#include "patSequenceIterator.h"
#include "patStlVectorIterator.h"
#include "patHybridMatrix.h"
#include "patNetworkAlphaIterator.h"
#include "patNetworkGevModel.h"

patModelSpec::patModelSpec() : 
  rootNodeName(patString("__ROOT")),
  automaticScaling(patFALSE),  
  choiceExpr(NULL),
  aggLastExpr(NULL),
  aggWeightExpr(NULL),
  panelExpr(NULL),
  weightExpr(NULL),
  excludeExpr(NULL),
  groupExpr(NULL),
  columnData(0) ,
  largestAlternativeUserId(0),
  largestGroupUserId(0),
  altInternalIdComputed(patFALSE),
  indexComputed(patFALSE),
  sampleEnumeration(0), 
  theNetworkGevModel(NULL),
  logitKernelChecked(patFALSE),
  equalityConstraints(NULL) ,
  inequalityConstraints(NULL),
  numberOfDraws(150) ,
  algoNumberOfDraws(150) ,
  allBetaIter(NULL),
  gnuplotDefined(patFALSE),
  firstGatherGradient(patTRUE),
  ordinalLogitLeftAlternative(patBadId) ,
  numberOfProbaInZf(0),
  allAlternativesAlwaysAvail(patTRUE),
  useModelForGianluca(patFALSE) ,
  syntaxError(NULL),
  betaIterator(NULL),
  scaleIterator(NULL),
  nlNestIterator(NULL),
  cnlNestIterator(NULL),
  cnlAlphaIterator(NULL),
  cnlFullAlphaIterator(NULL),
  networkNestIterator(NULL),
  networkAlphaIterator(NULL)
 {
  mu.defaultValue = 1.0 ;
  mu.estimated = 1.0 ;
  mu.isFixed = patTRUE ;
  mu.lowerBound = 0.0 ;
  mu.upperBound = 1.0 ;
}

patModelSpec::~patModelSpec() {
  DELETE_PTR(choiceExpr) ;
  DELETE_PTR(aggLastExpr) ;
  DELETE_PTR(aggWeightExpr) ;
  DELETE_PTR(weightExpr) ;
  DELETE_PTR(panelExpr) ;
  DELETE_PTR(excludeExpr) ;
  DELETE_PTR(groupExpr) ;
  DELETE_PTR(excludeExpr) ;

  DELETE_PTR(betaIterator) ;
  DELETE_PTR(scaleIterator) ;
  DELETE_PTR(nlNestIterator) ;
  DELETE_PTR(cnlNestIterator) ;
  DELETE_PTR(cnlAlphaIterator) ;
  DELETE_PTR(cnlFullAlphaIterator) ;
  DELETE_PTR(networkNestIterator) ;
  DELETE_PTR(networkAlphaIterator) ;

  for (map<patString,patArithNode*>::iterator i = expressions.begin() ;
       i != expressions.end() ;
       ++i) {
    DELETE_PTR(i->second) ;
  }
  for (map<patString,patArithVariable*>::iterator i = availExpressions.begin() ;
       i != availExpressions.end() ;
       ++i) {
    DELETE_PTR(i->second) ;
  }
  nonZeroAlphasPerAlt.erase(nonZeroAlphasPerAlt.begin(),
			    nonZeroAlphasPerAlt.end()) ;
  nonZeroAlphasPerNest.erase(nonZeroAlphasPerNest.begin(),
			     nonZeroAlphasPerNest.end()) ;
  for (long i = 0 ; i < iterPerNest.size() ; ++i) {
    DELETE_PTR(iterPerNest[i]) ;
  }
  for (long i = 0 ; i < iterPerAlt.size() ; ++i) {
    DELETE_PTR(iterPerAlt[i]) ;
  }
  DELETE_PTR(theNetworkGevModel) ;

  DELETE_PTR(equalityConstraints) ;
  DELETE_PTR(inequalityConstraints) ;
}

patModelSpec* patModelSpec::the() {
  return patBisonSingletonFactory::the()->patModelSpec_the() ;
}

void patModelSpec::readFile(const patString& fileName, 
			    patError*& err) {

  // Parse the model spec file

//   patBoolean python = patFALSE ;
//   if (python) {
//     PyModel thePyModel ;
//     int status = thePyModel.readModel(fileName.c_str()); 
//   }
//   else {

    patSpecParser parser(fileName) ;
    parser.parse(this) ;
//   }

  if (syntaxError != NULL) {
    err = syntaxError ;
    WARNING(err->describe()) ;
    return ;
  }
  //  DEBUG_MESSAGE("Done with parsing") ;
  
  createExpressionForOne() ;

  if (groupExpr == NULL) {
    setDefaultGroup() ;
  }


  compute_altIdToInternalId() ;

  computeIndex(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }


  buildCovariance(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  buildLinearRandomUtilities(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }


//   DEBUG_MESSAGE("getNbrDrawAttributesPerObservation=" << 
// 		getNbrDrawAttributesPerObservation()) ;

}


void patModelSpec::setChoice(patArithNode* choice) {
  choiceExpr = choice ;
}

void patModelSpec::setAggregateLast(patArithNode* aggLast) {
  aggLastExpr = aggLast ;
}

void patModelSpec::setAggregateWeight(patArithNode* aggWeight) {
  aggWeightExpr = aggWeight ;
}

void patModelSpec::setPanel(patArithNode* panel) {
  panelExpr = panel ;
}

void patModelSpec::setWeight(patArithNode* weight) {
  weightExpr = weight ;
}

void patModelSpec::addBeta(const patString& name,
			   patReal defaultValue,
			   patReal lowerBound,
			   patReal upperBound,
			   patBoolean isFixed) {

  patBetaLikeParameter b ;

  if (defaultValue < lowerBound) {
    WARNING("Default value for " << name << " set to " << lowerBound) ;
    defaultValue = lowerBound ;
  }
  if (defaultValue > upperBound) {
    WARNING("Default value for " << name << " set to " << upperBound) ;
    defaultValue = upperBound ;
  }

  b.defaultValue = defaultValue ;
  b.lowerBound = lowerBound ;
  b.upperBound = upperBound ;
  b.isFixed = isFixed ;
  b.name = name ;
  b.estimated = defaultValue ;
  b.index = patBadId ;
  b.hasDiscreteDistribution = patFALSE ;
  betaParam[name] = b ;

}

void patModelSpec::setMu(patReal defaultValue,
			 patReal lowerBound,
			 patReal upperBound,
			 patBoolean isFixed) {
  mu.name = "Mu" ;

  patReal minimumMu = patParameters::the()->getgevMinimumMu() ;
  if (lowerBound < minimumMu) {
    WARNING("Lower bound on mu set to " << minimumMu) ;
    WARNING(" Value defined by gevMinimumMu in " << patFileNames::the()->getParFile()) ;
    lowerBound = minimumMu ;
  }
  if (defaultValue < lowerBound) {
    WARNING("Default value for mu set to " << lowerBound) ;
    defaultValue = lowerBound ;
  }
  if (defaultValue > upperBound) {
    WARNING("Default value for mu set to " << upperBound) ;
    defaultValue = upperBound ;
  }
  mu.defaultValue = 
    mu.estimated = defaultValue ;
  mu.lowerBound = lowerBound ;
  mu.upperBound = upperBound ;
  mu.isFixed = isFixed ;
  mu.index = patBadId ;
}

void patModelSpec::setSampleEnumeration(long s) {
  sampleEnumeration = s ;
}

void patModelSpec::addUtil(unsigned long id, 
			   const patString& name, 
			   const patString& availHeader,
			   const patUtilFunction* function,
			   patError*& err) {
  assert (function != NULL) ;
  patAlternative alter ;
  alter.userId = id ;
  alter.name = name ;
  alter.id = patBadId ;
  alter.utilityFunction = *function ;
  map<patString, patAlternative>::iterator found = utilities.find(name) ;
  if (found != utilities.end()) {
    stringstream str ;
    str << "Alternative " << name << " is defined more than once." ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe());
    return ;
  }
  utilityFormulas[name] = utilities[name] = alter ;
  availName[id] = availHeader ;
  altInternalIdComputed = patFALSE ;
  indexComputed = patFALSE ;

  for (patUtilFunction::const_iterator i = alter.utilityFunction.begin() ;
       i != alter.utilityFunction.end() ;
       ++i) {
    usedAttributes[i->x] = patBadId ;
  }
}

void patModelSpec::setGroup(patArithNode* group) {
  groupExpr = group ;
}

void patModelSpec::setExclude(patArithNode* exclude) {

  for (unsigned long i = 0 ; i < attributeNames.size() ; ++i) {
    exclude->setAttribute(attributeNames[i],i) ;
  } 
  excludeExpr = exclude ;
}

void patModelSpec::setDefaultGroup() {
  patArithConstant* ptr = new patArithConstant(NULL) ;
  ptr->setValue(1) ;
  setGroup(ptr) ;
  addScale(1,
	   1.0,
	   1.0,
	   1.0,
	   patTRUE) ;
}


void patModelSpec::addScale(long groupId,
			    patReal defaultValue,
			    patReal lowerBound,
			    patReal upperBound,
			    patBoolean isFixed) {
  patString name = getScaleNameFromId(groupId) ;
  groupIds.push_back(groupId) ;
  if (groupId > largestGroupUserId) {
    largestGroupUserId = groupId ;
  }
  patBetaLikeParameter dv ;
  if (defaultValue < lowerBound) {
    WARNING("Default value for " << name << " set to " << lowerBound) ;
    defaultValue = lowerBound ;
  }
  if (defaultValue > upperBound) {
    WARNING("Default value for " << name << " set to " << upperBound) ;
    defaultValue = upperBound ;
  }
  dv.defaultValue = dv.estimated = defaultValue ;
  dv.lowerBound = lowerBound ;
  dv.upperBound =upperBound ;
  dv.isFixed = isFixed ;
  dv.name = name ;
  dv.index = patBadId ;
  scaleParam[name] = dv ;
}

void patModelSpec::setModelType(patModelType mt) {
  modelType = mt ;
}

void patModelSpec::addIIATest(patString name, const list<long>* listAltId) {
  iiaTests[name] = *listAltId ;
}

void patModelSpec::addProbaStandardError(patString b1, patString b2, patReal value, patError*& err) {
  pair<patString,patString> key(b1,b2) ;
  map<pair<patString,patString>,patReal>::iterator found = probaStandardErrors.find(key) ;
  if (found != probaStandardErrors.end() && value != found->second) {
    stringstream str ;
    str << "In section [ProbaStandardErrors], the pair (" << b1 << "," << b2 << ") is duplicate with two different values: " << value << " and " << found->second ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  probaStandardErrors[key] = value ;
}


void patModelSpec::addNest(const patString& name,
			   patReal defaultValue,
			   patReal lowerBound,
			   patReal upperBound,
			   patBoolean isFixed,
			   const list<long>* listAltId) {
  assert(listAltId != NULL) ;
  patBetaLikeParameter dv ;
  if (defaultValue < lowerBound) {
    WARNING("Default value for " << name << " set to " << lowerBound) ;
    defaultValue = lowerBound ;
  }
  if (defaultValue > upperBound) {
    WARNING("Default value for " << name << " set to " << upperBound) ;
    defaultValue = upperBound ;
  }
  dv.defaultValue = dv.estimated = defaultValue ;
  dv.lowerBound = lowerBound ;
  dv.upperBound = upperBound ;
  dv.isFixed = isFixed ;
  dv.name = name ;
  dv.index = patBadId ;
  patNlNestDefinition nd ;
  nd.nestCoef = dv ;
  nd.altInNest = *listAltId ;
  nlNestParam[name] = nd ;
}

void patModelSpec::addCNLNest(const patString& name,
			      patReal defaultValue,
			      patReal lowerBound,
			      patReal upperBound,
			      patBoolean isFixed) {

  patBetaLikeParameter dv ;
  if (defaultValue < lowerBound) {
    WARNING("Default value for " << name << " set to " << lowerBound) ;
    defaultValue = lowerBound ;
  }
  if (defaultValue > upperBound) {
    WARNING("Default value for " << name << " set to " << upperBound) ;
    defaultValue = upperBound ;
  }
  dv.defaultValue = 
    dv.estimated = defaultValue ;
  dv.lowerBound = lowerBound ;
  dv.upperBound = upperBound ;
  dv.isFixed = isFixed ;
  dv.name = name ;
  dv.index = patBadId ;

  cnlNestParam[name] = dv ;
}

void patModelSpec::addCovarParam(const patString& param1,
				 const patString& param2,
				 patReal defaultValue,
				 patReal lowerBound,
				 patReal upperBound,
				 patBoolean isFixed) {
  

  if (defaultValue < lowerBound) {
    WARNING("Default value for covar " << buildCovarName(param1,param2) << " set to " << lowerBound) ;
    defaultValue = lowerBound ;
  }
  if (defaultValue > upperBound) {
    WARNING("Default value for covar " << buildCovarName(param1,param2) << " set to " << upperBound) ;
    defaultValue = upperBound ;
  }

  pair<patString,patString> directCompleteName(param1,param2) ;
  
  map<pair<patString,patString>,patBetaLikeParameter*>::iterator found = 
    covarParameters.find(directCompleteName) ;
  if (found != covarParameters.end()) {
    WARNING(buildCovarName(param1,param2) << " already defined. New definition is ignored") ;
    return ;
  }

  pair<patString,patString>  swappedCompleteName(param2,param1) ;
  found =  covarParameters.find(swappedCompleteName) ;
  if (found != covarParameters.end()) {
    WARNING(buildCovarName(param1,param2) << " already defined. New definition is ignored") ;
    return ;
  }
  patString completeName = buildCovarName(param2,param1) ;

  patBetaLikeParameter dv ;
  dv.defaultValue = dv.estimated = defaultValue ;
  dv.lowerBound = lowerBound ;
  dv.upperBound = upperBound ;
  dv.isFixed = isFixed ;
  dv.name = completeName ;
  dv.index = patBadId ;

  betaParam[completeName] = dv ;
  covarParameters[directCompleteName] = &(betaParam[completeName]) ;
}


void patModelSpec::addCNLAlpha(const patString& altName,
			       const patString& nestName,
			       patReal defaultValue,
			       patReal lowerBound,
			       patReal upperBound,
			       patBoolean isFixed) {

  patString completeName = buildAlphaName(altName,nestName) ;
  if (defaultValue < lowerBound) {
    WARNING("Default value for " << completeName << " set to " << lowerBound) ;
    defaultValue = lowerBound ;
  }
  if (defaultValue > upperBound) {
    WARNING("Default value for " << completeName << " set to " << upperBound) ;
    defaultValue = upperBound ;
  }
  patBetaLikeParameter dv ;
  dv.defaultValue = dv.estimated = defaultValue ;
  dv.lowerBound = lowerBound ;
  dv.upperBound = upperBound ;
  dv.isFixed = isFixed ;
  dv.name = completeName ;
  dv.index = patBadId ;

  patCnlAlphaParameter ap ;
  ap.alpha = dv ;
  ap.altName = altName ;
  ap.nestName = nestName ;

  cnlAlphaParam[completeName] = ap ;
}


void patModelSpec::setDataHeader(vector<patString>& header) {
  headers = header ;
}

void patModelSpec::readDataHeader(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }


  //  DEBUG_MESSAGE("Read data headers") ;
  // Empty the data structure

  headers.erase(headers.begin(),headers.end()) ;
  headersPerFile.erase(headersPerFile.begin(),headersPerFile.end()) ;

  unsigned short nbrSampleFiles = patFileNames::the()->getNbrSampleFiles() ;
  headersPerFile.resize(nbrSampleFiles) ;

  vector<unsigned long> apparentColumnData ;

  for (unsigned short fileId = 0 ; fileId < nbrSampleFiles ; ++fileId) {

    unsigned short nHeaders(0) ;
    patString dataFile = patFileNames::the()->getSamFile(fileId,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }

    DETAILED_MESSAGE("Read headers in " << dataFile) ;
    ifstream f(dataFile.c_str()) ;
    if (!f) {
      stringstream str ;
      str << "Cannot open data file " << dataFile ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
    char testChar = '\0' ;
    while (testChar != '\n' && testChar != '\r') {
      do {
	f.get(testChar) ;
      } while (testChar == '\t') ;
      if (testChar != '\n' && testChar != '\r') {
#ifdef SGI_COMPILER
	f.putback(testChar) ;
#else
	f.unget() ;
#endif
	patString header ;
	f >> header ;
	++nHeaders ;
	headersPerFile[fileId].push_back(header) ;
	vector<patString>::iterator found = 
	  find(headers.begin(),headers.end(),header) ;
	
	if (found == headers.end()) {
	  headers.push_back(header) ;
	  addAttribute(header) ;
	}
      }
    }
    DETAILED_MESSAGE(nHeaders << " headers read in " << dataFile) ;
    apparentColumnData.push_back(nHeaders) ;
    //    for (vector<patString>::iterator i = headers.begin() ;
    //         i != headers.end() ;
    //         ++i) {
    //      cout << "[" << *i << "]" ;
    //    }
    //    cout << endl ;
    
    f.close() ;
  }

  DETAILED_MESSAGE("Total number of different headers: " << headers.size()) ;

//   if (apparentColumnData != columnData) {
//     stringstream str ;
//     str << "Consistency check failed for the data files structure" << endl ;
//     if (columnData.empty()) {
//       str << "The section [DataFile] has not been defined in " << patFileNames::the()->getModFile() << endl ;
//     }
//     else {
//       str << "Description in "<<patFileNames::the()->getModFile() <<" file:" << endl ;
//       str << "$COLUMNS = " ;
//       for (vector<unsigned long>::iterator i = columnData.begin() ;
// 	   i != columnData.end() ;
// 	   ++i) {
// 	if (i != columnData.begin()) {
// 	  str << " ; " ;
// 	}
// 	str << *i ;
//       }
//       str << endl ;
//     }
//     str << "Structure read by BIOGEME:" << endl ;
//     str << "$COLUMNS = " ;
//     for (vector<unsigned long>::iterator i = apparentColumnData.begin() ;
// 	 i != apparentColumnData.end() ;
// 	 ++i) {
//       if (i != columnData.begin()) {
// 	str << " ; " ;
//       }
//       str << *i ;
//     }
//     str << endl ;

//     if (apparentColumnData.size() == columnData.size()) {
//       for (unsigned int i = 0 ; i < columnData.size() ; ++i) {
// 	if (apparentColumnData[i] != columnData[i]) {
// 	  patString dataFile = patFileNames::the()->getSamFile(i,err) ;
// 	  if (err != NULL) {
// 	    WARNING(err->describe()) ;
// 	    return ;
// 	  }	    
// 	  str << "File " << dataFile << " apparently has " << apparentColumnData[i] << " columns and not " << columnData[i] << endl ;
// 	  if (apparentColumnData[i] = 2*columnData[i]) {
// 	    str << "There is probably a tab at the end of the first line. Remove it." << endl ;
// 	  }
// 	}
//       }
//     }
//     err = new patErrMiscError(str.str()) ;
//     WARNING(err->describe()) ;
//     return ;
//   }
    
}

patBoolean patModelSpec::isBP() const {
  return modelType == patBPtype ;
}

patBoolean patModelSpec::isOL() const {
  return modelType == patOLtype ;
}

patBoolean patModelSpec::isMNL() const {
  return modelType == patMNLtype ;
}

patBoolean patModelSpec::isNL() const {
  return modelType == patNLtype ;
}

patBoolean patModelSpec::isCNL() const {
  return modelType == patCNLtype ;
}

patBoolean patModelSpec::isNetworkGEV() const {
  return modelType == patNetworkGEVtype ;
}

patBoolean patModelSpec::isMixedLogit() const {
  return (randomParameters.size() > 0) ;
}

patBoolean patModelSpec::isGEV() const {
  return (modelType != patBPtype && modelType != patOLtype) ;
}


 unsigned long patModelSpec::getNbrAlternatives() const {
  return utilities.size() ;
}

 unsigned long patModelSpec::getNbrDataPerRow(unsigned short fileId) const {
  return headersPerFile[fileId].size() ;
}

 unsigned long patModelSpec::getNbrOrigBeta() const {
  return betaParam.size() ;
}

 unsigned long patModelSpec::getNbrTotalBeta() const {
  return betaParam.size() ;
}


 unsigned long patModelSpec::getNbrNests() const {
  switch (modelType) {
  case patBPtype :
    return 0 ;
  case patMNLtype :
    return 0 ;
  case patNLtype :
    return nlNestParam.size() ;
  case patCNLtype :
    return cnlNestParam.size() ;
  case patNetworkGEVtype :
    // Not clear what it should be 
    WARNING("Number of nests is irrelevant for Network GEV models. Should not be called") ;
    return patBadId ;
  default :
    return 0 ;
  }
}

 unsigned long patModelSpec::getNbrAttributes() const {
  return attributeNames.size() ;
}

 unsigned long patModelSpec::getNbrUsedAttributes() const {
  return usedAttributes.size() ;
}

 unsigned long patModelSpec::getNbrRandomParameters() const {
  return randomParameters.size() ;
}



void patModelSpec::assignNLNests(patNL* nestedModel,
			       patError*& err) const {

  static patBoolean xx = patFALSE ;

  if (xx) {
    err = new patErrMiscError("assignNLNests has already been called once") ;
    WARNING(err->describe()) ;
    return ;
  }

  xx = patTRUE ;

  map<unsigned long, patBoolean> checkAlternatives ;

  for (map<patString, patAlternative>::const_iterator i = utilities.begin() ;
       i != utilities.end() ;
       ++i) {
    checkAlternatives[i->second.userId] = patFALSE ;
  }

  if (modelType != patNLtype) {
    err = new patErrMiscError("Not a nested logit model") ;
    WARNING(err->describe()) ;
    return ;
  }
  unsigned long id = 0 ;
  for (map<patString,patNlNestDefinition>::const_iterator i = 
	 nlNestParam.begin() ;
       i != nlNestParam.end() ;
       ++i) {
    nestedModel->addNestName(id++,(*i).first,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    for (list<long>::const_iterator j = (*i).second.altInNest.begin() ;
	 j != (*i).second.altInNest.end() ;
	 ++j) {
      nestedModel->assignAltToNest(*j,(*i).first,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      checkAlternatives[*j] = patTRUE ;
    }
  }
  
  list<unsigned long> unassignedAlt ;
  for (map<unsigned long, patBoolean>::iterator i = checkAlternatives.begin() ;
       i != checkAlternatives.end() ;
       ++i) {
    if (!i->second) {
      unassignedAlt.push_back(i->first) ;
    }
  }
  if (!unassignedAlt.empty()) {
    stringstream str ;
    str << "Alternative(s) not assigned to any nest: " ;
    for (list<unsigned long>::iterator i = unassignedAlt.begin() ;
	 i != unassignedAlt.end() ;
	 ++i) {
      if (i != unassignedAlt.begin()){
	str << "," ;
      }
      str << *i ;
    }
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }

}

patUtilFunction* patModelSpec::getUtil(unsigned long altId,
				      patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  // Check if alternative is present
  
  map<unsigned long,patString>::const_iterator util =
    altIdToName.find(altId) ;
  
  if (util == altIdToName.end()) {
    stringstream  str ;
    str << "Alternatives: " << endl ;
    for (map<unsigned long,patString>::const_iterator i = altIdToName.begin() ;
	 i != altIdToName.end() ;
	 ++i) {
      str << i->first << '\t' << i->second << endl ;
    }
    str << "Alternative " << altId << " not found" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL ;
  }

  return &(utilities[util->second].utilityFunction) ;
}

patUtilFunction* patModelSpec::getUtilFormula(unsigned long altId,
					      patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  // Check if alternative is present
  
  map<unsigned long,patString>::const_iterator util =
    altIdToName.find(altId) ;
  
  if (util == altIdToName.end()) {
    stringstream  str ;
    str << "Alternatives: " << endl ;
    for (map<unsigned long,patString>::const_iterator i = altIdToName.begin() ;
	 i != altIdToName.end() ;
	 ++i) {
      str << i->first << '\t' << i->second << endl ;
    }
    str << "Alternative " << altId << " not found" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL ;
  }

  return &(utilityFormulas[util->second].utilityFunction) ;
}

unsigned long patModelSpec::getBetaIndex(const patString& betaName,
					patBoolean* exists) const {
  
  map<patString,patBetaLikeParameter>::const_iterator found =
    betaParam.find(betaName) ;
  
  if (found == betaParam.end()) {
    *exists = patFALSE ;
    return patBadId ;
  }
  *exists = patTRUE ;

  return found->second.index ;
  
}

patString patModelSpec::getHeaderName(unsigned short fileId,
				      unsigned long rank,
				      patError*& err) {
  if (fileId >= headersPerFile.size()) {
    err = new patErrOutOfRange<unsigned long>(fileId,0,headersPerFile.size()) ;
    WARNING(err->describe()) ;
    return patString();
  }
  if (rank >= headersPerFile[fileId].size()) {
    err = new patErrOutOfRange<unsigned long>(rank,0,headersPerFile[fileId].size()) ;
    WARNING(err->describe()) ;
    return patString();
  }
  return headersPerFile[fileId][rank] ;
}

unsigned long patModelSpec::getHeaderRank(unsigned short fileId,
					  const patString& headerName,
					  patError*& err) {
  
  if (fileId >= headersPerFile.size()) {
    err = new patErrOutOfRange<unsigned long>(fileId,0,headersPerFile.size()) ;
    WARNING(err->describe()) ;
    return unsigned();
  }
  

  vector<patString>::const_iterator headFound =
    find(headersPerFile[fileId].begin(),
	 headersPerFile[fileId].end(),
	 headerName) ;
    if (headFound == headersPerFile[fileId].end()) {
      stringstream str ;
      str << "Header " << headerName << " not defined in this file" ; 
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return unsigned();
    }
    unsigned long result = headFound - headersPerFile[fileId].begin() ;
    return(result) ;
}

patArithNode* patModelSpec::getChoiceExpr(patError*& err) {
  return choiceExpr ;
}

patArithNode* patModelSpec::getAggLastExpr(patError*& err) {
  return aggLastExpr ;
}

patArithNode* patModelSpec::getAggWeightExpr(patError*& err) {
  return aggWeightExpr ;
}

patArithNode* patModelSpec::getWeightExpr(patError*& err) {
  return weightExpr ;
}

patArithNode* patModelSpec::getPanelExpr(patError*& err) {
  return panelExpr ;
}

patArithNode* patModelSpec::getGroupExpr(patError*& err) {
  return groupExpr ;
}

patArithNode* patModelSpec::getExcludeExpr(patError*& err) {
  return excludeExpr ;
}

patArithNode* patModelSpec::getAvailExpr(unsigned long altId,
					 patError*& err) {
  
//   if (fileId >= headersPerFile.size()) {
//     err = new patErrOutOfRange<unsigned long>(fileId,0,headersPerFile.size()) ;
//     WARNING(err->describe()) ;
//     return NULL;
//   }

  patArithNode* result(NULL)  ;

  map<unsigned long, patString>::const_iterator found = availName.find(altId) ;
  if (found == availName.end()) {
    stringstream str ;
    str << "No availability has been defined for alt " << altId ;
    map<unsigned long,patString>::const_iterator debug =
      altIdToName.find(altId) ;
    if (debug != altIdToName.end()) {
      str << ". This alt has been defined as " << debug->first ;
    }
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL ;
  }

  map<patString,patArithNode*>::iterator refound = 
    expressions.find(found->second) ;

  if (refound == expressions.end()) {
    vector<patString>::iterator lastChance = find(headers.begin(),
						  headers.end(),
						  found->second) ;
    if (lastChance == headers.end()) {
      stringstream str ;
      str << "Expression [" << found->second 
	  << "] defining availability for alt. " 
	  << altId << " is not defined." ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return NULL;
    }
    else {
      map<patString,patArithVariable*>::iterator exprFound =
	availExpressions.find(*lastChance) ;
      if (exprFound == availExpressions.end()) {
	patArithVariable* ptr =  new patArithVariable(NULL) ;
	ptr->setName(*lastChance) ;
	availExpressions[*lastChance] = ptr ;
	result = ptr ;
      }
      else {
	result= exprFound->second ;
      }
    }
  }
  else {
    result = refound->second ;
  }

//   patBoolean ok = checkExpressionInFile(result,fileId,err) ;
//   if (err != NULL) {
//     WARNING(err->describe()) ;
//     return NULL ;
//   }

//   if (ok) {
//     return result ;
//   }
//   else {
//     stringstream str ;
//     str << "Expression [" << *result
// 	<< "] defining availability for alt. " 
// 	<< altId << " is not relevant for file " << fileId ;
//     err = new patErrMiscError(str.str()) ;
//     WARNING(err->describe()) ;
//     return NULL ;
//   }

  return result ;
}

patArithNode* patModelSpec::getVariableExpr(const patString& name,
					    patError*& err) {
  
//   DEBUG_MESSAGE("Currently defined expressions") ;
//   for (map<patString,patArithNode*>::iterator ii = expressions.begin() ;
//        ii != expressions.end();
//        ++ii) {
//     DEBUG_MESSAGE(ii->first << "=" << *(ii->second)) ;
//   }

  map<patString,patArithNode*>::iterator found = 
    expressions.find(name) ;
  
  if (found == expressions.end()) {
    vector<patString>::iterator lastChance = find(headers.begin(),
						  headers.end(),
						  name) ;
    if (lastChance == headers.end()) {
      return NULL ;
    }
    else {
	unsigned long ii = getAttributeId(*lastChance) ;
	patArithAttribute* ptr =  new patArithAttribute(NULL) ;
	ptr->setName(*lastChance) ;
	ptr->setId(ii) ;
	expressions[name] = ptr ;
	return ptr ;
    }
  }
  else {
    return found->second ;
  }


}


unsigned long patModelSpec::getAltId(unsigned long altInternalId,
				    patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return unsigned();
  }

  if (altInternalId >= getNbrAlternatives()) {
    stringstream str ;
    str << "Invalid id " << altInternalId << ". There are " 
	<< getNbrAlternatives() << " alternatives." ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return unsigned();
  }

  for (map<unsigned long,unsigned long>::const_iterator i = 
	 altIdToInternalId.begin() ;
       i != altIdToInternalId.end() ;
       ++i) {
    if (i->second == altInternalId) {
      return i->first ;
    }
  }
  
  return patBadId ;
}

unsigned long patModelSpec::getAltInternalId(unsigned long altId,
					    patError*& err) {

  if (err != NULL) {
    WARNING(err->describe()) ;
    return unsigned();
  }
    
  if (!altInternalIdComputed) {
    compute_altIdToInternalId() ;
  }

  map<unsigned long,unsigned long>::const_iterator found =
    altIdToInternalId.find(altId) ;
  
  if (found == altIdToInternalId.end()) {
    if (isOL()) {
      // For the ordinal logit model, the alternative id is irrelevant
      // and 0 is returned.
      return 0 ;
    }
    else {
      stringstream str ;
      str << "Alternative " << altId << " is unknown" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return unsigned();   
    } 
  }

  return found->second ;
  
}

  

void patModelSpec::compute_altIdToInternalId() {
  altIdToInternalId.erase(altIdToInternalId.begin(),
			  altIdToInternalId.end()) ;
  largestAlternativeUserId = 0 ;
  unsigned long id = 0 ;
  for (map<patString, patAlternative >::iterator i = 
	 utilities.begin() ;
       i != utilities.end() ;
       ++i) {
    if (i->second.userId > largestAlternativeUserId) {
      largestAlternativeUserId = i->second.userId ;
    }
    i->second.id = id ;
    altIdToInternalId[i->second.userId] = id ;
    altIdToName[i->second.userId] = i->first ;
    ++id ;
  }

//   for (map<unsigned long,unsigned long>::iterator iter = 
// 	 altIdToInternalId.begin() ;
//        iter != altIdToInternalId.end() ;
//        ++iter) {
//     DEBUG_MESSAGE(iter->first << " <--> " << iter->second) ;
//   }

  altInternalIdComputed = patTRUE ;
}

void patModelSpec::addExpression(const patString& name, 
				 patArithNode* expr,
				 patError*& err) {

  expr->expand(err) ;
  if (err != NULL) {
    WARNING(err->describe());
    return ;
  }
  expressions[name] = expr ;
  userOrderOfExpressions.push_back(name) ;
}



void patModelSpec::printExpressions(patError*& err) {
  cout << "Choice\t" ;
  if (choiceExpr == NULL) {
    cout << "undefined" ;
  }
  else {
    cout << *choiceExpr ;
  }
  cout << endl ;
  cout << "Weight\t" ;
  if (weightExpr == NULL) {
    cout << "undefined" ;
  }
  else {
    cout << *weightExpr ;
  }
  cout << endl ;
  cout << "Panel\t" ;
  if (panelExpr == NULL) {
    cout << "undefined" ;
  }
  else {
    cout << *panelExpr ;
  }
  cout << endl ;
  cout << "Exclude\t" ;
  if (excludeExpr == NULL) {
    cout << "undefined"  ;
  }
  else {
    cout << *excludeExpr ;
  }
  cout << endl ;
  cout << "Group\t" ;
  if (groupExpr == NULL) {
    cout << "undefined" ;
  }
  else {
    cout << *groupExpr ;
  }
  cout << endl ;
  cout << "Aggregate last:\t" ;
  if (aggLastExpr == NULL) {
    cout << "undefined" ;
  }
  else {
    cout << *aggLastExpr ;
  }
  
  cout << endl ;
  cout << "Aggregate weight:\t" ;
  if (aggWeightExpr == NULL) {
    cout << "undefined" ;
  }
  else {
    cout << *aggWeightExpr ;
  }
  
  for (map<patString,patArithNode*>::iterator i = expressions.begin() ;
       i != expressions.end() ;
       ++i) {
    cout << i->first << '\t' ;
    if (i->second == NULL) {
      cout << "undefined" << endl ;
    }
    else {
      cout << i->second->getExpression(err) << endl ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
  }
}

patBetaLikeParameter patModelSpec::getBeta(const patString& name,
					   patBoolean* betaFound) const  {
  map<patString,patBetaLikeParameter>::const_iterator found = 
    betaParam.find(name) ;
  if (found == betaParam.end()) {
    //	DEBUG_MESSAGE("Beta parameter " << name << " not found") ;
    *betaFound = patFALSE ;
    return patBetaLikeParameter() ;
  }
  *betaFound = patTRUE ;
  return found->second ;
}



patBetaLikeParameter patModelSpec::getScale(unsigned long id,
				patError*& err) const {

  patString name = getScaleNameFromId(id) ;
  map<patString,patBetaLikeParameter>::const_iterator found = 
    scaleParam.find(name) ;
  if (found == scaleParam.end()) {
    stringstream str ;
    str << "Scale parameter "<< name <<" for group " << id 
	<< " not found" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patBetaLikeParameter() ;
  }
  return found->second ;
  
}

patBetaLikeParameter patModelSpec::getScaleFromInternalId(unsigned long id,
							  patError*& err) const {

  for (map<patString,patBetaLikeParameter>::const_iterator i =
	 scaleParam.begin() ; 
       i != scaleParam.end() ;
       ++i) {
    if (i->second.id == id) {
      return i->second ;
    }
  }
  stringstream str ;
  str << "Scale parameter with ID "<< id 
      << " not found" ;
  err = new patErrMiscError(str.str()) ;
  WARNING(err->describe()) ;
  return patBetaLikeParameter() ;
}

patBetaLikeParameter patModelSpec::getMu(patError*& err) const {
  return mu ;
}

patBetaLikeParameter patModelSpec::getNlNest(const patString& name,
					     patBoolean* nlFound) const{
  
  map<patString,patNlNestDefinition>::const_iterator found = 
    nlNestParam.find(name) ;
  if (found == nlNestParam.end()) {
    *nlFound = patFALSE ;
    return patBetaLikeParameter() ;
  }
  *nlFound = patTRUE ;
  return found->second.nestCoef ;
}

patBetaLikeParameter
patModelSpec::getCnlNest(const patString& name,
			 patBoolean* cnlFound) const {

  map<patString,patBetaLikeParameter>::const_iterator found = 
    cnlNestParam.find(name) ;
  if (found == cnlNestParam.end()) {
    *cnlFound = patFALSE ;
    return patBetaLikeParameter() ;
  }
  *cnlFound = patTRUE ;
  return found->second ;
}

patBetaLikeParameter
patModelSpec::getCnlAlpha(const patString& nestName,
			  const patString& altName,
			  patBoolean* cnlFound) const {
  patString completeName = buildAlphaName(altName,nestName) ;
  map<patString,patCnlAlphaParameter>::const_iterator found = 
    cnlAlphaParam.find(completeName) ;
  if (found == cnlAlphaParam.end()) {
    *cnlFound = patFALSE ;
    return patBetaLikeParameter() ;
  }
  *cnlFound = patTRUE ;
  return found->second.alpha ;

}

patBetaLikeParameter
patModelSpec::getNetworkNode(const patString& name,
			     patBoolean* ngevFound) const {

  map<patString,patBetaLikeParameter>::const_iterator found = 
    networkGevNodes.find(name) ;
  if (found == networkGevNodes.end()) {
    WARNING("Network GEV node " << name << " not found") ;
    *ngevFound = patFALSE ;
    return patBetaLikeParameter() ;
  }
  *ngevFound = patTRUE ;
  return found->second ;
}

patBetaLikeParameter
patModelSpec::getNetworkLink(const patString& name,
			     patBoolean* ngevFound) const {

  map<patString,patNetworkGevLinkParameter>::const_iterator found = 
    networkGevLinks.find(name) ;
  if (found == networkGevLinks.end()) {
    *ngevFound = patFALSE ;
    return patBetaLikeParameter() ;
  }
  *ngevFound = patTRUE ;
  return found->second.alpha ;
}

patBoolean patModelSpec::doesAltExists(unsigned long altUserId) {
  map<unsigned long,patString>::iterator found =
    altIdToName.find(altUserId) ;

  return (found != altIdToName.end()) ;

}
patString patModelSpec::getAltName(unsigned long altId,
				      patError*& err) {

  map<unsigned long,patString>::iterator found =
    altIdToName.find(altId) ;

  if (found == altIdToName.end()) {
    return patString("BIOGEME__UnknownAlt") ;
  }

  return found->second ;
  
}


patString patModelSpec::getAvailName(unsigned long altId,
				     patError*& err) {
  map<unsigned long, patString>::iterator found =
    availName.find(altId) ;

  if (found == availName.end()) {
    stringstream str ;
    str << "Alternative " << altId << " not found" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patString("Unknown") ;
  }

  return found->second ;
  
}


patBoolean patModelSpec::isHeader(patString name) const {
    vector<patString>::const_iterator headFound =
      find(headers.begin(),headers.end(),name) ;
    return (headFound != headers.end()) ;
}


void patModelSpec::addAttribute(const patString& attrName) {
  vector<patString>::iterator found = find(attributeNames.begin(),
				     attributeNames.end(),
				     attrName) ;
  if (found == attributeNames.end()) {
    attribIdPerName[attrName] = attributeNames.size() ;
    attributeNames.push_back(attrName) ;
  }
}

patString patModelSpec::modelTypeName() const {
  static patBoolean first = patTRUE ;
  static map<patModelType,patString> names ;
  if (first) {
    names[patOLtype] = "Ordinal Logit" ;
    names[patBPtype] = "Binary Probit" ;
    names[patMNLtype] = "Logit" ;
    names[patNLtype] = "Nested Logit" ;
    names[patCNLtype] = "Cross-Nested Logit" ;
    names[patNetworkGEVtype] = "Network GEV model" ;
  }
  patString res = names[modelType] ;
  if (isPanelData()) {
    res += " for panel data" ;
  }
  return res ;
}


ostream& operator<<(ostream &str, const patModelSpec& x) {

  str << "Spec File            : " << patFileNames::the()->getModFile() 
      << endl ; 
  str << "Data Files           : " ;
  unsigned short nbrSampleFiles = patFileNames::the()->getNbrSampleFiles() ;
  for (unsigned short fileId = 0 ; fileId < nbrSampleFiles ; ++fileId) {

    patError* err = NULL;
    patString fileName = patFileNames::the()->getSamFile(fileId,err) ;
    if (fileId != 0) {
      str << "                       " ;
    }
    str << fileName << endl ;
  }  
  str << "Parameter file       : " << patFileNames::the()->getParFile() 
      << endl ; 
  if (x.choiceExpr == NULL) {
    str << "Choice               : $NONE" << endl ;
  }
  else {
    str << "Choice               : " << *x.choiceExpr << endl ;
  }
  if (x.weightExpr == NULL) {
    str << "Weight               : $NONE" << endl ;
  }
  else {
    str << "Weight               : " << *x.weightExpr << endl ;
  }
  if (x.panelExpr == NULL) {
    str << "Panel                : $NONE" << endl ;
  }
  else {
    str << "Panel                : " << *x.panelExpr << endl ;
  }
  if (x.excludeExpr == NULL) {
    str << "Exclude              : $NONE" << endl ;
  }
  else {
    str << "Exclude              : " << *x.excludeExpr << endl ;
  }
  if (x.groupExpr == NULL) {
    str << "Group                : $NONE" << endl ;
  }
  else {
    str << "Group                : " << *x.groupExpr << endl ;
  }
  str << "Model                : " << x.modelTypeName() << endl ;

  if (x.aggLastExpr == NULL) {
    str << "Aggregate last      : $NONE" << endl ;
  }
  else {
    str << "Aggregate last      : " << x.aggLastExpr << endl ;
  }
  if (x.aggWeightExpr == NULL) {
    str << "Aggregate weight     : $NONE" << endl ;
  }
  else {
    str << "Aggregate weight     : " << x.aggWeightExpr << endl ;
  }
  str << "Headers" << endl ;
  str << "~~~~~~~" << endl ;
  for (unsigned long i = 0 ; i < x.headers.size() ; ++i) {
    str << "   " << i << " " << x.headers[i] << endl ;
  }
  str << "Expressions" << endl ;
  str << "~~~~~~~~~~~" << endl ;
  for (map<patString,patArithNode*>::const_iterator i = x.expressions.begin() ;
       i != x.expressions.end() ;
       ++i) {
    str << i->first << ": " ;
    if (i->second == NULL) {
      str << "$NONE" << endl ;
    }
    else {
      str << *(i->second) << endl ;
    }
  }

  str << "Beta" << endl ;
  str << "~~~~" << endl ;
  for (map<patString,patBetaLikeParameter>::const_iterator i =
	 x.betaParam.begin() ;
       i != x.betaParam.end() ;
       ++i) {
    str << i->first << '\t' << i->second << endl ; 
  }

  str << "Attributes names" << endl ;
  str << "~~~~~~~~~~~~~~~~" << endl ;
  for (unsigned long i = 0 ;
       i < x.attributeNames.size() ;
       ++i) {
    str << i << " " << x.attributeNames[i] << endl ;
  }
  
  str << "Alternatives external/internal ID" << endl ;
  str << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl ;
  str << "Ext -> Int" << endl ;
  for (map<unsigned long,unsigned long>::const_iterator i = 
	 x.altIdToInternalId.begin() ;
       i != x.altIdToInternalId.end() ;
       ++i) {
    str << i->first << " -> " << i->second << endl ;
  }
  
  str << "Scale" << endl ;
  str << "~~~~~" << endl ;
  for (map<patString,patBetaLikeParameter>::const_iterator i = 
	 x.scaleParam.begin() ;
       i != x.scaleParam.end() ;
       ++i) {
    str << i->first << '\t' << i->second << endl ; 
  }

  str << "Mu" << endl ;
  str << "~~" << endl ;
  str << x.mu << endl ;


  str << "Scale" << endl ;
  str << "~~~~~" << endl ;
  for (map<patString,patBetaLikeParameter>::const_iterator i = 
	 x.scaleParam.begin() ;
       i != x.scaleParam.end() ;
       ++i) {
    str << i->first << '\t' << i->second << endl ; 
  }


  str << "Sample enumeration: " <<x.sampleEnumeration << endl ;
  str << endl ;
  str << "~~~~~~~~~~~~~" << endl;

  str << "Utility functions" << endl ;
  str << "~~~~~~~~~~~~~~~~~" << endl ;
  for (map<patString, patAlternative>::const_iterator i =
	 x.utilities.begin() ;
       i != x.utilities.end() ;
       ++i) {
    str << i->first << " [" << i->second.userId << "] " 
	<< i->second.utilityFunction << endl ;
  }
  str << "Nonlinear utilities" << endl ;
  str << "~~~~~~~~~~~~~~~~~~~" << endl ;
  for (map<unsigned long, patArithNode*>::const_iterator i =
	x.nonLinearUtilities.begin() ;
	i != x.nonLinearUtilities.end() ;
	++i) {
    str << i->first << ": " << *i->second  << endl ;
  }
  str << "Availability" << endl ;
  str << "~~~~~~~~~~~~" << endl ;
  for (map<unsigned long,patString>::const_iterator i = x.availName.begin() ;
       i != x.availName.end() ;
       ++i) {
    str << i->first << "->" << i->second << endl ;
  }

  str << "NL Nests" << endl ;
  str << "~~~~~~~~" << endl ;
  for (map<patString,patNlNestDefinition>::const_iterator i = 
	 x.nlNestParam.begin() ;
       i != x.nlNestParam.end() ;
       ++i) {
    str << i->first << '\t' << i->second.nestCoef << endl ; 
    str << "\tAlternatives: " ;
    for (list<long>::const_iterator j = i->second.altInNest.begin() ;
	 j != i->second.altInNest.end() ;
	 ++j) {
      if (j != i->second.altInNest.begin()) {
	str << "," ;
      }
      str << *j ;
    }
    str << endl ;
  }

  if (x.isCNL()) {
    str << "Nests parameters for the cross-nested logit" << endl ;
    str << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl ;
    
    for (map<patString,patBetaLikeParameter>::const_iterator i = 
	   x.cnlNestParam.begin() ;
	 i != x.cnlNestParam.end() ;
	 ++i) {
      str << i->first << '\t' << i->second << endl ; 
    }
    
    
    
    str << "Alpha parameters for the cross-nested logit" << endl ;
    str << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl ;
    for (map<patString,patCnlAlphaParameter>::const_iterator i = 
	   x.cnlAlphaParam.begin() ;
	 i != x.cnlAlphaParam.end() ;
	 ++i) {
      str << i->first << '\t' << " [" << i->second.altName << "," 
	  << i->second.nestName << "]" << i->second.alpha << endl ; 
    }
    
    str << "non Zero alphas per nest" << endl ;
    for (unsigned long i = 0 ; i < x.cnlNestParam.size() ; ++i) {
      str << "Nest " << i << ":" ;
      for (unsigned long j = 0 ; j < x.nonZeroAlphasPerNest[i].size() ; ++j) {
	str << " " << x.nonZeroAlphasPerNest[i][j] ;
      }
      str << endl ;
    }
    
    str << "non Zero alphas per alt" << endl ;
    for (unsigned long i = 0 ; i < x.getNbrAlternatives() ; ++i) {
      str << "Alt " << i << ":" ;
      for (unsigned long j = 0 ; j < x.nonZeroAlphasPerAlt[i].size() ; ++j) {
	str << " " << x.nonZeroAlphasPerAlt[i][j] ;
      }
      str << endl ;
    }
    str << "cumulative indices:" << endl ;
    for (unsigned long i = 0 ; i < x.cnlNestParam.size() ; ++i) {
      
      str << "Nest " << i ;
      if ( x.cumulIndex[i+1]-1 < x.cumulIndex[i] ) {
	str << " ---" << endl ;
      } else {
	str << " from " << x.cumulIndex[i] 
	    << " to " << x.cumulIndex[i+1]-1 << endl ;
      }
    }
    str << endl ;
  }
  str << "Free parameters" << endl ;
  for (unsigned long i = 0 ;
       i < x.nonFixedParameters.size() ;
       ++i) {
    str << "x[" << i << "]=" ;
    if (x.nonFixedParameters[i] == NULL) {
      str << "undefined" << endl ;
    }
    else {
      str << *(x.nonFixedParameters[i]) << endl ;
    }
  }

  if (x.isCNL()) {
    str << "Alpha index for CNL" << endl ;
    str << "*******************" << endl ;
    str << "Nest\tAlt\tAlphaIndex" << endl ;
    for (unsigned long nest = 0 ; nest < x.cnlNestParam.size() ; ++nest) {
      for (unsigned long alt = 0 ; alt < x.getNbrAlternatives() ; ++alt) {
	str << nest << '\t'
	    << alt << '\t' 
	    << x.getIdCnlAlpha(nest,alt) << endl ;
      }
    }
  }

  str << "Network GEV nodes" << endl ;
  str << "*****************" << endl ;
  for (map<patString, patBetaLikeParameter>::const_iterator i = 
	 x.networkGevNodes.begin();
       i != x.networkGevNodes.end() ;
       ++i) {
    str << i->first << '\t' << i->second << endl ; 
  }
  str << endl ;
  str << "Network GEV nodes" << endl ;
  str << "*****************" << endl ;
  for (map<patString, patNetworkGevLinkParameter>::const_iterator i = 
	 x.networkGevLinks.begin() ;
       i != x.networkGevLinks.end() ;
       ++i) {
    str << i->first << '\t'
	<< i->second.aNode << "->" << i->second.bNode << '\t'  
	<< i->second.alpha << endl ;
  }
  str << endl ;

  str << "RandomParameters" << endl ;
  str << "~~~~~~~~~~~~~~~~" << endl ;
  for (map<patString,pair<patRandomParameter,patArithRandom*> >::const_iterator i = 
	 x.randomParameters.begin() ;
       i != x.randomParameters.end() ;
       ++i) {
    str << i->first << '\t' 
	<< i->second.first << endl ;
  }
  str << "RandomExpressions" << endl ;
  str << "~~~~~~~~~~~~~~~~~" << endl ;
  for (map<patString,pair<patRandomParameter,patArithRandom*> >::const_iterator i = 
	 x.randomParameters.begin() ;
       i != x.randomParameters.end() ;
       ++i) {
    str << i->first << '\t' 
	<< *(i->second.second) << endl ;
  }
  str << "CovarParameters" << endl ;
  str << "~~~~~~~~~~~~~~~" << endl ;
  for (map<pair<patString,patString>,patBetaLikeParameter*>::const_iterator i = 
	 x.covarParameters.begin() ;
       i != x.covarParameters.end() ;
       ++i) {
    str << x.buildCovarName(i->first.first,i->first.second) << '\t'
	<< i->second << endl ;
  }
  return str ;
}

ostream& operator<<(ostream &str, const patUtilFunction& x) {
  
  for (list<patUtilTerm>::const_iterator i = x.begin() ;
       i != x.end() ;
       ++i) {
    if (i != x.begin()) {
      str << " + " ;
    }
    str << i->beta << "*" << i->x ;
  }

  return str ;
}

patString patModelSpec::buildAlphaName(const patString& altName, 
				       const patString& nestName) const  {
  return (nestName + patString("_") + altName) ;
}

patString patModelSpec::buildLinkName(const patString& aNode, 
				       const patString& bNode) const  {
  return (aNode + patString("_") + bNode) ;
}

patString patModelSpec::buildRandomName(const patString& meanName, 
					const patString& stdDevName) const {
  return (meanName + patString("_") + stdDevName) ;
}

patString patModelSpec::buildCovarName(const patString& name1, 
				       const patString& name2) const {
  return (name1 + patString("_") + name2) ;
}

pair<patString,patString> 
patModelSpec::getCnlAlphaAltNest(const patString& name,
				 patBoolean* cnlFound) const {


  map<patString,patCnlAlphaParameter>::const_iterator found =
    cnlAlphaParam.find(name) ;
  if (found == cnlAlphaParam.end()) {
    *cnlFound = patFALSE ;
    return pair<patString,patString>() ;
  }
  *cnlFound = patTRUE ;
  return pair<patString,patString>(found->second.altName,
				   found->second.nestName) ;
}

void patModelSpec::printVarExpressions(ostream& str,patBoolean forPython,patError*& err) {

  patBoolean empty = patTRUE ;
  for (vector<patString>::iterator ii = userOrderOfExpressions.begin() ;
       ii != userOrderOfExpressions.end() ;
       ++ii) {
    patString name = *ii ;
    patArithNode* theExpr = getVariableExpr(name,err) ;

    patString left = name + " ";
    patString right ;
    if (forPython) {
      right = "DefineVariable('" + 
	name + 
	"'," + 
	theExpr->getExpression(err) 
	+ ")" ;
    }
    else {
      right = theExpr->getExpression(err) ;
    }
    if (left != right) {
      str << left << " = " << right << endl ; 
      empty = patFALSE ;
    }
  }
  if (empty) {
    str << "$NONE" << endl ;
  }
}

void patModelSpec::computeIndex(patError*& err) {

  //  DEBUG_MESSAGE("COMPUTE INDEX");

  // index corresponds to the index in the vector of non fixed parameters to
  // be given to the optimization algorithm

  // id corresponds to the index in the betaParameter, scaleParameter and
  // modelParameter vectors respectively.

  // First check that the expressions are not auro-referencing

  for (map<patString,patArithNode*>::iterator iterExpr = expressions.begin() ;
       iterExpr != expressions.end() ;
       ++iterExpr) {
    vector<patString> thisList ;
    iterExpr->second->expand(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    iterExpr->second->getLiterals(&thisList,NULL,patFALSE,err) ;
    vector<patString>::iterator found = find(thisList.begin(), thisList.end(),iterExpr->first) ;
    if (found != thisList.end()) {
      // The headers can be found on both side of the expression
      // without causing a recursivity problem. If it is not a header,
      // there is a recursivity problem.
      vector<patString>::iterator headFound = 
	find(headers.begin(),headers.end(),*found) ;
      if (headFound == headers.end()) {
	stringstream str ;
	str << "Expression " << iterExpr->first << "=" << *iterExpr->second << " contains a recursive statement." ;
	str << "List of literals: " ;
	for (vector<patString>::iterator iterLit = thisList.begin() ;
	     iterLit != thisList.end() ;
	     ++iterLit) {
	  str << *iterLit << " " ;
	}
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return ;
      }
    }
  }

  // Second, store the attributes in the nonlinear utilities

  vector<patString> listOfLiterals ;
  
  if (choiceExpr != NULL) {
    choiceExpr->getLiterals(&listOfLiterals,
			    NULL,
			    patFALSE,
			    err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }

  if (aggLastExpr != NULL) {
    aggLastExpr->getLiterals(&listOfLiterals,
			     NULL,
			     patFALSE,
			     err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }

  if (aggWeightExpr != NULL) {
    aggWeightExpr->getLiterals(&listOfLiterals,
			       NULL,
			       patFALSE,
			       err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }


  if (weightExpr != NULL) {
    weightExpr->getLiterals(&listOfLiterals,
			    NULL,
			    patFALSE,
			    err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }

  if (panelExpr != NULL) {
    panelExpr->getLiterals(&listOfLiterals,
			   NULL,
			   patFALSE,
			   err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }

  if (excludeExpr != NULL) {
    excludeExpr->getLiterals(&listOfLiterals,
			     NULL,
			     patFALSE,
			     err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }

  if (groupExpr != NULL) {
    groupExpr->getLiterals(&listOfLiterals,
			   NULL,
			   patFALSE,
			   err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }

  for (map<patString,patArithVariable*>::iterator i = availExpressions.begin() ;
       i != availExpressions.end() ;
       ++i) {
    i->second->getLiterals(&listOfLiterals,
			   NULL,
			   patFALSE,
			   err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }

  for (vector<patDiscreteParameter>::iterator i = discreteParameters.begin() ;
       i != discreteParameters.end() ;
       ++i) {
    map<patString,patBetaLikeParameter>::iterator found = 
      betaParam.find(i->name) ;
    i->theParameter = &(found->second) ;
  }
  
  for (map<unsigned long, patArithNode*>::iterator i = nonLinearUtilities.begin() ;
       i != nonLinearUtilities.end() ;
       ++i) {
    
    i->second->getLiterals(&listOfLiterals,
			   NULL,
			   patFALSE,
			   err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }

  for (vector<patOneZhengFosgerau>::iterator i = zhengFosgerau.begin() ;
       i != zhengFosgerau.end() ;
       ++i) {
    if (!i->isProbability()) {
      i->expression->getLiterals(&listOfLiterals,
				 NULL,
				 patFALSE,
				 err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      
    }
  }

  for (vector<patString>::iterator ii = listOfLiterals.begin() ;
       ii != listOfLiterals.end() ;
       ++ii) {
    
    patBoolean success ;
    getIndexFromName(*ii,&success) ;
    if (!success) {
      // The literals is not a parameter to be estimated
      // Therefore, we define it as an attribute
      // We first check if it is already
      // considered as an attribute
      unsigned long attrId = getAttributeId(*ii) ;
      if (attrId == patBadId) {
	// If not, we add it.
	addAttribute(*ii) ;
      }
      usedAttributes[*ii] = patBadId ;
    }
  }
  

  // If mixed logit is used, the attribute "one" is used

  if (isMixedLogit()) {
    usedAttributes[patParameters::the()->getgevOne()] = patBadId ;
  }

  // Compute the indices of used attributes

  unsigned long i = 0 ;
  for (map<patString,unsigned long>::iterator j = usedAttributes.begin() ;
       j != usedAttributes.end() ;
       ++j) {
    j->second = i ;
    ++i ;
  }

  // Assign the computed indices in the nonlinear utility functions

  for (map<patString,unsigned long>::iterator ii = usedAttributes.begin() ;
       ii != usedAttributes.end() ;
       ++ii) {
    for (map<unsigned long, patArithNode*>::iterator i = nonLinearUtilities.begin() ;
	 i != nonLinearUtilities.end() ;
	 ++i) {
      
      i->second->setAttribute(ii->first,ii->second) ;
    }
    
    for (vector<patOneZhengFosgerau>::iterator i = zhengFosgerau.begin() ;
	 i != zhengFosgerau.end() ;
	 ++i) {
      if (!i->isProbability()) {
	i->expression->setAttribute(ii->first,ii->second) ;
      }
    }


    if (choiceExpr != NULL) {
      choiceExpr->setAttribute(ii->first,ii->second) ;
    }
    if (aggLastExpr != NULL) {
      aggLastExpr->setAttribute(ii->first,ii->second) ;
    }
    if (aggWeightExpr != NULL) {
      aggWeightExpr->setAttribute(ii->first,ii->second) ;
    }
    if (weightExpr != NULL) {
      weightExpr->setAttribute(ii->first,ii->second) ;
    }
    if (panelExpr != NULL) {
      panelExpr->setAttribute(ii->first,ii->second) ;
    }
    if (excludeExpr != NULL) {
      excludeExpr->setAttribute(ii->first,ii->second) ;
    }
    if (groupExpr != NULL) {
      groupExpr->setAttribute(ii->first,ii->second) ;
    }
    for (map<patString,patArithVariable*>::iterator i = availExpressions.begin() ;
	 i != availExpressions.end() ;
	 ++i) {
      i->second->setAttribute(ii->first,ii->second) ;
    }
  }


  


  if (isCNL()) {
    nonZeroAlphasPerAlt.resize(getNbrAlternatives()) ;
    iterPerAlt.resize(getNbrAlternatives(),NULL) ;
    nonZeroAlphasPerNest.resize(getNbrNests()) ;
    iterPerNest.resize(getNbrNests(),NULL) ;
    cumulIndex.resize(getNbrNests()+1) ;
  }

  unsigned long index = 0 ;
  unsigned long id = 0 ;

  //  DEBUG_MESSAGE("Set beta params") ;

  for (map<patString,patBetaLikeParameter>::iterator i = betaParam.begin() ;
       i != betaParam.end() ;
       ++i) {
    i->second.id = id++ ;
    if (i->second.isFixed) {
      patValueVariables::the()->setValue(i->second.name,i->second.defaultValue) ;

    }
    else if (!i->second.hasDiscreteDistribution) {
      i->second.index = index ;
      //      DEBUG_MESSAGE(i->second.name << " has index " << index) ;
      nonFixedParameters.push_back(&(i->second)) ;
      ++index ;
    }

  }
  nonFixedBetas = nonFixedParameters.size() ;
  id = 0 ;
  for (map<patString,patBetaLikeParameter>::iterator i = scaleParam.begin() ;
       i != scaleParam.end() ;
       ++i) {
    i->second.id = id++ ;
    if (!i->second.isFixed) {
      i->second.index = index ;
      nonFixedParameters.push_back(&(i->second)) ;
      ++index ;
    }
    else {
      i->second.index = patBadId ;
      //      DEBUG_MESSAGE("Set " << i->second.name << "=" << i->second.defaultValue) ;
      patValueVariables::the()->setValue(i->second.name,i->second.defaultValue) ;
    }
  }
  
  if (!mu.isFixed) {
    mu.index = index ;
    nonFixedParameters.push_back(&mu) ;
    ++index ;
  }
  else {
    //      DEBUG_MESSAGE("Set mu=" << mu.defaultValue) ;
    patValueVariables::the()->setValue(patString("MU"),mu.defaultValue) ;
    
  }

  //  DEBUG_MESSAGE("Set NL...") ;

  if (isNL()) {
    id = 0 ;
    for (map<patString,patNlNestDefinition>::iterator i = nlNestParam.begin() ;
	 i != nlNestParam.end() ;
	 ++i) {
      i->second.nestCoef.id = id++ ;
      if (!i->second.nestCoef.isFixed) {
	i->second.nestCoef.index = index ;
	nonFixedParameters.push_back(&(i->second.nestCoef)) ;
	++index ;
      }
      else {
	patValueVariables::the()->setValue(i->second.nestCoef.name,
					   i->second.nestCoef.defaultValue) ;
      }
    }
  }

  //  DEBUG_MESSAGE("Set CNL...") ;

  if (isCNL()) {

    //DEBUG_MESSAGE("YES, set CNL...") ;
    id = 0 ;
    for (map<patString,patBetaLikeParameter>::iterator i = 
	   cnlNestParam.begin() ;
	 i != cnlNestParam.end() ;
	 ++i) {
      i->second.id = id++ ;
      if (!i->second.isFixed) {
	i->second.index = index ;
	nonFixedParameters.push_back(&(i->second)) ;
	++index ;
      }
      else {
	DEBUG_MESSAGE("Set = " << i->second.name << "=" << i->second.defaultValue) ;
	patValueVariables::the()->setValue(i->second.name,i->second.defaultValue) ;
      }
    }

    DEBUG_MESSAGE("Set alphas CNL") ;

    // ptr must contain the id  of the first non fixed alpha parameter
    unsigned long ptr = id ;
    for (map<patString,patCnlAlphaParameter>::iterator i = 
	   cnlAlphaParam.begin() ;
	 i != cnlAlphaParam.end() ;
	 ++i) {
      unsigned long nest = cnlNestParam[i->second.nestName].id ;
      unsigned long alt = utilities[i->second.altName].id ;      
      
      i->second.alpha.id = id++ ;

      if (alt >= nonZeroAlphasPerAlt.size()) {
	WARNING("Error in processing " << i->first) ;
	WARNING("Check the order of the names in section [CNLAlpha] of " << patFileNames::the()->getModFile()) ;
	WARNING("--> Make sure the name of the alternative is followed by nest name") ;
	err = new patErrOutOfRange<unsigned long>(alt,0,nonFixedParameters.size()-1) ;
	WARNING(err->describe()) ;
	return ;
      }
      nonZeroAlphasPerAlt[alt].push_back(nest) ;
      if (nest >= nonZeroAlphasPerNest.size()) {
	WARNING("Error in processing " << i->first) ;
	WARNING("Check the order of the names in section [CNLAlpha] of " << patFileNames::the()->getModFile()) ;
	WARNING("--> Make sure the name of the alternative is followed by nest name") ;
	err = new patErrOutOfRange<unsigned long>(nest,0,nonZeroAlphasPerNest.size()-1) ;
	WARNING(err->describe()) ;
	return ;
	
      }
      nonZeroAlphasPerNest[nest].push_back(alt) ;
      if (!i->second.alpha.isFixed) {
	i->second.alpha.index = index ;
	nonFixedParameters.push_back(&(i->second.alpha)) ;
	++index ;
      }
      else {

	patValueVariables::the()->setValue(i->second.alpha.name,
					   i->second.alpha.defaultValue) ;
      }
    }

    // This index computation assumes that the cnlAlphaParam are sorted first
    // by nests and then by alternatives.

    cumulIndex[0] = ptr ;
    for (unsigned long i = 0 ; i < getNbrNests() ; ++i) {
      ptr += nonZeroAlphasPerNest[i].size() ;
      cumulIndex[i+1] = ptr ;
    }

  }

  if (isNetworkGEV()) {
    // First, add nodes corresponding to alternatives, if not explicitly listed
    
    for (map<patString, patAlternative>::iterator i = utilities.begin() ;
	 i != utilities.end() ;
	 ++i) {
      map<patString, patBetaLikeParameter>::iterator found = 
	networkGevNodes.find(i->first) ;
      if (found == networkGevNodes.end()) {
	//	DEBUG_MESSAGE("Default value for node " << i->first) ;
	patBetaLikeParameter def ;
	def.name = i->first ;
	def.defaultValue = 1.0 ;
	def.lowerBound = 1.0 ;
	def.upperBound = 1.0 ;
	def.isFixed = patTRUE ;
	def.estimated = 1.0 ;
	def.index = patBadId ;
	def.id = patBadId ;
	networkGevNodes[i->first] = def ;
      }
    }

    id = 0 ;
    for (map<patString,patBetaLikeParameter>::iterator i = 
	   networkGevNodes.begin() ;
	 i != networkGevNodes.end() ;
	 ++i) {

      // The mu parameter of the root is not added.
      if (i->second.name != patModelSpec::the()->rootNodeName) {
	//	DEBUG_MESSAGE(i->second.name << " NGEV param " << id) ;
	i->second.id = id++ ;
	if (!i->second.isFixed) {
	  i->second.index = index ;
	  nonFixedParameters.push_back(&(i->second)) ;
	  ++index ;
	}
	else {
	  //	  DEBUG_MESSAGE("Set " << i->second.name << "=" << i->second.defaultValue) ;
	  patValueVariables::the()->setValue(i->second.name,i->second.defaultValue) ;
	}
      }
    }

    for (map<patString,patNetworkGevLinkParameter>::iterator i = 
	   networkGevLinks.begin() ;
	 i != networkGevLinks.end() ;
	 ++i) {
      //      DEBUG_MESSAGE(i->second.alpha.name << " NGEV param " << id) ;
      i->second.alpha.id = id++ ;
      if (!i->second.alpha.isFixed) {
	i->second.alpha.index = index ;
	nonFixedParameters.push_back(&(i->second.alpha)) ;
	++index ;
      }
      else {
	// 	DEBUG_MESSAGE("Set " << i->second.alpha.name << "="
	// 		      << i->second.alpha.defaultValue) ;
	patValueVariables::the()->setValue(i->second.alpha.name,
					   i->second.alpha.defaultValue) ;
      }
    }

  }

  betaParameters.resize(betaParam.size()) ;


  if (isNL()) {
    modelParameters.resize(nlNestParam.size()) ;
  }
  if (isCNL()) {
    //    DEBUG_MESSAGE("CNL model with " << cnlNestParam.size() << " nest and " << cnlAlphaParam.size() << " alphas for a total of " << cnlNestParam.size() + cnlAlphaParam.size() << " parameters") ;
    modelParameters.resize(cnlNestParam.size() + cnlAlphaParam.size()) ;
  }
  if (isNetworkGEV()) {
    modelParameters.resize(networkGevNodes.size() + networkGevLinks.size()) ;
  }
  scaleParameters.resize(scaleParam.size()) ;

  if (equalityConstraints != NULL) {
    //     DEBUG_MESSAGE("Define variables for the " << equalityConstraints->size() 
    // 		  << " constraints") ;
    
    for (patListNonLinearConstraints::iterator i = equalityConstraints->begin() ;
	 i != equalityConstraints->end() ;
	 ++i) {
      //      DEBUG_MESSAGE("Set dimension to " << getNbrNonFixedParameters()) ;
      i->setDimension(getNbrNonFixedParameters()) ;

      patIterator<patBetaLikeParameter>* paramIter = 
	createAllParametersIterator() ;

      for (paramIter->first() ;
	   !paramIter->isDone() ;
	   paramIter->next()) {
	patBetaLikeParameter theParam = paramIter->currentItem() ;
	if (!theParam.isFixed) {
	  DEBUG_MESSAGE("x[" << theParam.index << "]=" << theParam.name) ;
	  i->setVariable(theParam.name,theParam.index) ;
	}
      }
    }
  }
  if (inequalityConstraints != NULL) {
    DEBUG_MESSAGE("Process inequality constraints") ;
    
    for (patListNonLinearConstraints::iterator i = inequalityConstraints->begin() ;
	 i != inequalityConstraints->end() ;
	 ++i) {
      i->setDimension(getNbrNonFixedParameters()) ;

      patIterator<patBetaLikeParameter>* paramIter = 
	createAllParametersIterator() ;

      for (paramIter->first() ;
	   !paramIter->isDone() ;
	   paramIter->next()) {
	patBetaLikeParameter theParam = paramIter->currentItem() ;
	if (!theParam.isFixed) {
	  DEBUG_MESSAGE("x[" << theParam.index << "]=" << theParam.name) ;
	  i->setVariable(theParam.name,theParam.index) ;
	}
      }

    }
  }

  // Assign beta parameters to random parameters

  for (map<patString,pair<patRandomParameter,patArithRandom*> >::iterator i = 
	 randomParameters.begin() ;
       i != randomParameters.end() ;
       ++i) {
    patString mean = i->second.second->getLocationParameter() ;
    map<patString,patBetaLikeParameter>::iterator foundMean =
      betaParam.find(mean) ;
    if (foundMean == betaParam.end()) {
      stringstream str ;
      str << "Unknown parameter " << mean << " defined in random parameter "
	  << i->first ; 
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
    patString stdDev = i->second.second->getScaleParameter() ;
    map<patString,patBetaLikeParameter>::iterator foundStdDev =
      betaParam.find(stdDev) ;
    if (foundStdDev == betaParam.end()) {
      stringstream str ;
      str << "Unknown parameter " << stdDev << " defined in random parameter "
	  << i->first ; 
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
    i->second.first.location = &(foundMean->second) ;
    i->second.first.scale = &(foundStdDev->second) ;
   

    // Check if it is a panel variable

    //    DEBUG_MESSAGE("#########################") ;
    patString theName = i->first ;
    list<patString>::iterator found = find(panelVariables.begin(),
					   panelVariables.end(),
					   theName) ;
    if (found == panelVariables.end()) {
      i->second.first.panel = patFALSE ;
      observationsParameters.push_back(&(i->second.first)) ;
    }
    else {
      DEBUG_MESSAGE(theName << " is for panel") ;
      i->second.first.panel = patTRUE ;
      individualParameters.push_back(&(i->second.first)) ;
    }


    // Check for errors in the syntax of panel data variables
  
    if ((panelExpr != NULL) && panelVariables.empty()) {
      stringstream str ;
      str << "User Id ["<<*panelExpr<<"] has been defined for panel data, but no effect variable has been included" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
  
    for (list<patString>::iterator ii = panelVariables.begin() ;
	 ii != panelVariables.end() ;
	 ++ii) {
      map<patString,pair<patRandomParameter,patArithRandom*> >::iterator 
	found = randomParameters.find(*ii) ;
      if (found == randomParameters.end()) {
	stringstream str ;
	str << "Panel variable " << *ii << " is unknown " ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return ;
      }
    }
  
    // Check if there is a mass at zero
    patReal foundMass(0.0) ;
    //    DEBUG_MESSAGE("Check if " << i->first << " has a mass at zero") ;
    for (list<pair<patString,patReal> >::iterator iterMass = 
	   listOfMassAtZero.begin() ;
	 iterMass != listOfMassAtZero.end() ;
	 ++iterMass) {
      if (iterMass->first == i->first) {
	foundMass = iterMass->second ;
	break ;
      }
    }
    if (foundMass > 0.0) {
      //      DEBUG_MESSAGE("YES") ;
      i->second.first.massAtZero = foundMass ;
    }
    else {
      //      DEBUG_MESSAGE("NO") ;
    }
  }
  

  // Set indices in linear utility functions

  //  DEBUG_MESSAGE("Set indices") ;

  for (map<patString, patAlternative>::iterator util = utilities.begin() ;
       util != utilities.end() ;
       ++util) {
    for (patUtilFunction::iterator term = util->second.utilityFunction.begin() ;
	 term != util->second.utilityFunction.end() ;
	 ++term) {
      if (!term->random) {
	term->betaIndex = patModelSpec::the()->getBetaId(term->beta,err) ;      
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
      }
      patBoolean xfound ;
      term->xIndex = patModelSpec::the()->getUsedAttributeId(term->x,&xfound) ;
      if (!xfound) {
	stringstream str ;
	str << "Attribute " << term->x << " undefined" ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe());
	return ;
      }
    }
  }
  
  for (map<patString,patArithNode*>::iterator ii = expressions.begin() ;
       ii != expressions.end() ;
       ++ii) {
    for (unsigned long i = 0 ; i < attributeNames.size() ; ++i) {
      
      ii->second->setAttribute(attributeNames[i],i) ;
      ii->second->expand(err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }
    } 
  }

  if (choiceExpr != NULL) {
    choiceExpr->expand(err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return ;
    }
  }
  if (aggLastExpr != NULL) {
    aggLastExpr->expand(err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return ;
    }
  }
  if (aggWeightExpr != NULL) {
    aggWeightExpr->expand(err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return ;
    }
  }
  if (weightExpr != NULL) {
    weightExpr->expand(err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return ;
    }
  }
  if (panelExpr != NULL) {
    panelExpr->expand(err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return ;
    }
  }
  if (excludeExpr != NULL) {
    excludeExpr->expand(err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return ;
    }
  }
  if (groupExpr != NULL) {
    groupExpr->expand(err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return ;
    }
  }

  allBetaIter = createAllBetaIterator() ;
  if (allBetaIter == NULL) {
    err = new patErrNullPointer("patIterator<patBetaLikeParameter>") ;
    WARNING(err->describe()) ;
    return  ;
  }


  // Identify parameters for the correction of the selection bias

  if (!selectionBiasParameters.empty()) {
    if (isMNL()) {
      err = new patErrMiscError("The correction for selection bias is captured by the constants for MNL. The section [SelectionBias] is designed for other GEV models. Just make sure you have a full set of constants and  remove section [SelectionBias]") ;
      WARNING(err->describe()) ;
      return ;
    }
    if (isMixedLogit()) {
      err = new patErrMiscError("The correction for selection bias is valid for GEV models only. Not for mixtures of GEV. Pls remove section [SelectionBias]") ;
      WARNING(err->describe()) ;
      return ;
    }
    if (isPanelData()) {
      err = new patErrMiscError("The correction for selection bias is not yet implemented for panel data models. Sorry.") ;
      WARNING(err->describe()) ;
      return ;
    }
    selectionBiasPerAlt.resize(getNbrAlternatives(),NULL) ;
    for (map<unsigned long, patString>::const_iterator i =  selectionBiasParameters.begin() ;
	 i != selectionBiasParameters.end() ;
	 ++i) {
      unsigned long internalId = getAltInternalId(i->first,err) ;
      map<patString,patBetaLikeParameter>::iterator found = 
	betaParam.find(i->second) ;
      if (found == betaParam.end()) {
	stringstream str ;
	str << "Beta parameter " << i->second << " not found" ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return ;
      }
      selectionBiasPerAlt[internalId] = &(found->second) ;
    }
  }

  // SNP terms

  DEBUG_MESSAGE("Check for SNP") ;
  if (applySnpTransform()) {
    DEBUG_MESSAGE("YES___ SNP:" << snpBaseParameter) ;
    if (!patModelSpec::the()->isIndividualSpecific(snpBaseParameter)) {
      stringstream str ;
      str << "In a panel data setting, only individual specific parameters can be tested with the SNP approach. Parameter " << snpBaseParameter << " must appear in the [PanelData] section" ; 
      err = new patErrMiscError(str.str()) ;
      return ;
    }
    unsigned short nbrOfSnpTerms = numberOfSnpTerms() ;
    ordersOfSnpTerms.resize(nbrOfSnpTerms) ;
    coeffOfSnpTerms.resize(nbrOfSnpTerms) ;
    unsigned short j = 0 ;
    for (list<pair<unsigned short,patString> >::iterator iter =
	   listOfSnpTerms.begin() ;
	 iter != listOfSnpTerms.end() ;
	 ++iter) {
      ordersOfSnpTerms[j] = iter->first ;
      map<patString,patBetaLikeParameter>::iterator found = betaParam.find(iter->second) ;
      if (found == betaParam.end()) {
	stringstream str ;
	str << "Coefficient " << iter->second << " is used for a SNP term, but not defined in section [Beta]" ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return ;
      }
      coeffOfSnpTerms[j] = &(found->second) ;
      idOfSnpBetaParameters.push_back(found->second.id) ;
      ++j ;
    }
  }

  // Parameters for ordinal logit

  for (map<unsigned long, patString>::iterator i = 
	 ordinalLogitThresholds.begin() ;
       i != ordinalLogitThresholds.end() ;
       ++i) {
    map<patString,patBetaLikeParameter>::iterator found = 
      betaParam.find(i->second) ;
    if (found == betaParam.end()) {
      stringstream str ;
      str << "Unknown parameter for Ordinal Logit: " << i->second ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
    }
    ordinalLogitBetaThresholds[i->first] = &(found->second) ;
  }

  if (useModelForGianluca) {

    // Compute the index of the dependent variable for the Gianluca model
    
    DEBUG_MESSAGE("---> regressionObservation = " << regressionObservation) ;
    
    patBoolean xfound ;
    gianlucaObservationId = getUsedAttributeId(regressionObservation,&xfound) ;
    if (!xfound) {
      stringstream str ;
      str << "Attribute " << regressionObservation<< " undefined" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe());
      return ;
    }
    
    DEBUG_MESSAGE("---> startingtime = " << fixationStartingTime) ;
    
    startingTimeId = getUsedAttributeId(fixationStartingTime,&xfound) ;
    if (!xfound) {
      stringstream str ;
      str << "Attribute " << fixationStartingTime << " undefined" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe());
      return ;
    }
    
    for (patUtilFunction::iterator term = acquisitionModel.begin() ;
	 term != acquisitionModel.end() ;
	 ++term) {
      
      term->betaIndex = patModelSpec::the()->getBetaId(term->beta,err) ;      
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      patBoolean xfound ;
      term->xIndex = patModelSpec::the()->getUsedAttributeId(term->x,&xfound) ;
      if (!xfound) {
	stringstream str ;
	str << "Attribute " << term->x << " undefined" ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe());
	return ;
      }
    }
    
    for (patUtilFunction::iterator term = validationModel.begin() ;
	 term != validationModel.end() ;
	 ++term) {
      
      term->betaIndex = patModelSpec::the()->getBetaId(term->beta,err) ;      
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      patBoolean xfound ;
      term->xIndex = patModelSpec::the()->getUsedAttributeId(term->x,&xfound) ;
      if (!xfound) {
	stringstream str ;
	str << "Attribute " << term->x << " undefined" ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe());
	return ;
      }
    }
    
    map<patString,patBetaLikeParameter>::iterator found =
      betaParam.find(durationNameParam) ;
    if (found == betaParam.end()) {
      WARNING("**** No parameter " << durationNameParam << " has been defined") ;
      durationModelParam = NULL ;
    }
    else {
      durationModelParam = &(found->second) ;
    }
    
    found = betaParam.find(sigmaAcqName) ;
    if (found == betaParam.end()) {
      WARNING("**** No parameter " << sigmaAcqName << " has been defined") ;
      sigmaAcq = NULL ;
    }
    else {
      sigmaAcq = &(found->second) ;
    }
    
    found = betaParam.find(sigmaValName) ;
    if (found == betaParam.end()) {
      WARNING("**** No parameter " << sigmaValName << " has been defined") ;
      sigmaVal = NULL ;
    }
    else {
      sigmaVal = &(found->second) ;
    }
  }

  // Generalized Extreme Value

  generalizedExtremeValueParameter = NULL ;
  if (generalizedExtremeValueParameterName != "") {
    map<patString,patBetaLikeParameter>::iterator found =
      betaParam.find(generalizedExtremeValueParameterName) ;
    if (found == betaParam.end()) {
      stringstream str ;
      str << "**** No parameter " << generalizedExtremeValueParameterName << " has been defined" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
    else {
      generalizedExtremeValueParameter = &(found->second) ;
    }
  }
  
  indexComputed = patTRUE ;
}


patString patModelSpec::getScaleNameFromId(unsigned long id) const {

  stringstream str ;
  str << "Scale" << id ;
  return (patString(str.str())) ;
 
}



unsigned long patModelSpec::getBetaId(const patString& name, patError*& err) {
  map<patString,patBetaLikeParameter>::const_iterator found =
    betaParam.find(name) ;
  if (found == betaParam.end()) {
    stringstream str ;
    str << "Parameter " << name << " is unknown" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return unsigned();    
  }
  return found->second.id ;
  
}

void patModelSpec::setEstimatedCoefficients(const patVariables* x,
					    patError*& err) {

  if (nonFixedParameters.size() != x->size()) {
    stringstream str ;
    str << "There are " << nonFixedParameters.size()
	<< " parameters to be estimated, not " << x->size() ;
    err = new patErrMiscError(err->describe()) ;
    WARNING(err->describe()) ;
    return ;
  }
  for (unsigned long i = 0 ; i < x->size() ; ++i) {
    patBetaLikeParameter* p = nonFixedParameters[i] ;
    if (p == NULL) {
      err = new patErrNullPointer("patBetaLikeParameter") ;
      WARNING(err->describe()) ;
      return ;
    }
    p->estimated = (*x)[i] ;
  }

  copyParametersValue(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}


patVariables patModelSpec::getEstimatedCoefficients(patError*& err) {

  patVariables x(nonFixedParameters.size()) ;
  for (unsigned long i = 0 ; i < x.size() ; ++i) {
    patBetaLikeParameter* p = nonFixedParameters[i] ;
    if (p == NULL) {
      err = new patErrNullPointer("patBetaLikeParameter") ;
      WARNING(err->describe()) ;
      return patVariables() ;
    }
    x[i] = p->estimated ;
  }
  return x ;
}

 unsigned long patModelSpec::getIdCnlNestCoef(unsigned long nest) {
  // The vector of parameters for CNL is composed of
  // 1) nNests mu_m
  // 2) For each nest, For each alt, the corresponding alpha 
  return nest ;
}

unsigned long patModelSpec::getIdCnlAlpha(unsigned long nest,
					      unsigned long alt) const {
  
   if (!indexComputed) {
     FATAL("Indices have not been computed yet") ;
   }
  // The vector of parameters for CNL is composed of
  // 1) nNests mu_m
  // 2) For each nest, For each alt such that alpha(nest,alpha) !=0, the
  // corresponding alpha

   assert(cumulIndex.size() == getNbrNests()+1) ;

   for (unsigned long ptr = cumulIndex[nest] ;
       ptr < cumulIndex[nest+1] ;
       ++ptr) {
    if (nonZeroAlphasPerNest[nest][ptr-cumulIndex[nest]] == alt) {
      return ptr ;
    }
  }

  return patBadId ;
	 
}

patBoolean patModelSpec::isCnlParamNestCoef(unsigned long index) {
  return (index < getNbrNests()) ;
}

 patBoolean patModelSpec::isCnlParamAlpha(unsigned long index) {
  return (!isCnlParamNestCoef(index)) ;
}

 pair<unsigned long,unsigned long>
patModelSpec::getNestAltFromParamIndex(unsigned long index) {
   unsigned long nest ;
   unsigned long alt ;
   if (index < getNbrNests()) {
     nest = index ;
     alt = patBadId ;
   }
   else {
     nest = 0 ;
     while (index >= cumulIndex[nest+1]) {
       ++nest ;
     }    
     alt = nonZeroAlphasPerNest[nest][index-cumulIndex[nest]] ;

     if (getIdCnlAlpha(nest,alt) != index) {
       FATAL("Index incompatibility") ;
     }
  }
  return pair<unsigned long,unsigned long>(nest,alt) ;
}


patVariables* patModelSpec::getPtrBetaParameters() {
  return &betaParameters ;
}

patReal* patModelSpec::getPtrMu() {
  return &muValue ;
}

patVariables* patModelSpec::getPtrModelParameters() {
  return &modelParameters ;
}

patVariables* patModelSpec::getPtrScaleParameters() {
  return &scaleParameters ;
}

patIterator<patBetaLikeParameter>* patModelSpec::createAllBetaIterator() {
  static patBoolean first(patTRUE) ;
  if (first) {
    DEBUG_MESSAGE("Should be called once") ;
    allBetaIterator.addIterator(createBetaIterator()) ;
    first = patFALSE ;
  }
  return &allBetaIterator ;
}

patIterator<patBetaLikeParameter>* patModelSpec::createAllModelIterator() {
  static patBoolean first(patTRUE) ;
  if (first) {
    if (isNL()) {
      allModelIterator.addIterator(createNlNestIterator()) ;
    }
    if (isCNL()) {
      allModelIterator.addIterator(createCnlNestIterator()) ;
      allModelIterator.addIterator(createCnlAlphaIterator()) ;
    }
    if (isNetworkGEV()) {
      allModelIterator.addIterator(createNetworkNestIterator()) ;
      allModelIterator.addIterator(createNetworkAlphaIterator()) ;
    }
    first = patFALSE ;
  }
  return &allModelIterator ;
}

patIterator<patBetaLikeParameter>* patModelSpec::createAllParametersIterator() {
  static patBoolean first(patTRUE) ;
  if (first) {
    allParametersIterator.addIterator(createAllBetaIterator()) ;
    allParametersIterator.addIterator(createAllModelIterator()) ;
    allParametersIterator.addIterator(createScaleIterator()) ;
    first = patFALSE ;
  }
  return &allParametersIterator ;
}


patIterator<patBetaLikeParameter>* patModelSpec::createBetaIterator()  {
  if (betaIterator == NULL) {
    betaIterator = new patBetaLikeIterator(&betaParam) ;
  }
  return betaIterator ;
}

patIterator<patBetaLikeParameter>* patModelSpec::createScaleIterator() {
  if (scaleIterator == NULL) {
    scaleIterator = new patBetaLikeIterator(&scaleParam) ;
  }
  return scaleIterator ;
}

patIterator<patBetaLikeParameter>* patModelSpec::createNlNestIterator() {
  if (nlNestIterator == NULL) {
    nlNestIterator = new patNlNestIterator(&nlNestParam) ;
  }
  return nlNestIterator ;
}

patIterator<patBetaLikeParameter>* patModelSpec::createCnlNestIterator() {
  if (cnlNestIterator == NULL) {
    cnlNestIterator = new patBetaLikeIterator(&cnlNestParam) ;
  }
  return cnlNestIterator ;
}

patIterator<patBetaLikeParameter>* patModelSpec::createCnlAlphaIterator() {
  if (cnlAlphaIterator == NULL) {
    cnlAlphaIterator = new patCnlAlphaIterator(&cnlAlphaParam) ;
  }
  return cnlAlphaIterator ;
}

patIterator<patCnlAlphaParameter>* patModelSpec::createFullCnlAlphaIterator() {
  if (cnlFullAlphaIterator == NULL) {
    cnlFullAlphaIterator = new patFullCnlAlphaIterator(&cnlAlphaParam) ;
  }
  return cnlFullAlphaIterator ;
}

patIterator<patBetaLikeParameter>* patModelSpec::createNetworkNestIterator() {
  if (networkNestIterator == NULL) {
    networkNestIterator = new patBetaLikeIterator(&networkGevNodes) ;
  }
  return networkNestIterator ;
}

patIterator<patBetaLikeParameter>* patModelSpec::createNetworkAlphaIterator() {
  if (networkAlphaIterator == NULL) {
    networkAlphaIterator = new patNetworkAlphaIterator(&networkGevLinks) ;
  }
  return networkAlphaIterator ;
}



unsigned long patModelSpec::getNbrNonFixedBetas() const {
  return nonFixedBetas ;
}


unsigned long patModelSpec::getNbrNonFixedParameters() const {
  return nonFixedParameters.size() ;
}

unsigned long patModelSpec::getNbrModelParameters() const {
  switch (modelType) {
  case patOLtype:
    return 0 ;
  case patBPtype:
    return 0 ;
  case patMNLtype:
    return 0 ;
  case patNLtype:
    return nlNestParam.size() ;
  case patCNLtype:
    return cnlNestParam.size() + cnlAlphaParam.size() ;
  case patNetworkGEVtype :
    if (theNetworkGevModel == NULL) {
      return 0 ;
    }
    return theNetworkGevModel->getNbrParameters() ;
  default :
    return patBadId ;
  }
}
  /**
   */
unsigned long patModelSpec::getNbrScaleParameters() const {
  return scaleParam.size() ;
}


void patModelSpec::writeReport(patString fileName, patError*& err) {


  // Compute the width of the largest name
  patString title ;
  long width = 0 ;
  patIterator<patBetaLikeParameter>* iter = createAllParametersIterator() ;
  for (iter->first() ;
       !iter->isDone() ;
       iter->next()) {
    patBetaLikeParameter beta = iter->currentItem() ;
    if (beta.name.size() > width) {
      width = beta.name.size() ;
    }
  }
  if (width < 14) {
    width = 15 ;
  } 
  else {
    ++width ;
  }

  DEBUG_MESSAGE("Write " << fileName) ;
  ofstream reportFile(fileName.c_str()) ;
  patAbsTime now ;
  now.setTimeOfDay() ;
  reportFile << "// This file has automatically been generated." << endl ;
  reportFile << "// " << now.getTimeString(patTsfFULL) << endl ;
  reportFile << "// " << patVersion::the()->getCopyright() << endl ;
  reportFile << endl ;
  reportFile << patVersion::the()->getVersionInfoDate() << endl ;
  reportFile << patVersion::the()->getVersionInfoAuthor() << endl ;

  reportFile << endl ;
  for (list<patString>::iterator i = modelDescription.begin() ;
       i != modelDescription.end() ;
       ++i) {
    reportFile << "   " << *i << endl ;
  }
  reportFile << endl ;

  unsigned short columnWidth = 32 ;

  reportFile << fillWithBlanks(patString("Model: "),columnWidth,0) ;
  if (isMixedLogit()) {
    reportFile << " Mixed " ;
  }  
  reportFile << modelTypeName() << endl ;
  if (isMixedLogit()) {
    patString tDraws("Number of ") ;
    if (estimationResults.halton) {
      tDraws += "Halton " ;
    }
    if (estimationResults.hessTrain) {
      tDraws += "Hess-Train " ;
    }
    tDraws += "draws: " ;
    reportFile << fillWithBlanks(tDraws,columnWidth,0) 
	       <<  getNumberOfDraws() << endl ;    
  }
  reportFile << fillWithBlanks(patString("Number of estimated parameters: "),columnWidth,0) 
	     << getNbrNonFixedParameters() << endl ;
  
  patString tObs("Number of ") ;
  if (isAggregateObserved()) {
    tObs += "aggregate " ;
  }
  tObs += "observations: " ;
  reportFile << fillWithBlanks(tObs,columnWidth,0)
  	     << estimationResults.numberOfObservations << endl ;
  reportFile << fillWithBlanks("Number of individuals: ",columnWidth,0) 
	     << estimationResults.numberOfIndividuals << endl ;
  reportFile << fillWithBlanks("Null log likelihood: ",columnWidth,0) 
	     << theNumber.formatStats(estimationResults.nullLoglikelihood) << endl ;
  if (allAlternativesAlwaysAvail) {
  reportFile << fillWithBlanks("Cte log likelihood: ",columnWidth,0) 
	     << theNumber.formatStats(estimationResults.cteLikelihood) << endl ;
  }
  reportFile << fillWithBlanks("Init log likelihood: ",columnWidth,0) 
	     << theNumber.formatStats(estimationResults.initLoglikelihood) << endl ;
  reportFile << fillWithBlanks("Final log likelihood: ",columnWidth,0) 
	     << theNumber.formatStats(estimationResults.loglikelihood) << endl ;
  reportFile << fillWithBlanks("Likelihood ratio test: ",columnWidth,0)
	     << theNumber.formatStats(-2.0 * (estimationResults.nullLoglikelihood - estimationResults.loglikelihood)) << endl ;
  reportFile << fillWithBlanks("Rho-square: ",columnWidth,0) 
	     << theNumber.formatStats(1.0 - (estimationResults.loglikelihood / estimationResults.nullLoglikelihood)) << endl ;
  reportFile << fillWithBlanks("Adjusted rho-square: ",columnWidth,0) 
	     << theNumber.formatStats(1.0 - ((estimationResults.loglikelihood-patReal(getNbrNonFixedParameters())) / estimationResults.nullLoglikelihood)) << endl ;

  reportFile << fillWithBlanks("Final gradient norm: ",columnWidth,0) 
	     << theNumber.format(patTRUE,
				 patFALSE,
				 3,
				 estimationResults.gradientNorm) << endl ;
  reportFile << fillWithBlanks("Diagnostic: ",columnWidth,0) ;
  reportFile << estimationResults.diagnostic << endl ;
  if (estimationResults.iterations != 0) {
    reportFile << fillWithBlanks("Iterations: ",columnWidth,0) ;
    reportFile << estimationResults.iterations << endl ;
  }
    reportFile << fillWithBlanks("Run time: ",columnWidth,0) ;
    reportFile << estimationResults.runTime << endl ;
  
  reportFile << fillWithBlanks("Variance-covariance: ",columnWidth,0) ;
  if (patParameters::the()->getgevVarCovarFromBHHH() == 0) {
    if (patModelSpec::the()->isSimpleMnlModel() && patParameters::the()->getBTRExactHessian()) {
      reportFile << "from analytical hessian" << endl ;
    }
    else {
      reportFile << "from finite difference hessian" << endl ;
    }
  }
  else {
    reportFile << "from BHHH matrix" << endl ;
  }
  
  unsigned short nSampleFiles = patFileNames::the()->getNbrSampleFiles() ;
  if (nSampleFiles == 1) {
    reportFile << fillWithBlanks("Sample file: ",columnWidth,0) ;
  }
  else {
    reportFile << fillWithBlanks("Sample files: ",columnWidth,0) ;
  }
  reportFile << patFileNames::the()->getSamFile(0,err) << endl ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  if (nSampleFiles > 1) {
    for (unsigned short i = 1 ; i < nSampleFiles ; ++i) {
      reportFile << "\t\t" << patFileNames::the()->getSamFile(i,err) << endl ;
    }
  }
  
  //  reportFile << setprecision(7) << setiosflags(ios::scientific|ios::showpos) ;  
  if (estimationResults.varCovarMatrix == NULL) {
    estimationResults.isVarCovarAvailable = patFALSE ;
  }
  if (estimationResults.robustVarCovarMatrix == NULL) {
    estimationResults.isRobustVarCovarAvailable = patFALSE ;
  }
  
  
  patVariables stdErr ;
  if (estimationResults.isVarCovarAvailable) {
    patMyMatrix* varCovar = estimationResults.varCovarMatrix ;
    DEBUG_MESSAGE("Var-covar is " << varCovar->nRows()
		  << "x" 
		  << varCovar->nCols()) ;
    for (unsigned long i = 0 ; i < varCovar->nRows() ; ++i) {
      if ((*varCovar)[i][i] < 0) {
	stdErr.push_back(patMaxReal) ;
      }
      else{
	stdErr.push_back(sqrt((*varCovar)[i][i])) ; 
      }
    }
  }
  //  DEBUG_MESSAGE("--> HERE 2: runtime = " << estimationResults.runTime) ;
  patVariables robustStdErr ;
  if (estimationResults.isRobustVarCovarAvailable) {
    patMyMatrix* robustVarCovar = estimationResults.robustVarCovarMatrix ;
    for (unsigned long i = 0 ; i < robustVarCovar->nRows() ; ++i) {
      if ((*robustVarCovar)[i][i] < 0) {
	robustStdErr.push_back(patMaxReal) ;
      } 
      else {
	robustStdErr.push_back(sqrt((*robustVarCovar)[i][i])) ; 
      }
    }
  }
 
  reportFile << endl ;

  reportFile << "Utility parameters" << endl ;
  reportFile << "******************" << endl ;
  
  
  unsigned short nCols = 10 ;
  patOutputTable tableUtilParam(nCols,patTRUE) ;
  
  vector<patString> aTitle(nCols) ;
  
  aTitle[0] = "Name" ;
  aTitle[1] = "Value" ;
  aTitle[2] = "Std err" ;
  aTitle[3] = "t-test" ;
  if (patParameters::the()->getgevPrintPValue()) {
    aTitle[4] = "p-val" ;
  }
  else {
    aTitle[4] = "" ;    
  }
  aTitle[5] = "" ;    // column for the * if non significant
  if (estimationResults.isRobustVarCovarAvailable) {
    aTitle[6] = "Rob. std err" ;
    aTitle[7] = "Rob. t-test" ;
    if (patParameters::the()->getgevPrintPValue()) {
      aTitle[8] = "Rob. p-val" ;
    }
    else {
      aTitle[8] = "" ;
    }
    aTitle[9] = "" ;    // column for the * if non significant
  }
  tableUtilParam.appendRow(aTitle,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  vector<patString> underline(nCols) ;
  for (unsigned short i = 0 ; i < nCols ; ++i) {
    underline[i] = patString(aTitle[i].size(),'-') ;
  }
  tableUtilParam.appendRow(underline,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  for (map<patString,patBetaLikeParameter>::const_iterator i = 
	 betaParam.begin() ;
       i != betaParam.end() ;
       ++i) { // loop on beta parameters
    vector<patString> aRow(nCols,"") ;
    
    if (!i->second.hasDiscreteDistribution) {
      aRow[0] = i->second.name ;
      aRow[1] = theNumber.formatParameters(i->second.estimated) ;
      if (i->second.isFixed) {
	aRow[2] = "--fixed--" ;
      } 
      else if (i->second.hasDiscreteDistribution) {
	aRow[2] = "--distrib--" ;
      }
      else {
	if (estimationResults.isVarCovarAvailable) {
	  patReal ttest = i->second.estimated  / stdErr[i->second.index] ; 
	  aRow[2] = theNumber.formatParameters(stdErr[i->second.index]) ;
	  aRow[3] = theNumber.formatTTests(ttest) ;
	  if (patParameters::the()->getgevPrintPValue()) {
	    patReal pvalue = patPValue(patAbs(ttest),err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
      return ;
	    }
	    aRow[4] = theNumber.formatTTests(pvalue) ;
	  }
	  if (patAbs(ttest) < patParameters::the()->getgevTtestThreshold() 
	      || !isfinite(ttest)) {
	    aRow[5] = patParameters::the()->getgevWarningSign() ;
	  }
	  if (estimationResults.isRobustVarCovarAvailable) {
	    patReal rttest = i->second.estimated  / robustStdErr[i->second.index] ; 
	    aRow[6] = theNumber.formatParameters(robustStdErr[i->second.index]) ;
	    aRow[7] = theNumber.formatTTests(rttest) ;
	    if (patParameters::the()->getgevPrintPValue()) {
	      
	      patReal pvalue = patPValue(patAbs(rttest),err) ;
	      if (err != NULL) {
		WARNING(err->describe()) ;
		return ;
	      }
	      aRow[8] = theNumber.formatTTests(pvalue) ;
	    }
	    
	    if (patAbs(rttest) < patParameters::the()->getgevTtestThreshold()
		|| !isfinite(rttest)) {
	      aRow[9] = patParameters::the()->getgevWarningSign() ;
	    }
	  }
	}
	else {
	  aRow[2] = "var-covar unavailable" ;
	}
      }
    }
    tableUtilParam.appendRow(aRow,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
  
  tableUtilParam.computeColWidth() ;
  reportFile << tableUtilParam ;
  
 
  if (!mu.isFixed || mu.estimated != 1.0) {  // mu is estimated
    reportFile << endl ;
    reportFile << "Homogeneity parameter (mu)" << endl ;
    reportFile << "**************************" << endl ;

    nCols = 13 ;
    patOutputTable tableMu(nCols,patTRUE) ;
    vector<patString> tMu(nCols) ;
    tMu[0] = "Value" ;
    tMu[1] = "Std err" ;
    tMu[2] = "t-test(0)" ;
    if (patParameters::the()->getgevPrintPValue()) {
      tMu[3] = "p-val(0)" ;
    }
    else {
      tMu[3] = "" ;
    }
    tMu[4] = "t-test (1)" ;
    if (patParameters::the()->getgevPrintPValue()) {
      tMu[5] = "p-val(1)" ;
    }
    else {
      tMu[5] = "" ;
    }
    tMu[6] = "" ; // column for the * if non significant
    if (estimationResults.isRobustVarCovarAvailable) {
      
      tMu[7] = "Rob. std err" ;
      tMu[8] = "Rob. t-test(0)" ;
      if (patParameters::the()->getgevPrintPValue()) {
	tMu[9] = "Rob. p-val(0)" ;
      }
      else {
	tMu[9] = "" ;
    }
      tMu[10] = "Rob. t-test(1)" ;
      if (patParameters::the()->getgevPrintPValue()) {
	tMu[11] = "Rob. p-val(1)" ;
      }
      else {
	tMu[11] = "" ;
      }
      tMu[12] = "" ; // column for the * if non significant
    }

    tableMu.appendRow(tMu,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }


    vector<patString> tUnd(nCols) ;
    for (unsigned short i = 0 ; i < nCols ; ++i) {
      tUnd[i] = patString(tMu[i].size(),'-') ;
    }
    tableMu.appendRow(tUnd,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }


    fill(tMu.begin(),tMu.end(),"") ;

    tMu[0] = theNumber.formatParameters(mu.estimated) ;
    if (mu.isFixed) {
      tMu[1] = "--fixed--" ;
    }
    else {
      if (estimationResults.isVarCovarAvailable) {
	patReal ttest0 = mu.estimated / stdErr[mu.index] ;
	patReal ttest1 = (mu.estimated-1.0) / stdErr[mu.index] ;
	tMu[1] = theNumber.formatParameters(stdErr[mu.index]) ;
	tMu[2] = theNumber.formatTTests(ttest0) ;
	if (patParameters::the()->getgevPrintPValue()) {

	  patReal pvalue = patPValue(patAbs(ttest0),err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return ;
	  }
	  tMu[3] = theNumber.formatTTests(pvalue) ;
	  
	}

	tMu[4] = theNumber.formatTTests(ttest1) ;
	if (patParameters::the()->getgevPrintPValue()) {
	  
	  patReal pvalue = patPValue(patAbs(ttest1),err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return ;
	  }
	  tMu[5] = theNumber.formatTTests(pvalue) ;
	}

	if (patAbs(ttest0) < patParameters::the()->getgevTtestThreshold() ||
	    patAbs(ttest1) < patParameters::the()->getgevTtestThreshold() ||
	    !isfinite(ttest0) ||
	    !isfinite(ttest1) ) {
	  tMu[6] = patParameters::the()->getgevWarningSign() ;
	}
	if (estimationResults.isRobustVarCovarAvailable) {
	  patReal rttest0 = mu.estimated / robustStdErr[mu.index] ;
	  patReal rttest1 = (mu.estimated-1.0) / robustStdErr[mu.index] ;
	  tMu[7] = theNumber.formatParameters(robustStdErr[mu.index]) ;
	  tMu[8] = theNumber.formatTTests(rttest0) ;
	  if (patParameters::the()->getgevPrintPValue()) {
	    
	    patReal pvalue = patPValue(patAbs(rttest0),err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	    tMu[9] = theNumber.formatTTests(pvalue) ;
	    
	  }
	  
	  tMu[10] = theNumber.formatTTests(rttest1) ;
	  if (patParameters::the()->getgevPrintPValue()) {
	    
	    patReal pvalue = patPValue(patAbs(rttest1),err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	    tMu[11] = theNumber.formatTTests(pvalue) ;
	    
	  }
	  
	  if (patAbs(rttest0) < patParameters::the()->getgevTtestThreshold() ||
	      patAbs(rttest1) < patParameters::the()->getgevTtestThreshold() ||
	      !isfinite(rttest0) ||
	      !isfinite(rttest1)) {
	    tMu[11] =  patParameters::the()->getgevWarningSign() ;
	  }
	}
      }
      else {
	tMu[2] = "var-covar unavailable" ;
      }
    }
    tableMu.appendRow(tMu,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    
    tableMu.computeColWidth() ;
    reportFile << tableMu << endl ;

  }

    
  patIterator<patBetaLikeParameter>* modelIter = createAllModelIterator() ;
  modelIter->first() ;
  if (!modelIter->isDone()) {

    reportFile << "Model parameters" << endl ;
    reportFile << "******************" << endl ;

    nCols = 14 ;
    patOutputTable tableModPar(nCols,patTRUE) ;
    vector<patString> tMod(nCols,"") ;
    tMod[0] = "Name" ;
    tMod[1] = "Value" ;
    tMod[2] = "Std err" ;
    tMod[3] = "t-test(0)" ;
    if (patParameters::the()->getgevPrintPValue()) {
      tMod[4] = "p-val(0)" ;
    }
    else {
      tMod[4] = "" ;
    }
    tMod[5] = "t-test(1)" ;
    if (patParameters::the()->getgevPrintPValue()) {
      tMod[6] = "p-val(1)" ;
    }
    else {
      tMod[6] = "" ;
    }
    tMod[7] = "" ; // column for the * if non significant
    if (estimationResults.isRobustVarCovarAvailable) {
      tMod[8] = "Rob. std err" ;
      tMod[9] = "Rob. t-test(0)" ;
      if (patParameters::the()->getgevPrintPValue()) {
	tMod[10] = "Rob. p-val(0)" ;
      }
    else {
      tMod[10] = "" ;
    }
      tMod[11] = "Rob. t-test(1)" ;
      if (patParameters::the()->getgevPrintPValue()) {
	tMod[12] = "Rob. p-val(1)" ;
      }
      else {
	tMod[12] = "" ;
      }
      tMod[13] = "" ; // column for the * if non significant
    }
    tableModPar.appendRow(tMod,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }


    vector<patString> tUnd(nCols) ;
    for (unsigned short i = 0 ; i < nCols ; ++i) {
      tUnd[i] = patString(tMod[i].size(),'-') ;
    }
    tableModPar.appendRow(tUnd,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    

    fill(tMod.begin(),tMod.end(),"") ;

    
    for (modelIter->first() ;
	 !modelIter->isDone() ;
	 modelIter->next()) {
      patBetaLikeParameter bb = modelIter->currentItem() ;

      tMod[0] = bb.name ;
      tMod[1] = theNumber.formatParameters(bb.estimated) ;
      if (bb.isFixed) {
	tMod[2] = "--fixed--" ;
	for (patULong k = 3 ; k < nCols ; ++k) {
	  tMod[k] = "" ;
	}
      }
      else {
	if (estimationResults.isVarCovarAvailable) {
	  patReal ttest0 = bb.estimated  / stdErr[bb.index] ;
	  patReal ttest1 = (bb.estimated-1.0)  / stdErr[bb.index] ;
	  tMod[2] = theNumber.formatParameters(stdErr[bb.index]);
	  tMod[3] = theNumber.formatTTests(ttest0) ;
	  if (patParameters::the()->getgevPrintPValue()) {
	    patReal pvalue = patPValue(patAbs(ttest0),err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	    tMod[4] = theNumber.formatTTests(pvalue) ;
	    
	  }
	  tMod[5] = theNumber.formatTTests(ttest1) ;
	  if (patParameters::the()->getgevPrintPValue()) {

	    patReal pvalue = patPValue(patAbs(ttest1),err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	    tMod[6] = theNumber.formatTTests(pvalue) ;
	    
	  }
	  if (patAbs(ttest0) < patParameters::the()->getgevTtestThreshold() ||
	      patAbs(ttest1) < patParameters::the()->getgevTtestThreshold() ||
	      !isfinite(ttest0) ||
	      !isfinite(ttest1)) {
	    tMod[7] = patParameters::the()->getgevWarningSign() ;
	  }
	  if (estimationResults.isRobustVarCovarAvailable) {
	    patReal rttest0 = bb.estimated  / robustStdErr[bb.index] ;
	    patReal rttest1 = (bb.estimated-1.0)  / robustStdErr[bb.index] ;
	    tMod[8] = theNumber.formatParameters(robustStdErr[bb.index]) ;
	    tMod[9] = theNumber.formatTTests(rttest0) ;
	    if (patParameters::the()->getgevPrintPValue()) {
	      patReal pvalue = patPValue(patAbs(rttest0),err) ;
	      if (err != NULL) {
		WARNING(err->describe()) ;
		return ;
	      }
	      tMod[10] = theNumber.formatTTests(pvalue) ;
	      
	    }
	    tMod[11] = theNumber.formatTTests(rttest1) ;
	    if (patParameters::the()->getgevPrintPValue()) {
	      patReal pvalue = patPValue(patAbs(rttest1),err) ;
	      if (err != NULL) {
		WARNING(err->describe()) ;
		return ;
	      }
	      tMod[12] = theNumber.formatTTests(pvalue) ;
	      
	    }
	    if (patAbs(rttest0) < patParameters::the()->getgevTtestThreshold() ||
		patAbs(rttest1) < patParameters::the()->getgevTtestThreshold() ||
		!isfinite(rttest0) ||
		!isfinite(rttest1)) {
	      tMod[13] = patParameters::the()->getgevWarningSign() ;
	    }
	  }
	}
	else {
	  tMod[2] = "var-covar unavailable" ;
	}
      }
      tableModPar.appendRow(tMod,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
    tableModPar.computeColWidth() ;
    reportFile << tableModPar << endl ;
  }
  
  patIterator<patBetaLikeParameter>* scaleIter = createScaleIterator() ;
  scaleIter->first() ;
  if (!scaleIter->isDone()) { // !isDone()
    patBoolean anythingToReport(patFALSE) ;
    
    unsigned short nCols = 10 ;
    patOutputTable tableScaleParam(nCols,patTRUE) ;
    
    vector<patString> aScale(nCols) ;
    
    aScale[0] = "Name" ;
    aScale[1] = "Value" ;
    aScale[2] = "Std err" ;
    aScale[3] = "t-test(1)" ;
    if (patParameters::the()->getgevPrintPValue()) {
      aScale[4] = "p-val(1)" ;
    }
    else {
      aScale[4] = "" ;    
    }
    aScale[5] = "" ;    // column for the * if non significant
    if (estimationResults.isRobustVarCovarAvailable) {
      aScale[6] = "Rob. std err" ;
      aScale[7] = "Rob. t-test(1)" ;
      if (patParameters::the()->getgevPrintPValue()) {
	aScale[8] = "Rob. p-val(1)" ;
      }
      else {
	aScale[8] = "" ;
      }
      aScale[9] = "" ;    // column for the * if non significant
            
    }

    tableScaleParam.appendRow(aScale,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    
    vector<patString> underline(nCols) ;
    for (unsigned short i = 0 ; i < nCols ; ++i) {
      underline[i] = patString(aScale[i].size(),'-') ;
    }
    tableScaleParam.appendRow(underline,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }


    for (scaleIter->first() ;
	 !scaleIter->isDone() ;
	 scaleIter->next()) {
      patBetaLikeParameter bb = scaleIter->currentItem() ;
      fill(aScale.begin(),aScale.end(),"") ;
      aScale[0] = bb.name ;
      aScale[1] = theNumber.formatParameters(bb.estimated) ;
      if (bb.isFixed) {
	aScale[2] = "--fixed--" ;
      }
      else {
	anythingToReport = patTRUE ;
	if (estimationResults.isVarCovarAvailable) {
	  patReal ttest1 = (bb.estimated-1.0)  / stdErr[bb.index] ;
	  aScale[2] = theNumber.formatParameters(stdErr[bb.index])  ;
	  aScale[3] = theNumber.formatTTests(ttest1) ;
	  if (patParameters::the()->getgevPrintPValue()) {
	    patReal pvalue = patPValue(patAbs(ttest1),err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	    aScale[4] = theNumber.formatTTests(pvalue) ;
	    
	  }
	  if (patAbs(ttest1) < patParameters::the()->getgevTtestThreshold() ||
	      !isfinite(ttest1)) {
	    aScale[5] = patParameters::the()->getgevWarningSign() ;
	  }
	  if (estimationResults.isRobustVarCovarAvailable) {
	    patReal rttest1 = (bb.estimated-1.0)  / robustStdErr[bb.index] ;
	    aScale[6] = theNumber.formatParameters(robustStdErr[bb.index])  ;
	    aScale[7] = theNumber.formatTTests(rttest1) ;
	    if (patParameters::the()->getgevPrintPValue()) {
	      patReal pvalue = patPValue(patAbs(rttest1),err) ;
	      if (err != NULL) {
		WARNING(err->describe()) ;
		return ;
	      }
	      aScale[8] = theNumber.formatTTests(pvalue) ;
	      
	    }
	    if (patAbs(rttest1) < patParameters::the()->getgevTtestThreshold() ||
		!isfinite(rttest1)) {
	      aScale[9] = patParameters::the()->getgevWarningSign() ;
	    }
	  }
	}
	else {
	  aScale[2] = "var-covar unavailable" ;
	}
      }
      
      tableScaleParam.appendRow(aScale,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
    
    if (anythingToReport) {
      reportFile << endl ;
      reportFile << "Scale parameters" << endl ;
      reportFile << "****************" << endl ;
      tableScaleParam.computeColWidth() ;
      reportFile << tableScaleParam << endl ;
    }
    
  }
  
  if (!ratios.empty()) {
    reportFile << "Ratio of parameters" << endl ;
    reportFile << "*******************" << endl ;
    title = "Name" ;
    for (long ii = title.size() ; 
	 ii <= width ; 
	 ++ii) {
      reportFile << " " ;
    }
    reportFile << title ;
    reportFile << "\tValue" << endl ;
    
    for (map<patString,pair<patString, patString> >::const_iterator i =
	   ratios.begin() ;
	 i != ratios.end() ;
	 ++i) {
      //    DEBUG_MESSAGE("Ratio of " << i->second.first << " and " << i->second.second) ;
      reportFile << i->first << '\t' ;
      patBoolean found ;
      patBetaLikeParameter num = getBeta(i->second.first,&found) ;
      if (!found) {
	reportFile << "Unknown parameter: " << i->second.first << endl ;
      }
      patBetaLikeParameter denom = getBeta(i->second.second,&found) ;
      if (!found) {
	reportFile << "Unknown parameter: " << i->second.second << endl ;
      }
      
      reportFile << theNumber.formatParameters(num.estimated / denom.estimated) << endl ;
    }
  }

  //  DEBUG_MESSAGE("--> HERE 3: runtime = " << estimationResults.runTime) ;
  std::set<patCorrelation,patCompareCorrelation> correlation ;
  if (estimationResults.isVarCovarAvailable) { //   if (estimationResults.isVarCovarAvailable)
    
    patMyMatrix* varCovar = estimationResults.varCovarMatrix ;
    patMyMatrix* robustVarCovar = estimationResults.robustVarCovarMatrix ;
    
    for (map<patString,patBetaLikeParameter>::const_iterator i = 
	   betaParam.begin() ;
	 i != betaParam.end() ;
	 ++i) {

      if (!i->second.isFixed && !i->second.hasDiscreteDistribution) {
	patReal varI = (*varCovar)[i->second.index][i->second.index] ;
	patReal robustVarI ;
	if (estimationResults.isRobustVarCovarAvailable) {
	  robustVarI = (*robustVarCovar)[i->second.index][i->second.index] ;
	}
	for (map<patString,patBetaLikeParameter>::const_iterator j = 
	       betaParam.begin() ;
	     j != betaParam.end() ;
	     ++j) {
	  if (!j->second.isFixed  && 
	      !j->second.hasDiscreteDistribution && 
	      i->second.id != j->second.id) {
	    patReal varJ = (*varCovar)[j->second.index][j->second.index] ;
	    patReal robustVarJ ;
	    if (estimationResults.isRobustVarCovarAvailable) {
	      robustVarJ = (*robustVarCovar)[j->second.index][j->second.index] ;
	    }	    
	    patCorrelation c ;
	    if (i->second.name < j->second.name) {
	      c.firstCoef = i->second.name ;
	      c.secondCoef = j->second.name ;
	    }
	    else {
	      c.firstCoef = j->second.name ;
	      c.secondCoef = i->second.name ;
	    }
	    
	    c.covariance = (*varCovar)[i->second.index][j->second.index] ;
	    patReal tmp = varI * varJ ;
	    c.correlation = (tmp > 0) 
	      ? c.covariance / sqrt(tmp) 
	      : 0.0 ;
	    tmp = varI + varJ - 2.0 * c.covariance ;
	    c.ttest = (tmp > 0)
	      ? (i->second.estimated - j->second.estimated) / sqrt(tmp) 
	      : 0.0  ;
	    if (estimationResults.isRobustVarCovarAvailable) {
	      c.robust_covariance = (*robustVarCovar)[i->second.index][j->second.index] ;
	      patReal tmp = robustVarI * robustVarJ ;
	      c.robust_correlation = (tmp > 0)
		? c.robust_covariance / sqrt(tmp) 
		: 0.0 ;
	      tmp = robustVarI + robustVarJ - 2.0 * c.robust_covariance ;
	      c.robust_ttest = (tmp > 0)
		? (i->second.estimated - j->second.estimated) / sqrt(tmp) 
		: 0.0  ;
	    }
	    correlation.insert(c) ;

	  }
	}
      }
    }

    computeVarCovarOfRandomParameters(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }

    if (!varianceRandomCoef.empty()) {

      reportFile << endl ;
      reportFile << "Variance of normal random coefficients" << endl ;
      reportFile << "**************************************" << endl ;

      nCols = 4 ;
      patOutputTable tableVar(nCols,patTRUE) ;
      vector<patString> aRow(nCols) ;
      aRow[0] = "Name" ;
      aRow[1] = "Value" ;
      aRow[2] = "Std err" ;
      aRow[3] = "t-test" ;
      tableVar.appendRow(aRow,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      vector<patString> tUnd(nCols) ;
      for (unsigned short i = 0 ; i < nCols ; ++i) {
	tUnd[i] = patString(aRow[i].size(),'-') ;
      }
      tableVar.appendRow(tUnd,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }

      fill(aRow.begin(),aRow.end(),"") ;
      
      for (map<patString,pair<patReal,patReal> >::iterator i = 
	     varianceRandomCoef.begin() ;
	   i != varianceRandomCoef.end() ;
	   ++i) {
	
	map<patString,pair<patRandomParameter,patArithRandom*> >::iterator 
	  found = randomParameters.find(i->first) ;
	if (found == randomParameters.end()) {
	  stringstream str ;
	  str << "Parameter " << i->first << " not found" ;
	  err = new patErrMiscError(str.str()) ;
	  WARNING(err->describe()) ;
	  return ;
	}
	patDistribType theType = found->second.second->getDistribution()  ;

	aRow[0] = i->first ;

	patReal variance(0.0) ;
	patReal stdErr(0.0) ;
	patReal ttest(0.0) ;
	if (theType == NORMAL_DIST) {
	  variance = i->second.first ;
	  stdErr = i->second.second ;
	  ttest = i->second.first / i->second.second ;
	}
	else if (theType == UNIF_DIST) {
	  variance = i->second.first / 3.0 ;
	  stdErr = i->second.second / 3.0 ;
	  ttest = i->second.first / i->second.second ;
	}
	

	aRow[1] = theNumber.formatParameters(variance) ;
	if (relevantVariance[i->first]) {
	  aRow[2] = theNumber.formatParameters(stdErr) ;
	  aRow[3] = theNumber.formatTTests(ttest) ;
	}
	else {
	  aRow[2] = "--fixed--" ;
	}
	tableVar.appendRow(aRow,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
	
      }
      tableVar.computeColWidth() ;
      reportFile << tableVar << endl ;
    }

    if (!covarianceRandomCoef.empty()) {

      reportFile << "Covariance of random coefficients" << endl ;
      reportFile << "*******************************" << endl ;
      title = "Name1" ;
      for (long ii = title.size() ; 
	   ii <= width ; 
	   ++ii) {
	reportFile << " " ;
      }
      reportFile << title ;
      reportFile << '\t' ;
      title = "Name2" ;
      for (long ii = title.size() ; 
	   ii <= width ; 
	   ++ii) {
	reportFile << " " ;
      }
      reportFile << title ;
      reportFile << "\tValue\t\tStd err\t\tt-test\tRobust Std err\t\tRobust t-test" << endl ;
      
      for (map<pair<patString,patString>,pair<patReal,patReal> >::iterator i = 
	     covarianceRandomCoef.begin() ;
	   i != covarianceRandomCoef.end() ;
	   ++i) {
	
	for (long ii = i->first.first.size() ; 
	     ii <= width ; 
	     ++ii) {
	  reportFile << " " ;
	}
	reportFile << i->first.first << " " ;
	for (long ii = i->first.second.size() ; 
	     ii <= width ; 
	     ++ii) {
	  reportFile << " " ;
	}
	reportFile << i->first.second << " " ;
	
	reportFile << theNumber.formatParameters(i->second.first) << '\t' ;
	if (relevantCovariance[i->first]) {
	  reportFile << theNumber.formatParameters(i->second.second) << '\t' 
		     << theNumber.formatTTests(i->second.first/i->second.second) ;
	}
	else {
	  reportFile << "fixed" ;
	}
	reportFile << endl ;
      }
    }    

  }
  reportFile << endl ;
  
  reportFile << "Utility functions" << endl ;
  reportFile << "*****************" << endl ;
  unsigned long J = getNbrAlternatives() ;
  for (unsigned long alt = 0 ; alt < J ; ++alt) {
    unsigned long userId = getAltId(alt,err) ;
    reportFile << userId << '\t' ;
    reportFile << getAltName(userId,err) << '\t' ;
    reportFile << getAvailName(userId,err) << '\t' ;
    patUtilFunction* util = 
      patModelSpec::the()->getUtilFormula(userId,err) ;
    if (util->begin() == util->end()) {
      reportFile << "$NONE" ;
    }
    else {
      for (list<patUtilTerm>::iterator ii = util->begin() ;
	   ii != util->end() ;
	   ++ii) {
	if (ii != util->begin()) {
	  reportFile << " + " ;
	}
	if (ii->random) {
	  reportFile << ii->randomParameter->getOperatorName() << " * " << ii->x;
	}
	else {
	  reportFile << ii->beta << " * " << ii->x ;
	}
      }
    }
    map<unsigned long, patArithNode*>::iterator found =  
      nonLinearUtilities.find(userId) ;
    if (found != nonLinearUtilities.end()) {
      reportFile << " + " << *(found->second) ;
    }
    reportFile << endl ;
  }
    
  reportFile << endl ;
    

  if (applySnpTransform()) {
    reportFile << "Seminonparametric transform:" << endl ;
    reportFile << "***************************" << endl ;
    reportFile << "Base parameter: " << snpBaseParameter << endl ;
    reportFile << "List of terms:" << endl ;
    reportFile << "Term of order\tCoefficient" << endl ;
    reportFile << "-------------\t-----------" << endl ;
    for (list<pair<unsigned short,patString> >::iterator iter = 
	   listOfSnpTerms.begin() ;
	 iter != listOfSnpTerms.end() ;
	 ++iter) {
      reportFile << iter->first << '\t' << '\t' << iter->second << endl ;
    }
  }

    
  //    reportFile << setiosflags(ios::showpos) ;
  reportFile << endl ;
  reportFile << "Correlation of coefficients" << endl ;
  reportFile << "***************************" << endl ;

  nCols = 10 ;
  patOutputTable tableCorr(nCols,patTRUE) ;
    
  vector<patString> aCorr(nCols) ;
    
  aCorr[0] = "Coeff1" ;
  aCorr[1] = "Coeff2" ;
  aCorr[2] = "Covariance" ;
  aCorr[3] = "Correlation" ;
  aCorr[4] = "t-test" ;
  aCorr[5] = "" ;    // column for the * if non significant
  if (estimationResults.isRobustVarCovarAvailable) {
    aCorr[6] = "Rob. covar." ;
    aCorr[7] = "Rob. correl." ;
    aCorr[8] = "Rob. t-test" ;
    aCorr[9] = "" ;    // column for the * if non significant
  }
  tableCorr.appendRow(aCorr,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
    
  vector<patString> corrUnd(nCols) ;
  for (unsigned short i = 0 ; i < nCols ; ++i) {
    corrUnd[i] = patString(aCorr[i].size(),'-') ;
  }
  tableCorr.appendRow(corrUnd,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
    
    
    
  for (std::set<patCorrelation,patCompareCorrelation>::iterator i = 
	 correlation.begin() ;
       i != correlation.end() ;
       ++i) { // loop on corr entries
    fill(aCorr.begin(),aCorr.end(),"") ;
    aCorr[0] = i->firstCoef ;
    aCorr[1] = i->secondCoef ;
    aCorr[2] = theNumber.formatParameters(i->covariance) ;
    aCorr[3] = theNumber.formatParameters(i->correlation) ; 
    aCorr[4] = theNumber.formatTTests(i->ttest) ;
    if (patAbs(i->ttest) < patParameters::the()->getgevTtestThreshold() ||
	!isfinite(i->ttest)) {
      aCorr[5] =  patParameters::the()->getgevWarningSign() ;
    }
    if (estimationResults.isRobustVarCovarAvailable) {
      aCorr[6] = theNumber.formatParameters(i->robust_covariance) ;
      aCorr[7] = theNumber.formatParameters(i->robust_correlation) ;
      aCorr[8] = theNumber.formatTTests(i->robust_ttest) ;
      if (patAbs(i->robust_ttest) < patParameters::the()->getgevTtestThreshold() ||
	  !isfinite(i->robust_ttest)) {
	aCorr[9] = patParameters::the()->getgevWarningSign() ;
      }
    }
    tableCorr.appendRow(aCorr,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
  tableCorr.computeColWidth() ;
  reportFile << tableCorr << endl ;
    
  // Constraints
    
  patListProblemLinearConstraint eqCons = 
    patModelSpec::the()->getLinearEqualityConstraints(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return  ;
  }
  patListProblemLinearConstraint ineqCons = 
    patModelSpec::the()->getLinearInequalityConstraints(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }      
  
  if (!eqCons.empty() || !ineqCons.empty()) {
    reportFile << endl ;
    reportFile << "User defined linear constraints" << endl ;
    reportFile << "************************" << endl ;
  }
  for (patListProblemLinearConstraint::iterator iter = eqCons.begin() ;
       iter != eqCons.end() ;
       ++iter) {     
    reportFile << patModelSpec::the()->printEqConstraint(*iter) << endl ;
  }
  for (patListProblemLinearConstraint::iterator iter = ineqCons.begin() ;
       iter != ineqCons.end() ;
       ++iter) {     
    reportFile << patModelSpec::the()->printIneqConstraint(*iter) << endl ;
  }
  if (equalityConstraints != NULL) {
    reportFile << "Nonlinear equality constraints" << endl ;
    reportFile << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl ;
    for (patListNonLinearConstraints::iterator i = equalityConstraints->begin() ;
	 i != equalityConstraints->end() ;
	 ++i) {
      reportFile << *i << "=0" << endl ;
      
    }
  }
  if (inequalityConstraints != NULL) {
    reportFile << "Nonlinear inequality constraints" << endl ;
    reportFile << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl ;
    for (patListNonLinearConstraints::iterator i = inequalityConstraints->begin() ;
	 i != inequalityConstraints->end() ;
	 ++i) {
      reportFile << *i << "=0" << endl ;
      
    }
  }


  // Eigen vector

  reportFile << "Smallest singular value of the hessian: " << estimationResults.smallestSingularValue << endl << endl ;
  if (!estimationResults.eigenVectors.empty()) {
    reportFile << endl ;
    reportFile << "Unidentifiable model" << endl ;
    reportFile << "********************" << endl ;
    //    reportFile << "Norm of A*z = " << estimationResults.Az << endl ;
    reportFile << "The log likelihood is (almost) flat along the following combination of parameters" << endl ;

    patReal threshold = patParameters::the()->getgevSingularValueThreshold() ;
    patIterator<patBetaLikeParameter>* theParamIter = createAllParametersIterator() ;
    
    for(map<patReal,patVariables>::iterator iter = 
	  estimationResults.eigenVectors.begin() ;
	iter != estimationResults.eigenVectors.end() ;
	++iter) {
      reportFile << "Sing. value = " << iter->first  << endl ; 
      
      nCols = 3 ;
      patOutputTable tableSing(nCols,patTRUE) ;
      vector<patString> aRow(nCols) ;
      
      vector<patBoolean> printed(iter->second.size(),patFALSE) ;
      for (theParamIter->first() ;
	   !theParamIter->isDone() ;
	   theParamIter->next()) {
	patBetaLikeParameter bb = theParamIter->currentItem() ;
	if (!bb.isFixed && !bb.hasDiscreteDistribution) {
	  unsigned long j = bb.index ;
	  if (patAbs(iter->second[j]) >= threshold) {
	    aRow[0] = theNumber.formatParameters(iter->second[j])  ;
	    aRow[1] = "*" ;
	    aRow[2] = bb.name ;
	    tableSing.appendRow(aRow,err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	    printed[j] = patTRUE ;
	  }
	}
	  
      }
      for (int j = 0 ; j < iter->second.size() ; ++j) {
	if (patAbs(iter->second[j]) >= threshold && !printed[j]) {
	  stringstream str ;
	  str << "Param[" << j << "]" ;
	  aRow[0] = theNumber.formatParameters(iter->second[j])  ;
	  aRow[1] = "*" ;
	  aRow[2] = str.str() ;
	  tableSing.appendRow(aRow,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return ;
	  }
	}
      }
      tableSing.computeColWidth() ;
      reportFile << tableSing << endl ;
    }
    
    DEBUG_MESSAGE("About to delete the iterator") ;
    
    
  }

    DEBUG_MESSAGE("Report file done") ;
    
    reportFile.close() ;
    patOutputFiles::the()->addCriticalFile(fileName,"Estimation results in text format");

  // Summary file
  //      Write one line in the summary file containing:
  //      0) The date and time
  //      1) The model name 
  //      2) The report file name
  //      3) The final log likelihood
  //      4) The sample size
  //      5) The run time of the algorithm
  //      6) The estimated value and t-test of parameters in the summary.lis file
  //      7) The exclusion condition

  //  DEBUG_MESSAGE("--> HERE 5: runtime = " << estimationResults.runTime) ;
  patString sumFileName = patParameters::the()->getgevSummaryFile() ;
  patBoolean headers = !patFileExists()(sumFileName) ;
  ofstream sumFile(sumFileName.c_str(),ios::out|ios::app) ;
  if (headers) {
    sumFile << "<html>" << endl ;
    sumFile << "<head>" << endl ;
    sumFile << "<script src=\"http://transp-or.epfl.ch/biogeme/sorttable.js\"></script>" << endl ;
    sumFile << "<meta http-equiv=\"content-type\" content=\"text/html;charset=utf-8\">" << endl ;
    sumFile << "<title>Report from " << patVersion::the()->getVersionInfoDate() << "</title>" << endl ;
    sumFile << "<meta name=\"keywords\" content=\"biogeme, discrete choice, random utility\">" << endl ;
    sumFile << "<meta name=\"description\" content=\"Report from " << patVersion::the()->getVersionInfoDate() << "\">" << endl ;
    sumFile << "<meta name=\"author\" content=\"Michel Bierlaire\">" << endl ;
    sumFile << "</head>" << endl ;
    sumFile << "<style>" << endl ;
    sumFile << "<!--table" << endl ;
    sumFile << ".biostyle" << endl ;
    sumFile << "	{font-size:10.0pt;" << endl ;
    sumFile << "	font-weight:400;" << endl ;
    sumFile << "	font-style:normal;" << endl ;
    sumFile << "	font-family:Courier;}" << endl ;
    sumFile << "-->" << endl ;
    sumFile << "</style>" << endl ;
    sumFile << "" << endl ;
    sumFile << "<body bgcolor=\"#ffffff\">" << endl ;
    
    sumFile << "<h1>" << patVersion::the()->getVersionInfoDate() << "</h1>" << endl ;
    sumFile << "<h2>" << patVersion::the()->getVersionInfoAuthor() << "</h2>" << endl ;
    
    sumFile << "<p>This file has automatically been generated.</p>" << endl ;
    sumFile << "<p> " << now.getTimeString(patTsfFULL) << "</p>" << endl ;
    sumFile << endl ;
    
    sumFile << "<p>" << endl ;
    sumFile << "<p><font size='+1' color='blue'>Tip: click on the columns headers to sort a table </font> [<a href='http://www.kryogenix.org/code/browser/sorttable/' target='_blank'>Credits</a>]</p>" << endl ;
    sumFile << "<table border=\"1\" class=\"sortable\">" << endl ;
    sumFile << "<tr class=biostyle >" ;
    sumFile << "<th align=left>Time</th>" ; 
    sumFile << "<th align=left>Model</th>" ; 
    sumFile << "<th align=left>Report</th>" ; 
    sumFile << "<th align=left>Loglike</th>" ; 
    sumFile << "<th align=left>Rho bar squared</th>" ; 
    sumFile << "<th align=left>Observations</th>" ; 
    sumFile << "<th align=left>Individuals</th>" ; 
    sumFile << "<th align=left>Run time</th>" ; 
    for (vector<patString>::iterator i = summaryParameters.begin() ;
	 i != summaryParameters.end() ;
	 ++i) {
      sumFile << "<th align=left>" << *i << "</th>" ; 
      sumFile << "<th align=left>(t-test 0)</th>" ; 
      sumFile << "<th align=left>(robust t-test 0)</th>" ; 
    }
    sumFile << "<th align=left>Exclude</th>" ; 
    sumFile << "</tr>" ; 

  }
  now.setTimeOfDay() ;
  sumFile << "<tr class=biostyle >" ;
  sumFile << "<td>" << now.getTimeString(patTsfFULL) << "</td>"  ;
  sumFile << "<td>" << patFileNames::the()->getModelName() << "</td>"  ;
  sumFile << "<td>" << patFileNames::the()->getHtmlFile(err) << "</td>"  ;
  sumFile << "<td>" 
	  << theNumber.formatStats(estimationResults.loglikelihood) << "</td>" ;
  patReal rhoBarSquare = 1.0 - ((estimationResults.loglikelihood-patReal(getNbrNonFixedParameters())) / estimationResults.nullLoglikelihood) ;
  sumFile << "<td>"
	  << theNumber.formatStats(rhoBarSquare) 
	  << "</td>" << endl ;
  sumFile << "<td>" << estimationResults.numberOfObservations << "</td>" ;
  sumFile << "<td>" << estimationResults.numberOfIndividuals << "</td>" ;
  //  DEBUG_MESSAGE("--> HERE 6: runtime = " << estimationResults.runTime) ;
  sumFile << "<td>" << estimationResults.runTime << "</td>" ;
  //  sumFile << setprecision(7) << setiosflags(ios::scientific) ;  
  for (vector<patString>::iterator i = summaryParameters.begin() ;
       i != summaryParameters.end() ;
       ++i) {
    patBoolean found ;
    patBetaLikeParameter beta = getParameter(*i,&found) ;
    if (found) {
      if (beta.hasDiscreteDistribution) {
	sumFile << "<td>-dist.-</td>" ;
	sumFile << "<td align=center>-</td>" ;
      }
      else {
	sumFile << "<td>" 
		<< theNumber.formatParameters(beta.estimated) << "</td>" ; 
	if (beta.isFixed) {
	  sumFile << "<td>-fixed-</td>" ;
	  sumFile << "<td>-fixed-</td>" ;
	}
	else {
	  if (estimationResults.isVarCovarAvailable) {
	    patReal ttest = beta.estimated  / stdErr[beta.index] ; 
	    sumFile << "<td>" 
		    << theNumber.formatTTests(ttest) << "</td>" ;
	  }
	  else {
	    sumFile << "<td align=center>-</td>";
	  }
	  if (estimationResults.isRobustVarCovarAvailable) {
	    patReal rttest = beta.estimated  / robustStdErr[beta.index] ; 
	    sumFile << "<td>" 
		    << theNumber.formatTTests(rttest) << "</td>" ;
	  }
	  else {
	    sumFile << "<td align=center>-</td>";
	  }
	  
	}
      }
    }
    else {
      sumFile << "<td align=center>-</td>" << '\t';
      sumFile << "<td align=center>-</td>" << '\t';
    }
  }

  patArithNode* excl = getExcludeExpr(err) ;
  if (err == NULL && excl !=  NULL) {
    sumFile << "<td>" << excl->getExpression(err) << "</td>"  ;
  }
  else {
    sumFile << "<td></td>" ;
  }
  sumFile << "</tr>" ;
  sumFile.close() ;
  patOutputFiles::the()->addUsefulFile(sumFileName,"Summary file in HTML format");
  
  patString alogitFileName = patFileNames::the()->getALogitFile(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  ofstream alogitFile(alogitFileName.c_str()) ;
  now.setTimeOfDay() ;
  alogitFile << "Biogeme " << patFileNames::the()->getModelName() << endl ;
  alogitFile << "This file [" << alogitFileName << "] has automatically been generated." << endl ;
  alogitFile << now.getTimeString(patTsfFULL) << endl ;
  alogitFile << patVersion::the()->getCopyright() << endl ;
  alogitFile << patVersion::the()->getVersionInfoDate() << endl ;
  alogitFile << patVersion::the()->getVersionInfoAuthor() << endl ;
  for (list<patString>::iterator i = modelDescription.begin() ;
       i != modelDescription.end() ;
       ++i) {
    alogitFile << "   " << *i << endl ;
  }
  alogitFile << "END" << endl ;
  patIterator<patBetaLikeParameter>* paramIter = 
    createAllParametersIterator() ;
  
  for (paramIter->first() ;
       !paramIter->isDone() ;
       paramIter->next()) {
    patBetaLikeParameter theParam = paramIter->currentItem() ;
    if (theParam.index != patBadId) {
      alogitFile << theParam.index << '\t' ;
      alogitFile << theParam.name << '\t' ;
      if (theParam.isFixed) {
	alogitFile << "T" << '\t' ;
      }
      else {
	alogitFile << "F" << '\t' ;
      }
      alogitFile << theParam.estimated << '\t' ;
      if (theParam.isFixed) {
	alogitFile << "99999" << '\t' ;
      }
      else if (theParam.index != patBadId) {
	if (estimationResults.isRobustVarCovarAvailable) {
	  alogitFile << robustStdErr[theParam.index] ;
	}
	else   if (estimationResults.isVarCovarAvailable) {
	  alogitFile << stdErr[theParam.index] ;
	}
	else {
	  alogitFile << "XXX" ;
	}
      }
      alogitFile << endl ;
    }
  }

  alogitFile << "-1" << endl ;

  alogitFile << estimationResults.numberOfObservations << '\t'
	     << estimationResults.nullLoglikelihood << '\t'
	     << estimationResults.cteLikelihood << '\t' 
	     << estimationResults.loglikelihood << endl
	     << setw(4) << estimationResults.iterations
	     << "   0" << ' ' 
	     << now.getTimeString(patAlogit) << endl ;

  if (estimationResults.isRobustVarCovarAvailable) {
    estimationResults.computeCorrelation() ;
    short colCounter = 1 ;
    unsigned long n = estimationResults.robustCorrelation->nRows() ;
    for (unsigned long k = 0 ; k < n ; ++k) {
      for(unsigned j = 0 ; j < k ; ++j) {
	alogitFile << setw(7) << long(100000 *  (*estimationResults.robustCorrelation)[j][k]) ;
	if (colCounter == 10) {
	  alogitFile << endl ;
	  colCounter = 1 ;
	}
	else {
	  ++colCounter ;
	}
      }
    }
  }
  

  alogitFile.close() ;
  patOutputFiles::the()->addUsefulFile(alogitFileName,"Estimation results in ALogit format");

}


void patModelSpec::writePythonResults(patPythonResults* pythonRes, 
				      patError*& err) {

  if (pythonRes == NULL) {
    WARNING("No database to store results for Python") ;
    return ;
  }
  stringstream str ;

  patAbsTime now ;
  now.setTimeOfDay() ;

  pythonRes->timeStamp = now.getTimeString(patTsfFULL) ;

  pythonRes->version = patVersion::the()->getVersionInfoDate() ;

  str << " " ; 
  for (list<patString>::iterator i = modelDescription.begin() ;
       i != modelDescription.end() ;
       ++i) {
    str << *i << endl ;
  }
  pythonRes->description = str.str() ;
  str.str(""); 

  if (isMixedLogit()) {
    str << "Mixed " ;

  }
  str << modelTypeName()  ;
  pythonRes->model = str.str() ;
  str.str(""); 

  if (isMixedLogit()) {
    
    if (estimationResults.halton) {
      str << "Halton " ;
    }
    else if (estimationResults.hessTrain) {
      str << "Hess-Train " ;
    }
    else {
      str << "Random" ;
    }

    pythonRes->drawsType = str.str() ;
    str.str("") ;

    pythonRes->numberOfDraws = getNumberOfDraws() ;
  }
    
  pythonRes->numberOfParameters = getNbrNonFixedParameters() ;

  pythonRes->numberOfObservations =  estimationResults.numberOfObservations  ;
  pythonRes->numberOfIndividuals = estimationResults.numberOfIndividuals ;

  pythonRes->nullLoglikelihood = estimationResults.nullLoglikelihood ;
  pythonRes->initLoglikelihood = estimationResults.initLoglikelihood ;
  pythonRes->finalLoglikelihood = estimationResults.loglikelihood ; 
  pythonRes->likelihoodRatioTest =  -2.0 * (estimationResults.nullLoglikelihood - estimationResults.loglikelihood) ;
  pythonRes->rhoSquare = 1.0 - (estimationResults.loglikelihood / estimationResults.nullLoglikelihood) ;

  pythonRes->rhoBarSquare = 1.0 - ((estimationResults.loglikelihood-getNbrNonFixedParameters()) / estimationResults.nullLoglikelihood) ;

  pythonRes->finalGradientNorm = estimationResults.gradientNorm ;

  if (patParameters::the()->getgevVarCovarFromBHHH() == 0) {
    if (patModelSpec::the()->isSimpleMnlModel() && patParameters::the()->getBTRExactHessian()) {
      str << "from analytical hessian" << endl ;
    }
    else {
      str << "from finite difference hessian" ;
      
    }
  }
  else {
    str << "from BHHH matrix"  ;
  }

  pythonRes->varianceCovariance = str.str() ;
  str.str("") ;

  if (estimationResults.varCovarMatrix == NULL) {
    estimationResults.isVarCovarAvailable = patFALSE ;
  }
  if (estimationResults.robustVarCovarMatrix == NULL) {
    estimationResults.isRobustVarCovarAvailable = patFALSE ;
  }



  patVariables stdErr ;
  if (estimationResults.isVarCovarAvailable) {
    patMyMatrix* varCovar(estimationResults.varCovarMatrix) ;
    DEBUG_MESSAGE("Var-covar is " << varCovar->nRows()
		  << "x" 
		  << varCovar->nCols()) ;
    for (unsigned long i = 0 ; i < varCovar->nRows() ; ++i) {
      if ((*varCovar)[i][i] < 0.0) {
	stdErr.push_back(patMaxReal) ; 
      }
      else {
	stdErr.push_back(sqrt((*varCovar)[i][i])) ; 
      }
    }
  }
  patVariables robustStdErr ;
  if (estimationResults.isRobustVarCovarAvailable) {
    patMyMatrix* robustVarCovar(estimationResults.robustVarCovarMatrix) ;
    for (unsigned long i = 0 ; i < robustVarCovar->nRows() ; ++i) {
      if ((*robustVarCovar)[i][i] < 0) {
	robustStdErr.push_back(patMaxReal) ;
      }
      else {
	robustStdErr.push_back(sqrt((*robustVarCovar)[i][i])) ; 
      }
    }
  }

  pythonRes->totalNumberOfParameters = 0 ;
  patIterator<patBetaLikeParameter>* theIterator =
    createAllParametersIterator() ;
  
  for (theIterator->first() ;
       !theIterator->isDone() ;
       theIterator->next()) { // loop on all parameters
    
    
    patBetaLikeParameter bb = theIterator->currentItem() ;
    
    if (!bb.hasDiscreteDistribution) {
      
      pythonRes->paramIndex[bb.name] = pythonRes->paramNames.size() ;
      pythonRes->paramNames.push_back(bb.name) ;


      pythonRes->estimates.push_back(bb.estimated) ;
      
      if (bb.isFixed) {
	pythonRes->fixed.push_back(1) ;
      }
       else {
 	pythonRes->fixed.push_back(0) ;
 	if (bb.hasDiscreteDistribution) {
 	  pythonRes->distributed.push_back(1) ;
 	}
 	else {
 	  pythonRes->distributed.push_back(0) ;
 	  if (estimationResults.isVarCovarAvailable) {
 	    patReal ttest =  bb.estimated  / stdErr[bb.index] ;
 	    pythonRes->tTest.push_back(ttest) ; 
 	    pythonRes->stdErr.push_back(stdErr[bb.index]) ;
   
 	    pythonRes->pValue.push_back(patPValue(patAbs(ttest),err)) ;
 	    if (err != NULL) {
 	      WARNING(err->describe()) ;
 	      return ;
 	    }
 	  }
 	  if (estimationResults.isRobustVarCovarAvailable) {
 	    patReal rttest = bb.estimated  / robustStdErr[bb.index] ;
 	    pythonRes->tTestRobust.push_back(rttest) ; 
 	    pythonRes->stdErrRobust.push_back(robustStdErr[bb.index]) ;
     
 	    pythonRes->pValueRobust.push_back(patPValue(patAbs(rttest),err)) ;
 	    if (err != NULL) {
 	      WARNING(err->describe()) ;
 	      return ;
 	    }
 	  }
	}
       }
    }
  }

  pythonRes->totalNumberOfParameters = pythonRes->paramNames.size() ;
//   pythonRes->estimates = new patReal[estimates.size()] ;
//   for (unsigned int i = 0 ; i < estimates.size() ; ++i) {
//     pythonRes->estimates[i] = estimates[i] ;
//   }





}
void patModelSpec::writeHtml(patString fileName, patError*& err) {
  DEBUG_MESSAGE("Write " << fileName) ;
  ofstream htmlFile(fileName.c_str()) ;
  patAbsTime now ;
  now.setTimeOfDay() ;

  htmlFile << "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">" << endl ;
    htmlFile << "" << endl ;
  htmlFile << "<html>" << endl ;
  htmlFile << "<head>" << endl ;
  htmlFile << "<script src=\"http://transp-or.epfl.ch/biogeme/sorttable.js\"></script>" << endl ;
  htmlFile << "<meta http-equiv=\"content-type\" content=\"text/html;charset=utf-8\">" << endl ;
  htmlFile << "<title>"<< fileName <<" - Report from " << patVersion::the()->getVersionInfoDate() << "</title>" << endl ;
  htmlFile << "<meta name=\"keywords\" content=\"biogeme, discrete choice, random utility\">" << endl ;
  htmlFile << "<meta name=\"description\" content=\"Report from " << patVersion::the()->getVersionInfoDate() << "\">" << endl ;
  htmlFile << "<meta name=\"author\" content=\"Michel Bierlaire\">" << endl ;
  htmlFile << "<style type=text/css>" << endl ;
  htmlFile << "<!--table" << endl ;
  htmlFile << ".biostyle" << endl ;
  htmlFile << "	{font-size:10.0pt;" << endl ;
  htmlFile << "	font-weight:400;" << endl ;
  htmlFile << "	font-style:normal;" << endl ;
  htmlFile << "	font-family:Courier;}" << endl ;
  htmlFile << ".boundstyle" << endl ;
  htmlFile << "	{font-size:10.0pt;" << endl ;
  htmlFile << "	font-weight:400;" << endl ;
  htmlFile << "	font-style:normal;" << endl ;
  htmlFile << "	font-family:Courier;" << endl ;
  htmlFile << "        color:red}" << endl ;
  htmlFile << "-->" << endl ;
  htmlFile << "</style>" << endl ;
  htmlFile << "</head>" << endl ;
  htmlFile << "" << endl ;
  htmlFile << "<body bgcolor=\"#ffffff\">" << endl ;

  htmlFile << "<h1>" << patVersion::the()->getVersionInfoDate() << "</h1>" << endl ;
  htmlFile << "<h2>" << patVersion::the()->getVersionInfoAuthor() << "</h2>" << endl ;

  htmlFile << "<p>This file has automatically been generated.</p>" << endl ;
  htmlFile << "<p> " << now.getTimeString(patTsfFULL) << "</p>" << endl ;
  htmlFile << "<p><font size='+1'>Tip: click on the columns headers to sort a table  [<a href='http://www.kryogenix.org/code/browser/sorttable/' target='_blank'>Credits</a>]</font></p>" << endl ;
  htmlFile << endl ;

  htmlFile << "<table>" << endl ;
  for (list<patString>::iterator i = modelDescription.begin() ;
       i != modelDescription.end() ;
       ++i) {
    htmlFile << "<tr><td>" << endl ;
    htmlFile << *i << endl ;
    htmlFile << "</td></tr>" << endl ;
  }
  htmlFile << "</table>" << endl ;
  htmlFile << endl ;



  htmlFile << "<table border=\"0\">" << endl ;
  htmlFile << "<tr class=biostyle ><td align=right><strong>Model</strong>:</td> <td>" ; 
  if (isMixedLogit()) {
    htmlFile << "Mixed " ;
  }
  htmlFile << modelTypeName() <<"</td></tr>" << endl ;
  if (isMixedLogit()) {
    htmlFile << "<tr class=biostyle ><td align=right><strong>Number of ";
    if (estimationResults.halton) {
      htmlFile << "Halton " ;
    }
    if (estimationResults.hessTrain) {
      htmlFile << "Hess-Train " ;
    }
    
    htmlFile << "draws</strong>:</td> <td>"<< getNumberOfDraws() <<"</td></tr>" << endl ;
  }
  htmlFile << "<tr class=biostyle><td align=right ><strong>Number of estimated parameters</strong>: </td> <td>"<< getNbrNonFixedParameters() <<"</td></tr>" << endl ;
  htmlFile << "<tr class=biostyle><td align=right ><strong>Number of " ;
  if (isAggregateObserved()) {
    htmlFile << "aggregate" ;
  }
  htmlFile << " observations</strong>: </td> <td>"<< estimationResults.numberOfObservations <<"</td></tr>" << endl ;
  htmlFile << "<tr class=biostyle><td align=right ><strong>Number of individuals</strong>: </td> <td>"<< estimationResults.numberOfIndividuals <<"</td></tr>" << endl ;
  htmlFile << "<tr class=biostyle><td align=right><strong>Null log likelihood</strong>:	</td> <td>"
	   <<  theNumber.formatStats(estimationResults.nullLoglikelihood)
	   <<"</td></tr>" << endl ;
  if (allAlternativesAlwaysAvail) {
    htmlFile << "<tr class=biostyle><td align=right><strong>Cte log likelihood</strong>:	</td> <td>"
	     <<  theNumber.formatStats(estimationResults.cteLikelihood)
	   <<"</td></tr>" << endl ;
  }
  htmlFile << "<tr class=biostyle><td align=right><strong>Init log likelihood</strong>:	</td> <td>"<< theNumber.formatStats(estimationResults.initLoglikelihood) 
	   <<"</td></tr>" << endl ;
  htmlFile << "<tr class=biostyle><td align=right><strong>Final log likelihood</strong>:	</td> <td>"
	   << theNumber.formatStats(estimationResults.loglikelihood) 
<<"</td></tr>" << endl ;
  htmlFile << "<tr class=biostyle><td align=right><strong>Likelihood ratio test</strong>:	</td> <td>" 
	   << theNumber.formatStats( -2.0 * (estimationResults.nullLoglikelihood - estimationResults.loglikelihood))
 <<"</td></tr>" << endl ;
  htmlFile << "<tr class=biostyle><td align=right><strong>Rho-square</strong>:		</td> <td>"
	   <<  theNumber.formatStats(1.0 - (estimationResults.loglikelihood / estimationResults.nullLoglikelihood))
	   <<"</td></tr>" << endl ;
  htmlFile << "<tr class=biostyle><td align=right><strong>Adjusted rho-square</strong>:		</td> <td>"
	   <<  theNumber.formatStats(1.0 - ((estimationResults.loglikelihood-getNbrNonFixedParameters()) / estimationResults.nullLoglikelihood)) <<"</td></tr>" << endl ;
  htmlFile << "<tr class=biostyle><td align=right><strong>Final gradient norm</strong>:	</td> <td>"
	   << theNumber.format(patTRUE,
			       patFALSE,
			       3,
			       estimationResults.gradientNorm) 
	   <<"</td></tr>" << endl ;
  htmlFile << "<tr class=biostyle><td align=right><strong>Diagnostic</strong>:	</td> <td>"
	   << estimationResults.diagnostic 
	   <<"</td></tr>" << endl ;
  if (estimationResults.iterations != 0) {
    htmlFile << "<tr class=biostyle><td align=right><strong>Iterations</strong>:	</td> <td>"
	     << estimationResults.iterations
	     <<"</td></tr>" << endl ;
  }
  htmlFile << "<tr class=biostyle><td align=right><strong>Run time</strong>:	</td> <td>"
	   << estimationResults.runTime
	   <<"</td></tr>" << endl ;
  htmlFile << "<tr class=biostyle><td align=right><strong>Variance-covariance</strong>:	</td> <td>" ; 
													   
  if (patParameters::the()->getgevVarCovarFromBHHH() == 0) {
    if (patModelSpec::the()->isSimpleMnlModel() && patParameters::the()->getBTRExactHessian()) {
      htmlFile << "from analytical hessian" << endl ;
    }
    else {
      htmlFile << "from finite difference hessian" ;
      
    }
  }
  else {
    htmlFile << "from BHHH matrix"  ;
  }
  htmlFile << "</td></tr>" << endl ;

  unsigned short nSampleFiles = patFileNames::the()->getNbrSampleFiles() ;
  if (nSampleFiles == 1) {
    htmlFile << "<tr class=biostyle><td align=right><strong>Sample file</strong>:	</td>" ;
  }
  else {
    htmlFile << "<tr class=biostyle><td align=right><strong>Sample files</strong>:	</td>" ;

  }
  htmlFile << "<td>" << patFileNames::the()->getSamFile(0,err) 
	   << "</td></tr>" ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  if (nSampleFiles > 1) {
    for (unsigned short i = 1 ; i < nSampleFiles ; ++i) {
      htmlFile << " <tr class=biostyle><td></td><td>" 
	       << patFileNames::the()->getSamFile(i,err) 
	       << "</td></tr>" << endl ;
    }
  }


  

  htmlFile << "</table>" << endl ;

  if (estimationResults.varCovarMatrix == NULL) {
    estimationResults.isVarCovarAvailable = patFALSE ;
  }
  if (estimationResults.robustVarCovarMatrix == NULL) {
    estimationResults.isRobustVarCovarAvailable = patFALSE ;
  }



  patVariables stdErr ;
  if (estimationResults.isVarCovarAvailable) {
    patMyMatrix* varCovar(estimationResults.varCovarMatrix) ;
    DEBUG_MESSAGE("Var-covar is " << varCovar->nRows()
		  << "x" 
		  << varCovar->nCols()) ;
    for (unsigned long i = 0 ; i < varCovar->nRows() ; ++i) {
      if ((*varCovar)[i][i] < 0.0) {
	stdErr.push_back(patMaxReal) ; 
      }
      else {
	stdErr.push_back(sqrt((*varCovar)[i][i])) ; 
      }
    }
  }
  patVariables robustStdErr ;
  if (estimationResults.isRobustVarCovarAvailable) {
    patMyMatrix* robustVarCovar(estimationResults.robustVarCovarMatrix) ;
    for (unsigned long i = 0 ; i < robustVarCovar->nRows() ; ++i) {
      if ((*robustVarCovar)[i][i] < 0) {
	robustStdErr.push_back(patMaxReal) ;
      }
      else {
	robustStdErr.push_back(sqrt((*robustVarCovar)[i][i])) ; 
      }
    }
  }
  htmlFile << "<h2>Utility parameters</h2>" << endl ;
  htmlFile << "<table border=\"1\" class=\"sortable\">" << endl ;
  htmlFile << "<tr class=biostyle>" ;
  // Column 1
  htmlFile << "<th>Name </th>" ;
  // Column 2
  htmlFile << "<th>Value		</th>" ;
  // Column 3
  htmlFile << "<th>Std err		</th>" ;
  // Column 4
  htmlFile << "<th>t-test</th>" ;
  if (patParameters::the()->getgevPrintPValue()) {
    // Column 5
    htmlFile << "<th>p-value</th>" ;
  }
  // Column 6
  htmlFile << "<th></th>" ;
  if (estimationResults.isRobustVarCovarAvailable) {
    // Column 7
    htmlFile << "<th>Robust Std err\t\t</th>" ;
    // Column 8
    htmlFile << "<th>Robust t-test</th>" ;
    if (patParameters::the()->getgevPrintPValue()) {
      // Column 9
      htmlFile << "<th>p-value</th>" ;
    }
    // Column 10
    htmlFile << "<th></th>" ;
  }
  htmlFile << "</tr>" ;
  htmlFile << endl ;

  for (map<patString,patBetaLikeParameter>::const_iterator i = 
	 betaParam.begin() ;
       i != betaParam.end() ;
       ++i) { // loop on beta like parameters
    
    if (!i->second.hasDiscreteDistribution) {
      if (patAbs(i->second.estimated - i->second.lowerBound) <= patEPSILON ||
	  patAbs(i->second.estimated - i->second.upperBound) <= patEPSILON ||
          i->second.isFixed) {
	htmlFile << "<tr class=boundstyle>" << endl ;
      }
      else {
	htmlFile << "<tr class=biostyle>" << endl ;
      }
      // Column 1
      htmlFile << "<td>" << i->second.name << "</td>" ;
      // Column 2
      htmlFile << "<td>" << theNumber.formatParameters(i->second.estimated) << "</td>" ;
      if (i->second.isFixed) {
	// Column 3
	htmlFile << "<td>fixed</td>" ;
	// Column 4
	htmlFile << "<td></td>" ;
	if (patParameters::the()->getgevPrintPValue()) {
	  // Column 5
	  htmlFile << "<td></td>" ;
	}
	// Column 6
	htmlFile << "<td></td>" ;
	// Column 7
	htmlFile << "<td></td>" ;
	// Column 8
	htmlFile << "<td></td>" ;
	if (patParameters::the()->getgevPrintPValue()) {
	  // Column 9
	  htmlFile << "<td></td>" ;
	}
	// Column 10
	htmlFile << "<td></td>" ;
      }
      else if (i->second.hasDiscreteDistribution) {
	htmlFile << "<td>distrib.</td>" ;
	// Column 4
	htmlFile << "<td></td>" ;
	if (patParameters::the()->getgevPrintPValue()) {
	  // Column 5
	  htmlFile << "<td></td>" ;
	}
	// Column 6
	htmlFile << "<td></td>" ;
	// Column 7
	htmlFile << "<td></td>" ;
	// Column 8
	htmlFile << "<td></td>" ;
	if (patParameters::the()->getgevPrintPValue()) {
	  // Column 9
	  htmlFile << "<td></td>" ;
	}
	// Column 10
	htmlFile << "<td></td>" ;
      }
      else {
	if (estimationResults.isVarCovarAvailable) {
	  patReal ttest = i->second.estimated  / stdErr[i->second.index] ; 
	// Column 3
	  htmlFile << "<td>" << theNumber.formatParameters(stdErr[i->second.index]) << "</td>" ;
	  // Column 4
	  htmlFile << "<td>" << theNumber.formatTTests(ttest) << "</td>";
	  if (patParameters::the()->getgevPrintPValue()) {
	    patReal pvalue = patPValue(patAbs(ttest),err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	    // Column 5
	    htmlFile << "<td>" << theNumber.formatTTests(pvalue) << "</td>" ;
	    
	  }
	  // Column 6
	  if (patAbs(ttest) < patParameters::the()->getgevTtestThreshold() ||
	      !isfinite(ttest)) {
	    htmlFile << "<td>" << patParameters::the()->getgevWarningSign() << "</td>" ;
	  }
	  else {
	    htmlFile << "<td></td>" ;
	  }
	  if (estimationResults.isRobustVarCovarAvailable) {
	    patReal rttest = i->second.estimated  / robustStdErr[i->second.index] ; 
	    
	  // Column 7
	    htmlFile << "<td>" << theNumber.formatParameters(robustStdErr[i->second.index]) << "</td>" ;
	  // Column 8
	    htmlFile << "<td>" << theNumber.formatTTests(rttest) << "</td>";
	    if (patParameters::the()->getgevPrintPValue()) {
	      patReal pvalue = patPValue(patAbs(rttest),err) ;
	      if (err != NULL) {
		WARNING(err->describe()) ;
		return ;
	      }
	    // Column 9
	      htmlFile << "<td>" << theNumber.formatTTests(pvalue) << "</td>" ;
	    }
	    if (patAbs(rttest) < patParameters::the()->getgevTtestThreshold() ||
		!isfinite(rttest)) {
	      
	      // Column 10
	      htmlFile << "<td>" << patParameters::the()->getgevWarningSign() << "</td>";
	    }
	    else {
	      htmlFile << "<td></td>" ;
	      
	    }
	  }
	  else {
	    // Column 7
	    htmlFile << "</td><td>" ;
	    // Column 8
	    if (patParameters::the()->getgevPrintPValue()) {
	      htmlFile << "</td><td>" ;
	    }
	    // Column 9
	    htmlFile << "</td><td>" ;
	    // Column 10
	    htmlFile << "</td><td>" ;
	  }
	}
	else {
	  htmlFile << "<td>var-covar unavailable</td>" ;
	}
      }
      htmlFile << "</tr>" << endl ;
    }
  }
  htmlFile << "</table>" << endl ;
    
  if (!mu.isFixed) {
    htmlFile << "<h2>Homogeneity parameter (mu)</h2>" << endl ;
    htmlFile << "<table border=\"1\" class=\"sortable\">" << endl ;

    htmlFile << "<tr class=biostyle>" ;
    // Column 1
    htmlFile << "<th>Value</th>" ;
    // Column 2
    htmlFile << "<th>Std err</th>" ;
    // Column 3
    htmlFile << "<th>t-test0</th>" ;
    // Column 4
    if (patParameters::the()->getgevPrintPValue()) {
      htmlFile << "<th>p-value</th>" ;
    }
    // Column 5
    htmlFile << "<th>t-test1</th>" ;
    // Column 6
    if (patParameters::the()->getgevPrintPValue()) {
      htmlFile << "<th>p-value</th>" ;
    }
    // Column 7
    htmlFile << "<th></th>" ;
    // Column 8
    htmlFile << "<th>Robust Std err</th>" ;
    // Column 9
    htmlFile << "<th>Robust t-test0</th>" ;
    // Column 10
    if (patParameters::the()->getgevPrintPValue()) {
      htmlFile << "<th>p-value</th>" ;
    }
    // Column 11
    htmlFile << "<th>Robust t-test1</th>" ;
    // Column 12
    if (patParameters::the()->getgevPrintPValue()) {
      htmlFile << "<th>p-value</th>" ;
    }
    // Column 13
    htmlFile << "<th></th>" ;
    htmlFile << "</tr>" << endl ;
    
    htmlFile << "<tr class=biostyle>" << endl ;
    // Column 1
    htmlFile << "<td>" << mu.estimated << "</td>" ;
    if (mu.isFixed) {
      // Column 1
      htmlFile << "<td>fixed</td>" << endl ;
      // Column 2
      htmlFile << "<td></td>" << endl ;
      // Column 3
      htmlFile << "<td></td>" << endl ;
      // Column 4
      if (patParameters::the()->getgevPrintPValue()) {
	htmlFile << "<td></td>" << endl ;
      }
      // Column 5
      htmlFile << "<td></td>" << endl ;
      // Column 6
      if (patParameters::the()->getgevPrintPValue()) {
	htmlFile << "<td></td>" << endl ;
      }
      // Column 7
      htmlFile << "<td></td>" << endl ;
      // Column 8
      htmlFile << "<td></td>" << endl ;
      // Column 9
      htmlFile << "<td></td>" << endl ;
      // Column 10
      if (patParameters::the()->getgevPrintPValue()) {
	htmlFile << "<td></td>" << endl ;
      }
      // Column 11
      htmlFile << "<td></td>" << endl ;
      // Column 12
      if (patParameters::the()->getgevPrintPValue()) {
	htmlFile << "<td></td>" << endl ;
      }
      // Column 13
      htmlFile << "<td></td>" << endl ;
    }
    else {
      if (estimationResults.isVarCovarAvailable) {

	patReal ttest0 = mu.estimated / stdErr[mu.index] ;
	patReal ttest1 = (mu.estimated-1.0) / stdErr[mu.index] ;
	// Column 2
	htmlFile << "<td>" << theNumber.formatParameters(stdErr[mu.index]) << "</td>" ;
	// Column 3
	htmlFile << "<td>" << theNumber.formatTTests(ttest0) << "</td>" ;
	// Column 4
	if (patParameters::the()->getgevPrintPValue()) {
	  
	  patReal pvalue = patPValue(patAbs(ttest0),err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return ;
	  }
	  htmlFile << "<td>" << theNumber.formatTTests(pvalue) << "</td>" ;
	}
	// Column 5
	htmlFile << "<td>" << theNumber.formatTTests(ttest1) << "</td>";
	// Column 6
	if (patParameters::the()->getgevPrintPValue()) {
	  
	  patReal pvalue = patPValue(patAbs(ttest1),err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return ;
	  }
	  htmlFile << "<td>" << theNumber.formatTTests(pvalue) << "</td>" ;
	}
	// Column 7
	if (patAbs(ttest0) < patParameters::the()->getgevTtestThreshold() ||
	    patAbs(ttest1) < patParameters::the()->getgevTtestThreshold() ||
	    !isfinite(ttest0)  ||
	    !isfinite(ttest1) ) {
	  htmlFile << "<td>" << patParameters::the()->getgevWarningSign() << "</td>" ;
	}
	else {
	  htmlFile << "<td></td>" ;
	}
	if (estimationResults.isRobustVarCovarAvailable) {
	  patReal rttest0 = mu.estimated / robustStdErr[mu.index] ;
	  patReal rttest1 = (mu.estimated-1.0) / robustStdErr[mu.index] ;
	  // Column 8
	  htmlFile << "<td>" << theNumber.formatParameters(robustStdErr[mu.index]) << "</td>" ;
	  // Column 9
	  htmlFile << "<td>" << theNumber.formatTTests(rttest0) << "</td>" ;
	  // Column 10
	  if (patParameters::the()->getgevPrintPValue()) {
	    patReal pvalue = patPValue(patAbs(rttest0),err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	    htmlFile << "<td>" << theNumber.formatTTests(pvalue) << "</td>" ;
	  }
	  // Column 11
	  htmlFile << "<td>" << rttest1 << "</td>" ;
	  // Column 12
	  if (patParameters::the()->getgevPrintPValue()) {
	    patReal pvalue = patPValue(patAbs(rttest1),err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	    htmlFile << "<td>" << theNumber.formatTTests(pvalue) << "</td>" ;
	  }
	  // Column 13
	  if (patAbs(rttest0) < patParameters::the()->getgevTtestThreshold() ||
	      patAbs(rttest1) < patParameters::the()->getgevTtestThreshold() ||
	      !isfinite(rttest0)  ||
	      !isfinite(rttest1) ) {
	    htmlFile << "<td>" << patParameters::the()->getgevWarningSign() << "</td>";
	  }
	  else {
	    htmlFile << "<td></td>" ;
	  }
	}
	else {
	  // Column 8
	  htmlFile << "<td></td>" ;
	  // Column 9
	  htmlFile << "<td></td>" ;
	  // Column 10
	  if (patParameters::the()->getgevPrintPValue()) {
	    htmlFile << "<td></td>" ;
	  }
	  // Column 11
	  htmlFile << "<td></td>" ;
	  // Column 12
	  if (patParameters::the()->getgevPrintPValue()) {
	    htmlFile << "<td></td>" ;
	  }
	  // Column 13
	  htmlFile << "<td></td>" ;
	}
      }
      else {
	htmlFile << "<td>var-covar unavailable</td>" ;
      }
      htmlFile << "</tr>" << endl ;
    }
    htmlFile << "</table>" << endl ;
  }




  patIterator<patBetaLikeParameter>* modelIter = createAllModelIterator() ;
  modelIter->first() ;
  if (!modelIter->isDone()) {

    htmlFile << "<h2>Model parameters</h2>" << endl ;
    htmlFile << "<table border=\"1\" class=\"sortable\">" << endl ;
      
    htmlFile << "<tr class=biostyle>" ;
    // Column 1
    htmlFile << "<th>Name</th>" ;
    // Column 2
    htmlFile << "<th>Value</th>" ;
    // Column 3
    htmlFile << "<th>Std err</th>" ;
    // Column 4
    htmlFile << "<th>t-test 0</th>" ;
    // Column 5
    if (patParameters::the()->getgevPrintPValue()) {
      htmlFile << "<th>p-value</th>" ;
    }
    // Column 6
    htmlFile << "<th>t-test 1</th>" ;
    // Column 7
    if (patParameters::the()->getgevPrintPValue()) {
      htmlFile << "<th>p-value</th>" ;
    }
    // Column 8
    htmlFile << "<th></th>" ;
    // Column 9
    htmlFile << "<th>Robust Std err</th>" ;
    // Column 10
    htmlFile << "<th>Robust t-test 0</th>" ;
    // Column 11
    if (patParameters::the()->getgevPrintPValue()) {
      htmlFile << "<th>p-value</th>" ;
    }
    // Column 12
    htmlFile << "<th>Robust t-test 1</th>" ;
    // Column 13
    if (patParameters::the()->getgevPrintPValue()) {
      htmlFile << "<th>p-value</th>" ;
    }
    // Column 14
    htmlFile << "<th></th>" ;
    htmlFile << "</tr>" << endl ;
      
    for (modelIter->first() ;
	 !modelIter->isDone() ;
	 modelIter->next()) {
      patBetaLikeParameter bb = modelIter->currentItem() ;
      if (patAbs(bb.estimated - bb.lowerBound) <= patEPSILON ||
	  patAbs(bb.estimated - bb.upperBound) <= patEPSILON ||
          bb.isFixed) {
	htmlFile << "<tr class=boundstyle>" << endl ;
      }
      else {
	htmlFile << "<tr class=biostyle>" << endl ;
      }
      // Column 1
      htmlFile << "<td>" << bb.name << "</td>" ;
      // Column 2
      htmlFile << "<td>" << theNumber.formatParameters(bb.estimated) << "</td>" ;
      if (bb.isFixed) {
	// Column 3
	htmlFile << "<td>fixed</td>" ;
	// Column 4
	htmlFile << "<td></td>" ;
	// Column 5
	if (patParameters::the()->getgevPrintPValue()) {
	  htmlFile << "<td></td>" ;
	}
	// Column 6
	htmlFile << "<td></td>" ;
	// Column 7
	if (patParameters::the()->getgevPrintPValue()) {
	  htmlFile << "<td></td>" ;
	}
	// Column 8
	htmlFile << "<td></td>" ;
	// Column 9
	htmlFile << "<td></td>" ;
	// Column 10
	htmlFile << "<td></td>" ;
	// Column 11
	if (patParameters::the()->getgevPrintPValue()) {
	  htmlFile << "<td></td>" ;
	}
	// Column 12
	htmlFile << "<td></td>" ;
	// Column 13
	if (patParameters::the()->getgevPrintPValue()) {
	  htmlFile << "<td></td>" ;
	}
	// Column 14
	htmlFile << "<td></td>" ;
      }
      else {
	if (estimationResults.isVarCovarAvailable) {
	  patReal ttest0 = bb.estimated  / stdErr[bb.index] ;
	  patReal ttest1 = (bb.estimated-1.0)  / stdErr[bb.index] ;
	  // Column 3
	  htmlFile << "<td>" << theNumber.formatParameters(stdErr[bb.index]) << "</td>" ;
	  // Column 4
	  htmlFile << "<td>" << theNumber.formatTTests(ttest0) << "</td>";
	  // Column 5
	  if (patParameters::the()->getgevPrintPValue()) {

	    patReal pvalue = patPValue(patAbs(ttest0),err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	    htmlFile << "<td>" << theNumber.formatTTests(pvalue) << "</td>" ;
	  }
	  // Column 6
	  htmlFile << "<td>" << theNumber.formatTTests(ttest1) << "</td>" ;
	  // Column 7
	  if (patParameters::the()->getgevPrintPValue()) {

	    patReal pvalue = patPValue(patAbs(ttest1),err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	    htmlFile << "<td>" << theNumber.formatTTests(pvalue) << "</td>" ;
	  }
	  // Column 8

	  if (patAbs(ttest0) < patParameters::the()->getgevTtestThreshold() ||
	      patAbs(ttest1) < patParameters::the()->getgevTtestThreshold() ||
	      !isfinite(ttest0) ||
	      !isfinite(ttest1)) {
	    htmlFile << "<td>" << patParameters::the()->getgevWarningSign() << "</td>";
	  }
	  else {
	    htmlFile << "<td></td>" ;
	  }
	  if (estimationResults.isRobustVarCovarAvailable) {
	    patReal rttest0 = bb.estimated  / robustStdErr[bb.index] ;
	    patReal rttest1 = (bb.estimated-1.0)  / robustStdErr[bb.index] ;
	    // Column 9
	    htmlFile << "<td>" << theNumber.formatParameters(robustStdErr[bb.index]) << "</td>" ;
	    // Column 10
	    htmlFile << "<td>" << theNumber.formatTTests(rttest0) << "</td>";
	    // Column 11
	    if (patParameters::the()->getgevPrintPValue()) {
	      
	      patReal pvalue = patPValue(patAbs(rttest0),err) ;
	      if (err != NULL) {
		WARNING(err->describe()) ;
		return ;
	      }
	      htmlFile << "<td>" << theNumber.formatTTests(pvalue) << "</td>" ;
	    }
	    // Column 12
	    htmlFile << "<td>" << theNumber.formatTTests(rttest1) << "</td>" ;

	    // Column 13
	    if (patParameters::the()->getgevPrintPValue()) {
	      
	      patReal pvalue = patPValue(patAbs(rttest1),err) ;
	      if (err != NULL) {
		WARNING(err->describe()) ;
		return ;
	      }
	      htmlFile << "<td>" << theNumber.formatTTests(pvalue) << "</td>" ;
	      
	    }
	    // Column 14
	    if (patAbs(rttest0) < patParameters::the()->getgevTtestThreshold() ||
		patAbs(rttest1) < patParameters::the()->getgevTtestThreshold() ||
		!isfinite(rttest0) ||
		!isfinite(rttest1)) {
	      htmlFile << "<td>" << patParameters::the()->getgevWarningSign() << "</td>";
	    }
	    else {
	      htmlFile << "<td></td>" ;
	    }
	  }
	  else {
	    // Column 10
	    htmlFile << "<td></td>" ;
	    // Column 11
	    if (patParameters::the()->getgevPrintPValue()) {
	      htmlFile << "<td></td>" ;
	    }
	    // Column 12
	    htmlFile << "<td></td>" ;
	    // Column 13
	    if (patParameters::the()->getgevPrintPValue()) {
	      htmlFile << "<td></td>" ;
	    }
	    // Column 14
	    htmlFile << "<td></td>" ;
	    // Column 15
	    htmlFile << "<td></td>" ;
	  }
	}
	else {
	  htmlFile << "<td>var-covar unavailable</td>" ;
	}
      }
      htmlFile << "</tr>" << endl ;
    }
    htmlFile << "</table>" << endl ;
  }
  patIterator<patBetaLikeParameter>* scaleIter = createScaleIterator() ;
  patBoolean oneScale(patFALSE) ;
  for (scaleIter->first() ;
       !scaleIter->isDone() ;
       scaleIter->next()) {
   patBetaLikeParameter bb = scaleIter->currentItem() ;
   if (!bb.isFixed) {
     oneScale = patTRUE ;
     break ;
   }    
  }

  if (oneScale) {
    
    htmlFile << "<h2>Scale parameters</h2>" << endl ;
    htmlFile << "<table border=\"1\" class=\"sortable\">" << endl ;
    htmlFile << "<tr class=biostyle>" ; 
    // Column 1
    htmlFile << "<th>Name</th>" ;
    // Column 2
    htmlFile << "<th>Value</th>" ;
    // Column 3
    htmlFile << "<th>Std err</th>" ;
    // Column 4
    htmlFile << "<th>t-test 1</th>" ;
    // Column 5
    if (patParameters::the()->getgevPrintPValue()) {
      htmlFile << "<th>p-value</th>" ;
    }
    // Column 6
    htmlFile << "<th></th>" ;
    // Column 7
    htmlFile << "<th>Robust Std err</th>" ;
    // Column 8
    htmlFile << "<th>Robust t-test 1</th>" ;
    // Column 9
    if (patParameters::the()->getgevPrintPValue()) {
      htmlFile << "<th>p-value</th>" ;
    }
    // Column 10
    htmlFile << "<th></th>" ;
    htmlFile << "</tr>" << endl ;
    
    for (scaleIter->first() ;
	 !scaleIter->isDone() ;
	 scaleIter->next()) {
      patBetaLikeParameter bb = scaleIter->currentItem() ;
      if (patAbs(bb.estimated - bb.lowerBound) <= patEPSILON ||
	  patAbs(bb.estimated - bb.upperBound) <= patEPSILON ||
          bb.isFixed) {
	htmlFile << "<tr class=boundstyle>" << endl ;
      }
      else {
	htmlFile << "<tr class=biostyle>" << endl ;
      }
      // Column 1
      htmlFile << "<td>" << bb.name << "</td>" ;
      // Column 2
      htmlFile << "<td>" << theNumber.formatParameters(bb.estimated) << "</td>" ;
      if (bb.isFixed) {
	// Column 3
	htmlFile << "<td>fixed</td>" ;
	// Column 4
	htmlFile << "<td></td>" ;
	// Column 5
	if (patParameters::the()->getgevPrintPValue()) {
	  htmlFile << "<td></td>" ;
	}
	// Column 6
	htmlFile << "<td></td>" ;
	// Column 7
	htmlFile << "<td></td>" ;
	// Column 8
	htmlFile << "<td></td>" ;
	// Column 9
	if (patParameters::the()->getgevPrintPValue()) {
	  htmlFile << "<td></td>" ;
	}
	// Column 10
	htmlFile << "<td></td>" ;
      }
      else {
	if (estimationResults.isVarCovarAvailable) {
	  patReal ttest1 = (bb.estimated-1.0)  / stdErr[bb.index] ;
	  // Column 3
	  htmlFile << "<td>" << theNumber.formatParameters(stdErr[bb.index]) << "</td>" ;
	  // Column 4
	  htmlFile << "<td>" << theNumber.formatTTests(ttest1)<< "</td>" ;
	  // Column 5
	  if (patParameters::the()->getgevPrintPValue()) {

	    patReal pvalue = patPValue(patAbs(ttest1),err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	    htmlFile << "<td>" << theNumber.formatTTests(pvalue) << "</td>" ;
	    
	  }
	  // Column 6
	  if (patAbs(ttest1) < patParameters::the()->getgevTtestThreshold() ||
	      !isfinite(ttest1)) {
	    htmlFile << "<td>" << patParameters::the()->getgevWarningSign()<< "</td>" ;
	  }
	  else {
	    htmlFile << "<td></td>" ;
	  }
	  if (estimationResults.isRobustVarCovarAvailable) {
	    patReal rttest1 = (bb.estimated-1.0)  / robustStdErr[bb.index] ;
	    // Column 7
	    htmlFile << "<td>" << theNumber.formatParameters(robustStdErr[bb.index]) << "</td>" ;
	    // Column 8
	    htmlFile << "<td>" << theNumber.formatTTests(rttest1) << "</td>";
	    // Column 9
	    if (patParameters::the()->getgevPrintPValue()) {
	      
	      patReal pvalue = patPValue(patAbs(rttest1),err) ;
	      if (err != NULL) {
		WARNING(err->describe()) ;
		return ;
	      }
	      htmlFile << "<td>" << theNumber.formatTTests(pvalue) << "</td>" ;
	    }
	    // Column 10
	    if (patAbs(rttest1) < patParameters::the()->getgevTtestThreshold() ||
		!isfinite(rttest1)) {
	      htmlFile << "<td>" << patParameters::the()->getgevWarningSign() << "</td>" ;
	    }
	    else {
	      htmlFile << "<td></td>" ;
	    }
	  }
	  else {
	    // Column 7
	    htmlFile << "<td></td>" ;
	    // Column 8
	    htmlFile << "<td></td>" ;
	    // Column 9
	    if (patParameters::the()->getgevPrintPValue()) {
	      htmlFile << "<td></td>" ;
	    }
	    // Column 10
	    htmlFile << "<td></td>" ;
	  }
	  
	}
	else {
	    // Column 3
	  htmlFile << "<td colspan='8'>var-covar unavailable</td>" ;
	}
      }
      htmlFile << "</tr>" << endl ;
    }
    htmlFile << "</table>" << endl ;
  }

  if (!isOL()) {
    htmlFile << "<h2>Utility functions</h2>" << endl ;
    htmlFile << "<table border=\"1\" class=\"sortable\">" << endl ;
    unsigned long J = getNbrAlternatives() ;
    htmlFile << "<tr class=biostyle>" ;
    // Column 1
    htmlFile << "<th>Id</th>" ;
    // Column 2
    htmlFile << "<th>Name</th>" ;
    // Column 3
    htmlFile << "<th>Availability</th>" ;
    // Column 4
    htmlFile << "<th>Specification</th>" ;
    htmlFile << "</tr>" ;
    for (unsigned long alt = 0 ; alt < J ; ++alt) {
      unsigned long userId = getAltId(alt,err) ;
      htmlFile << "<tr class=biostyle>" ;
      // Column 1
      htmlFile << "<td>" << userId << "</td>" ;
      // Column 2
      htmlFile << "<td>" << getAltName(userId,err) << "</td>" ;
      // Column 3
      htmlFile << "<td>" << getAvailName(userId,err) << "</td>" ;
      patUtilFunction* util = 
	patModelSpec::the()->getUtilFormula(userId,err) ;
      // Column 4
      htmlFile << "<td>" ;
      if (util->begin() == util->end()) {
	htmlFile << "$NONE" ;
      }
      else {
	for (list<patUtilTerm>::iterator ii = util->begin() ;
	     ii != util->end() ;
	     ++ii) {
	  if (ii != util->begin()) {
	    htmlFile << " + " ;
	  }
	  if (ii->random) {
	    htmlFile << ii->randomParameter->getOperatorName() << " * " << ii->x;
	  }
	  else {
	    htmlFile << ii->beta << " * " << ii->x ;
	  }
	}
      }
      map<unsigned long, patArithNode*>::iterator found =  
	nonLinearUtilities.find(userId) ;
      if (found != nonLinearUtilities.end()) {
	htmlFile << " + " << *(found->second) ;
      }
      htmlFile << "</td></tr>" ;
      htmlFile << endl ;
    }
    htmlFile << "</table>" << endl ;
    
    htmlFile << endl ;
    
    
    //    htmlFile << setiosflags(ios::showpos) ;
    htmlFile << endl ;
    
  }
  else {
    // Output for the ordinal logit
    htmlFile << "<h2>Difference of utility functions</h2>" << endl ;
    htmlFile << "<p  class=biostyle>" << endl ;
    unsigned long userId = getAltId(0,err) ;
    patUtilFunction* util = 
      patModelSpec::the()->getUtilFormula(userId,err) ;
    if (util->begin() == util->end()) {
      htmlFile << "$NONE" ;
    }
    else {
      for (list<patUtilTerm>::iterator ii = util->begin() ;
	     ii != util->end() ;
	     ++ii) {
	  if (ii != util->begin()) {
	    htmlFile << " + " ;
	  }
	  if (ii->random) {
	    htmlFile << ii->randomParameter->getOperatorName() << " * " << ii->x;
	  }
	  else {
	    htmlFile << ii->beta << " * " << ii->x ;
	  }
      }
    }
    map<unsigned long, patArithNode*>::iterator found =  
      nonLinearUtilities.find(userId) ;
    if (found != nonLinearUtilities.end()) {
      htmlFile << " + " << *(found->second) ;
    }

    htmlFile << " - ( " ; 
    
    userId = getAltId(1,err) ;
    util = patModelSpec::the()->getUtilFormula(userId,err) ;
    if (util->begin() == util->end()) {
      htmlFile << "$NONE" ;
    }
    else {
      for (list<patUtilTerm>::iterator ii = util->begin() ;
	     ii != util->end() ;
	     ++ii) {
	  if (ii != util->begin()) {
	    htmlFile << " + " ;
	  }
	  if (ii->random) {
	    htmlFile << ii->randomParameter->getOperatorName() << " * " << ii->x;
	  }
	  else {
	    htmlFile << ii->beta << " * " << ii->x ;
	  }
      }
    }
    found = nonLinearUtilities.find(userId) ;
    if (found != nonLinearUtilities.end()) {
      htmlFile << " + " << *(found->second) ;
    }

 

   
    htmlFile << " )</p>" ; 
    htmlFile << endl ;
    
    
    //    htmlFile << setiosflags(ios::showpos) ;
    htmlFile << endl ;
    
  }

  if (applySnpTransform()) {
    htmlFile << "<h2>Seminonparametric transform</h2>" << endl ;
    htmlFile << "<p>Base parameter: " << snpBaseParameter << "</p>" << endl ;
    htmlFile << "<p>List of terms:<p> " << endl ;
    htmlFile << "<table border=\"1\" class=\"sortable\">" << endl ;
    htmlFile << "<tr class=biostyle><th>Term of order </th><th>Coefficient</th></tr>" << endl ;
    for (list<pair<unsigned short,patString> >::iterator iter = 
	   listOfSnpTerms.begin() ;
	 iter != listOfSnpTerms.end() ;
	 ++iter) {
      htmlFile << "<tr class=biostyle><td>"
	       <<iter->first
	       << "</td><td>"
	       <<iter->second
	       << "</td></tr>" << endl ;
      }
    htmlFile << "</table>" << endl ;
  }
  
    if (!ratios.empty()) {
      htmlFile << "<h2>Ratio of parameters</h2>" << endl ;
      htmlFile << "<table border=\"1\" class=\"sortable\">" << endl ;
      htmlFile << "<tr class=biostyle><th>Name </th><th>Value		</th></tr>" << endl ;
      
      for (map<patString,pair<patString, patString> >::const_iterator i =
	   ratios.begin() ;
	 i != ratios.end() ;
	 ++i) {
      //    DEBUG_MESSAGE("Ratio of " << i->second.first << " and " << i->second.second) ;
      htmlFile << "<tr class=biostyle><td>" << endl ;
      htmlFile << i->first << '\t' ;
      patBoolean found ;
      patBetaLikeParameter num = getBeta(i->second.first,&found) ;
      if (!found) {
	htmlFile << "</td><td>" ;
	htmlFile << "Unknown parameter: " << i->second.first << endl ;
      }
      patBetaLikeParameter denom = getBeta(i->second.second,&found) ;
      if (!found) {
	htmlFile << "</td><td>" ;
	htmlFile << "Unknown parameter: " << i->second.second << endl ;
      }
      
      htmlFile << "</td><td>" ;
      htmlFile << theNumber.formatParameters(num.estimated / denom.estimated) ;
      htmlFile << "</td></tr>" << endl ;
    }
    htmlFile << "</table>" << endl ;
  }

  htmlFile << endl ;



  if (estimationResults.isVarCovarAvailable) { // varcovar avail
    
    patMyMatrix* varCovar = estimationResults.varCovarMatrix ;
    patMyMatrix* robustVarCovar = estimationResults.robustVarCovarMatrix ;
    
    map<patString,map<patString,patReal> > covarianceMatrix ;
    map<patString,map<patString,patReal> > correlationMatrix ;
    map<patString,map<patString,patReal> > robustCovarianceMatrix ;
    map<patString,map<patString,patReal> > robustCorrelationMatrix ;
    map<patString,patReal> aRowCovMat ;
    map<patString,patReal> aRowCorMat ;
    map<patString,patReal> aRowRobCovMat ;
    map<patString,patReal> aRowRobCorMat ;
  
    std::set<patCorrelation,patCompareCorrelation> correlation ;
    
    for (map<patString,patBetaLikeParameter>::const_iterator i = 
	   betaParam.begin() ;
	 i != betaParam.end() ;
	 ++i) {
      aRowCovMat.erase(aRowCovMat.begin(),aRowCovMat.end()) ;
      aRowCorMat.erase(aRowCorMat.begin(),aRowCorMat.end()) ;
      aRowRobCovMat.erase(aRowRobCovMat.begin(),aRowRobCovMat.end()) ;
      aRowRobCorMat.erase(aRowRobCorMat.begin(),aRowRobCorMat.end()) ;
      if (!i->second.isFixed && !i->second.hasDiscreteDistribution) {
	patReal varI = (*varCovar)[i->second.index][i->second.index] ;
	patReal robustVarI ;
	if (estimationResults.isRobustVarCovarAvailable) {
	  robustVarI = (*robustVarCovar)[i->second.index][i->second.index] ;
	}
	for (map<patString,patBetaLikeParameter>::const_iterator j = 
	       betaParam.begin() ;
	     j != betaParam.end() ;
	     ++j) {
	  if (!j->second.isFixed && !j->second.hasDiscreteDistribution) {
	    patReal varJ = (*varCovar)[j->second.index][j->second.index] ;
	    patReal robustVarJ ;
	    if (estimationResults.isRobustVarCovarAvailable) {
	      robustVarJ = (*robustVarCovar)[j->second.index][j->second.index] ;
	    }	    
	    if (i->second.id != j->second.id) {
	      patCorrelation c ;
	      if (i->second.name < j->second.name) {
		c.firstCoef = i->second.name ;
		c.secondCoef = j->second.name ;
	      }
	      else {
		c.firstCoef = j->second.name ;
		c.secondCoef = i->second.name ;
	      }
	      
	      c.covariance = (*varCovar)[i->second.index][j->second.index] ;
	      patReal tmp = varI * varJ ;
	      c.correlation = (tmp > 0) 
		? c.covariance / sqrt(tmp) 
		: 0.0 ;
	      tmp = varI + varJ - 2.0 * c.covariance ;
	      c.ttest = (tmp > 0)
		? (i->second.estimated - j->second.estimated) / sqrt(tmp) 
		: 0.0  ;
	      if (estimationResults.isRobustVarCovarAvailable) {
		c.robust_covariance = (*robustVarCovar)[i->second.index][j->second.index] ;
		patReal tmp = robustVarI * robustVarJ ;
		c.robust_correlation = (tmp > 0) 
		  ? c.robust_covariance / sqrt(tmp) 
		  : 0.0 ;
		tmp = robustVarI + robustVarJ - 2.0 * c.robust_covariance ;
		c.robust_ttest = (tmp > 0)
		  ? (i->second.estimated - j->second.estimated) / sqrt(tmp) 
		  : 0.0  ;
	      }
	      correlation.insert(c) ;
	      aRowCovMat[j->second.name] = c.covariance ;
	      aRowCorMat[j->second.name] = c.correlation ;
	      if (estimationResults.isRobustVarCovarAvailable) {
		aRowRobCovMat[j->second.name] = c.robust_covariance ;
		aRowRobCorMat[j->second.name] = c.robust_correlation ;
	      }
	    }
	    else {
	      aRowCovMat[j->second.name] = varJ ;
	      aRowCorMat[j->second.name] = 1.0 ;
	      if (estimationResults.isRobustVarCovarAvailable) {
		aRowRobCovMat[j->second.name] = robustVarJ ;
		aRowRobCorMat[j->second.name] = 1.0 ;	      
	      }
	    }
	  }
	}
	covarianceMatrix[i->second.name] = aRowCovMat ;
	correlationMatrix[i->second.name] = aRowCorMat ;
	if (estimationResults.isRobustVarCovarAvailable) {
	  robustCovarianceMatrix[i->second.name] = aRowRobCovMat ;
	  robustCorrelationMatrix[i->second.name] = aRowRobCorMat ;
	} 
      }
    }
  
    computeVarCovarOfRandomParameters(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }

    if (!varianceRandomCoef.empty()) { // varcovar

      htmlFile << "<h2>Variance of random coefficients</h2>" << endl ;
      
      htmlFile << "<table border=\"1\" class=\"sortable\">" << endl ;
      htmlFile << "<tr class=biostyle><th>Name</th><th>Value</th><th>Std err</th><th>t-test</th><th>Robust Std err</th><th>Robust t-test</th>" << endl ;
      
      for (map<patString,pair<patReal,patReal> >::iterator i = 
	     varianceRandomCoef.begin() ;
	   i != varianceRandomCoef.end() ;
	   ++i) {
      
	map<patString,pair<patRandomParameter,patArithRandom*> >::iterator 
	  found = randomParameters.find(i->first) ;
	if (found == randomParameters.end()) {
	  stringstream str ;
	  str << "Parameter " << i->first << " not found" ;
	  err = new patErrMiscError(str.str()) ;
	  WARNING(err->describe()) ;
	  return ;
	}
	patDistribType theType = found->second.second->getDistribution()  ;

	htmlFile << "<tr class=biostyle><td>" << endl ;
	
	htmlFile << i->first ;
	htmlFile << "</td><td>" ;

	patReal variance(0.0) ;
	patReal stdErr(0.0) ;
	patReal ttest(0.0) ;
	if (theType == NORMAL_DIST) {
	  variance = i->second.first ;
	  stdErr = i->second.second ;
	  ttest = i->second.first / i->second.second ;
	}
	else if (theType == UNIF_DIST) {
	  variance = i->second.first / 3.0 ;
	  stdErr = i->second.second / 3.0 ;
	  ttest = i->second.first / i->second.second ;
	}

	htmlFile << theNumber.formatParameters(variance)  ;
	htmlFile << "</td><td>" ;
	if (relevantVariance[i->first]) {
	  htmlFile << theNumber.formatParameters(stdErr) << "</td><td>" 
		   << theNumber.formatTTests(ttest) << "</td><td>" ;
	}
	else {
	  htmlFile << "fixed" << "</td><td></td><td>";
	}
	htmlFile << "</td></tr>" << endl ;
      }
      htmlFile << "</table>" << endl ;
    }

    if (!covarianceRandomCoef.empty()) {

      htmlFile << "<h2>Covariance of random coefficients</h2>" << endl ;
      htmlFile << "<table border=\"1\" class=\"sortable\">" << endl ;
      htmlFile << "<tr class=biostyle><th>Name1</th><th> Name2</th><th>Value</th><th>Std err</th><th>t-test</th><th>Robust Std err</th><th>Robust t-test</tr>" << endl ;
      
      for (map<pair<patString,patString>,pair<patReal,patReal> >::iterator i = 
	     covarianceRandomCoef.begin() ;
	   i != covarianceRandomCoef.end() ;
	   ++i) {
	
	htmlFile << "<tr class=biostyle><td>" << endl ;
	htmlFile << i->first.first << "</td><td>" ;
	htmlFile << i->first.second << "</td><td>" ;
	htmlFile << i->second.first << "</td><td>" ;
	if (relevantCovariance[i->first]) {
	  htmlFile << theNumber.formatParameters(i->second.second) << "</td><td>" 
		   << theNumber.formatParameters(i->second.first/i->second.second)  << "</td><td>" ;
	}
	else {
	  htmlFile << "fixed" << "</td><td></td><td>";
	}
	htmlFile << "</td></tr>" << endl ;
      }
      
      htmlFile << "</table>" << endl ;
      
    }
    
    if (patParameters::the()->getgevPrintVarCovarAsList()) {
      
      htmlFile << "<h2>Correlation of coefficients</h2>" << endl ;
      htmlFile << "<table border=\"1\" class=\"sortable\">" << endl ;
      if (patParameters::the()->getgevPrintPValue()) {
	htmlFile << "<tr class=biostyle><th>Coefficient1 </th><th>Coefficient2</th><th>Covariance</th><th>Correlation</th><th>t-test</th><th>p-value</th><th></th><th>Rob. cov.</th><th>Rob. corr.</th><th>Rob. t-test</th><th>p-value</th><th></th></tr>" << endl ;
      }
      else {
	htmlFile << "<tr class=biostyle><th>Coefficient1 </th><th>Coefficient2</th><th>Covariance</th><th>Correlation</th><th>t-test</th><th></th><th>Rob. cov.</th><th>Rob. corr.</th><th>Rob. t-test</th><th></th></tr>" << endl ;
      }
      
      for (std::set<patCorrelation,patCompareCorrelation>::iterator i = correlation.begin() ;
	   i != correlation.end() ;
	   ++i) {
	htmlFile << "<tr class=biostyle><td>" << endl ;
	htmlFile << i->firstCoef << '\t' ;
	htmlFile << "</td><td>" ;
	htmlFile << i->secondCoef ;
	htmlFile << "</td><td>" ;
	htmlFile << theNumber.formatParameters(i->covariance) ;
	htmlFile << "</td><td>" ;
	htmlFile << theNumber.formatParameters(i->correlation);
	htmlFile << "</td><td>" ;
	htmlFile << theNumber.formatTTests(i->ttest) ;
	htmlFile << "</td><td>" ;
	if (patParameters::the()->getgevPrintPValue()) {
	  
	  patReal pvalue = patPValue(patAbs(i->ttest),err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return ;
	  }
	  htmlFile << theNumber.formatTTests(pvalue) << '\t' ;
	  htmlFile << "</td><td>" ;
	  
	}
	if (patAbs(i->ttest) < patParameters::the()->getgevTtestThreshold()) {
	  htmlFile << patParameters::the()->getgevWarningSign() ;
	}
	htmlFile << "</td><td>" ;
	if (estimationResults.isRobustVarCovarAvailable) {
	  htmlFile << theNumber.formatParameters(i->robust_covariance) ;
	  htmlFile << "</td><td>" ;
	  htmlFile << theNumber.formatParameters(i->robust_correlation) ;
	  htmlFile << "</td><td>" ;
	  htmlFile << theNumber.formatTTests(i->robust_ttest) ;
	  htmlFile << "</td><td>" ;
	  if (patParameters::the()->getgevPrintPValue()) {

	    patReal pvalue = patPValue(patAbs(i->robust_ttest),err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	    htmlFile << theNumber.formatTTests(pvalue) << '\t' ;
	    htmlFile << "</td><td>" ;
	    
	  }
	  if (patAbs(i->robust_ttest) < patParameters::the()->getgevTtestThreshold()) {
	    htmlFile << patParameters::the()->getgevWarningSign() ;
	  }
	  htmlFile << "</td>" ;
	  
	}
      }
      htmlFile << "</table>" << endl ;
    }
    if (patParameters::the()->getgevPrintVarCovarAsMatrix()) {
      htmlFile << "<h2>Covariance of coefficients</h2>" << endl ;
      htmlFile << "<table border=\"1\" class=\"sortable\">" << endl ;
      htmlFile << "<tr class=biostyle><th> </th>" ;
      for (map<patString,map<patString,patReal> >::iterator ii = 
	     covarianceMatrix.begin() ;
	   ii != covarianceMatrix.end() ;
	   ++ii) {
	htmlFile << "<th>" << ii->first << "</th>" ;
      }
      htmlFile << "</tr>" << endl ;
      for (map<patString,map<patString,patReal> >::iterator ii = 
	     covarianceMatrix.begin() ;
	   ii != covarianceMatrix.end() ;
	   ++ii) {

	htmlFile << "<tr><th>" << ii->first << "</th>" ;
	for (map<patString,patReal>::iterator jj = ii->second.begin() ;
	     jj != ii->second.end() ;
	     ++jj) {
	  htmlFile << "<td>" << theNumber.formatParameters(jj->second) << "</td>" ;
	}
	htmlFile << "</tr>" << endl ;
      }
      htmlFile << "</table>" << endl ;


      htmlFile << "<h2>Correlation of coefficients</h2>" << endl ;
      htmlFile << "<table border=\"1\" class=\"sortable\">" << endl ;
      htmlFile << "<tr class=biostyle><th> </th>" ;
      for (map<patString,map<patString,patReal> >::iterator ii = 
	     correlationMatrix.begin() ;
	   ii != correlationMatrix.end() ;
	   ++ii) {
	htmlFile << "<th>" << ii->first << "</th>" ;
      }
      htmlFile << "</tr>" << endl ;
      for (map<patString,map<patString,patReal> >::iterator ii = 
	     correlationMatrix.begin() ;
	   ii != correlationMatrix.end() ;
	   ++ii) {

	htmlFile << "<tr><th>" << ii->first << "</th>" ;
	for (map<patString,patReal>::iterator jj = ii->second.begin() ;
	     jj != ii->second.end() ;
	     ++jj) {
	  htmlFile << "<td>" << theNumber.formatParameters(jj->second) << "</td>" ;
	}
	htmlFile << "</tr>" << endl ;
      }
      htmlFile << "</table>" << endl ;

      if (estimationResults.isRobustVarCovarAvailable) {
	htmlFile << "<h2>Robust covariance of coefficients</h2>" << endl ;
	htmlFile << "<table border=\"1\" class=\"sortable\">" << endl ;
	htmlFile << "<tr class=biostyle><th> </th>" ;
	for (map<patString,map<patString,patReal> >::iterator ii = 
	       robustCovarianceMatrix.begin() ;
	     ii != robustCovarianceMatrix.end() ;
	     ++ii) {
	  htmlFile << "<th>" << ii->first << "</th>" ;
	}
	htmlFile << "</tr>" << endl ;
	for (map<patString,map<patString,patReal> >::iterator ii = 
	       robustCovarianceMatrix.begin() ;
	     ii != robustCovarianceMatrix.end() ;
	     ++ii) {
	  
	  htmlFile << "<tr><th>" << ii->first << "</th>" ;
	  for (map<patString,patReal>::iterator jj = ii->second.begin() ;
	       jj != ii->second.end() ;
	       ++jj) {
	    htmlFile << "<td>" << theNumber.formatParameters(jj->second) << "</td>" ;
	  }
	  htmlFile << "</tr>" << endl ;
	}
	htmlFile << "</table>" << endl ;
	
	
	htmlFile << "<h2>Robust correlation of coefficients</h2>" << endl ;
	htmlFile << "<table border=\"1\" class=\"sortable\">" << endl ;
	htmlFile << "<tr class=biostyle><th> </th>" ;
	for (map<patString,map<patString,patReal> >::iterator ii = 
	       robustCorrelationMatrix.begin() ;
	     ii != robustCorrelationMatrix.end() ;
	     ++ii) {
	  htmlFile << "<th>" << ii->first << "</th>" ;
	}
	htmlFile << "</tr>" << endl ;
	for (map<patString,map<patString,patReal> >::iterator ii = 
	       robustCorrelationMatrix.begin() ;
	     ii != robustCorrelationMatrix.end() ;
	     ++ii) {
	  
	  htmlFile << "<tr><th>" << ii->first << "</th>" ;
	  for (map<patString,patReal>::iterator jj = ii->second.begin() ;
	       jj != ii->second.end() ;
	       ++jj) {
	    htmlFile << "<td>" << theNumber.formatParameters(jj->second) << "</td>" ;
	  }
	  htmlFile << "</tr>" << endl ;
	}
	htmlFile << "</table>" << endl ;
	
      }
    }
  }


  // Constraints
  
  patListProblemLinearConstraint eqCons = 
    patModelSpec::the()->getLinearEqualityConstraints(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return  ;
  }
  patListProblemLinearConstraint ineqCons = 
    patModelSpec::the()->getLinearInequalityConstraints(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }      
  
  if (!eqCons.empty() || !ineqCons.empty()) {
    htmlFile << endl ;
    htmlFile << "<h2>User defined linear constraints</h2>" << endl ;
  }
  for (patListProblemLinearConstraint::iterator iter = eqCons.begin() ;
       iter != eqCons.end() ;
       ++iter) {     
    htmlFile << "<p>" << patModelSpec::the()->printEqConstraint(*iter) << "</p>"<< endl ;
  }
  for (patListProblemLinearConstraint::iterator iter = ineqCons.begin() ;
       iter != ineqCons.end() ;
       ++iter) {     
    htmlFile << "<p>" << patModelSpec::the()->printIneqConstraint(*iter) << "</p>"<< endl ;
  }
  if (equalityConstraints != NULL) {
    htmlFile << "<h2>Nonlinear equality constraints</h2>" << endl ;
    for (patListNonLinearConstraints::iterator i = equalityConstraints->begin() ;
	 i != equalityConstraints->end() ;
	 ++i) {
      htmlFile << "<p>" << *i << "=0</p>" << endl ;
      
    }
  }
  if (inequalityConstraints != NULL) {
    htmlFile << "<h2>Nonlinear inequality constraints</h2>" << endl ;
    for (patListNonLinearConstraints::iterator i = inequalityConstraints->begin() ;
	 i != inequalityConstraints->end() ;
	 ++i) {
      htmlFile << "<p>" << *i << "=0</p>" << endl ;
      
    }
  }

  // Eigen vector
  
  htmlFile << "<p>Smallest singular value of the hessian: " << estimationResults.smallestSingularValue << "</p>" << endl ;
  if (!estimationResults.eigenVectors.empty()) {
    htmlFile << "<h2>Unidentifiable model</h2>" << endl ;
    htmlFile << "<p>The log likelihood is (almost) flat along the following combinations of parameters</p>" << endl ;
    htmlFile << "<table border=\"0\">" << endl ;

    patReal threshold = patParameters::the()->getgevSingularValueThreshold() ;
    patIterator<patBetaLikeParameter>* theParamIter = createAllParametersIterator() ;
    
    for(map<patReal,patVariables>::iterator iter = 
	  estimationResults.eigenVectors.begin() ;
	iter != estimationResults.eigenVectors.end() ;
	++iter) {
      htmlFile << "<tr class=biostyle><td>Sing. value</td><td>=</td><td>"
	       << iter->first 
	       << "</td></tr>" << endl ; 
      
      vector<patBoolean> printed(iter->second.size(),patFALSE) ;
      for (theParamIter->first() ;
	   !theParamIter->isDone() ;
	   theParamIter->next()) {
	patBetaLikeParameter bb = theParamIter->currentItem() ;
	if (!bb.isFixed && !bb.hasDiscreteDistribution) {
	  unsigned long j = bb.index ;
	  if (patAbs(iter->second[j]) >= threshold) {
	    htmlFile << "<tr class=biostyle><td>" 
		     << iter->second[j] 
		     << "</td><td>*</td><td>" 
		     << bb.name << "</td></tr>" << endl ;
	    printed[j] = patTRUE ;
	  }
	}
      }
      for (int j = 0 ; j < iter->second.size() ; ++j) {
	if (patAbs(iter->second[j]) >= threshold && !printed[j]) {
	  htmlFile << "<tr class=biostyle><td>" 
		   << iter->second[j] 
		   << "</td><td>*</td><td>Param[" 
		   << j << "]</td></tr>" << endl ;
	  
	}
      }
    }


    htmlFile << "</table>" << endl ;



    
  }

  htmlFile << "</html>" << endl;
  htmlFile.close() ;
  patOutputFiles::the()->addCriticalFile(fileName,"Estimation results in HTML format");

}


void patModelSpec::writeLatex(patString fileName, patError*& err) {
  DEBUG_MESSAGE("Write " << fileName) ;
  ofstream latexFile(fileName.c_str()) ;
  patAbsTime now ;
  now.setTimeOfDay() ;

  latexFile << "%% This file is designed to be included into a LaTeX document" << endl ;
  latexFile << "%% See http://www.latex-project.org/ for information about LaTeX" << endl ;
  latexFile << "%% " << patVersion::the()->getVersionInfo() << endl ;
  latexFile << "%% Compiled " <<patVersion::the()->getVersionDate() << endl ;
  latexFile << "%% " <<patVersion::the()->getVersionInfoAuthor()  << endl ;
  latexFile << "%% Report created on " <<now.getTimeString(patTsfFULL) << endl ;


  for (list<patString>::iterator i = modelDescription.begin() ;
       i != modelDescription.end() ;
       ++i) {
    latexFile << *i <<  "\\\\" << endl ;
  }
  latexFile << endl << endl ;
  latexFile << "\\begin{flushleft}" << endl ;
  latexFile << "\\begin{tabular}{rcl}" << endl ;
  latexFile << "\\hline" << endl ;
  latexFile << "Model &:& " ;
  if (isMixedLogit()) {
    latexFile << "Mixed " ;
  }
  latexFile << modelTypeName() <<"\\\\" << endl ;
  if (isMixedLogit()) {
    latexFile << "Number of " ;
    if (estimationResults.halton) {
      latexFile << "Halton " ;
    }
    if (estimationResults.hessTrain) {
      latexFile << "Hess-Train " ;
    }
    
    latexFile << "draws &:&"<< getNumberOfDraws() <<"\\\\" << endl ;
  }
  latexFile << "Number of estimated parameters&:&" << getNbrNonFixedParameters() <<"\\\\" << endl ;
  latexFile << "Number of " ;
  if (isAggregateObserved()) {
    latexFile << "aggregate" ;
  }
  latexFile << " observations &:& "<< estimationResults.numberOfObservations <<"\\\\" << endl ;
  latexFile << "Number of individuals&:&"<< estimationResults.numberOfIndividuals <<"\\\\" << endl ;
  latexFile << "Null log likelihood&:&"
	   <<  theNumber.formatStats(estimationResults.nullLoglikelihood)
	   << "\\\\" << endl ;
  if (allAlternativesAlwaysAvail) {
    latexFile << "Cte log likelihood&:&"
	      <<  theNumber.formatStats(estimationResults.cteLikelihood)
	      << "\\\\" << endl ;
  }
  latexFile << "Init log likelihood&:&" << theNumber.formatStats(estimationResults.initLoglikelihood) 
	   <<"\\\\" << endl ;
  latexFile << "Final log likelihood&:&"
	   << theNumber.formatStats(estimationResults.loglikelihood) 
	    <<"\\\\" << endl ;
  latexFile << "Likelihood ratio test &:&"
	   << theNumber.formatStats( -2.0 * (estimationResults.nullLoglikelihood - estimationResults.loglikelihood))
 <<"\\\\" << endl ;
  latexFile << "Rho-square&:&"
	   <<  theNumber.formatStats(1.0 - (estimationResults.loglikelihood / estimationResults.nullLoglikelihood))
	   <<"\\\\" << endl ;
  latexFile << "Adjusted rho-square&:&"
	   <<  theNumber.formatStats(1.0 - ((estimationResults.loglikelihood-getNbrNonFixedParameters()) / estimationResults.nullLoglikelihood)) <<"\\\\" << endl ;
  latexFile << "Final gradient norm&:&"
	    << theNumber.format(patTRUE,
				patFALSE,
				3,
				estimationResults.gradientNorm) 
	    <<"\\\\" << endl ;
  latexFile << "Diagnostic&:&"
	    << estimationResults.diagnostic
	    <<"\\\\" << endl ;
  if (estimationResults.iterations != 0) {
    latexFile << "Iterations&:&"
	      << estimationResults.iterations
	      <<"\\\\" << endl ;
  }
  latexFile << "Run time&:&"
	    << estimationResults.runTime
	    <<"\\\\" << endl ;

  
  latexFile << "Variance-covariance&:&" ; 
  if (patParameters::the()->getgevVarCovarFromBHHH() == 0) {
    if (patModelSpec::the()->isSimpleMnlModel() && patParameters::the()->getBTRExactHessian()) {
      latexFile << "from analytical hessian" << endl ;
    }
    else {
      latexFile << "from finite difference hessian" ;
    }
  }
  else {
    latexFile << "from BHHH matrix"  ;
  }
  latexFile << "\\\\" << endl ;

  unsigned short nSampleFiles = patFileNames::the()->getNbrSampleFiles() ;
  if (nSampleFiles == 1) {
    latexFile << "Sample file&:&" ;
  }
  else {
    latexFile << "Sample files&:&" ;
  }
  latexFile << patFileNames::the()->getSamFile(0,err) 
	    <<"\\\\" << endl ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  if (nSampleFiles > 1) {
    for (unsigned short i = 1 ; i < nSampleFiles ; ++i) {
      latexFile << "&&" << patFileNames::the()->getSamFile(i,err) 
		<<"\\\\" << endl ;
    }
  }



  latexFile << "\\end{tabular}" << endl ;
  latexFile << "\\end{flushleft}" << endl ;

  patVariables stdErr ;
  if (estimationResults.isVarCovarAvailable) {
    patMyMatrix* varCovar(estimationResults.varCovarMatrix) ;
    for (unsigned long i = 0 ; i < varCovar->nRows() ; ++i) {
      if ((*varCovar)[i][i] < 0.0) {
	stdErr.push_back(patMaxReal) ; 
      }
      else {
	stdErr.push_back(sqrt((*varCovar)[i][i])) ; 
      }
    }
  }
  patVariables robustStdErr ;
  if (estimationResults.isRobustVarCovarAvailable) {
    patMyMatrix* robustVarCovar(estimationResults.robustVarCovarMatrix) ;
    for (unsigned long i = 0 ; i < robustVarCovar->nRows() ; ++i) {
      if ((*robustVarCovar)[i][i] < 0) {
	robustStdErr.push_back(patMaxReal) ;
      }
      else {
	robustStdErr.push_back(sqrt((*robustVarCovar)[i][i])) ; 
      }
    }
  }


  latexFile << "%%" << endl ;
  latexFile << "%%" << endl ;
  latexFile << "%%" << endl ;
  latexFile << "  \\begin{tabular}{l}" << endl ;
  if (patParameters::the()->getgevPrintPValue()) {
    latexFile << "\\begin{tabular}{rlr@{.}lr@{.}lr@{.}lr@{.}l}" << endl ;
    if (estimationResults.isRobustVarCovarAvailable) {
      latexFile << "         &                       &   \\multicolumn{2}{l}{}    & \\multicolumn{2}{l}{Robust}  &     \\multicolumn{4}{l}{}   \\\\" << endl ;
    }
    latexFile << "Parameter &                       &   \\multicolumn{2}{l}{Coeff.}      & \\multicolumn{2}{l}{Asympt.}  &     \\multicolumn{4}{l}{}   \\\\" << endl ;
    latexFile << "number &  Description                     &   \\multicolumn{2}{l}{estimate}      & \\multicolumn{2}{l}{std. error}  &   \\multicolumn{2}{l}{$t$-stat}  &   \\multicolumn{2}{l}{$p$-value}   \\\\" << endl ;
  }
  else {
    latexFile << "\\begin{tabular}{rlr@{.}lr@{.}lr@{.}l}" << endl ;
    if (estimationResults.isRobustVarCovarAvailable) {
      latexFile << "         &                       &   \\multicolumn{2}{l}{}    & \\multicolumn{2}{l}{Robust}  &     \\multicolumn{2}{l}{}   \\\\" << endl ;
    }
    latexFile << "Variable &                       &   \\multicolumn{2}{l}{Coeff.}      & \\multicolumn{2}{l}{Asympt.}  &     \\multicolumn{2}{l}{}   \\\\" << endl ;
    latexFile << "number &  Description                     &   \\multicolumn{2}{l}{estimate}      & \\multicolumn{2}{l}{std. error}  &   \\multicolumn{2}{l}{$t$-stat}   \\\\" << endl ;
  }
  latexFile << "" << endl ;
  latexFile << "\\hline" << endl ;
  latexFile << "" << endl ;

 patULong paramNumber = 0 ;
 
 if (allBetaIter == NULL) {
   err = new patErrNullPointer("patIterator") ;
   WARNING(err->describe()) ;
   return ;
 }
 for (allBetaIter->first() ;
       !allBetaIter->isDone() ;
       allBetaIter->next()) {
    patBetaLikeParameter bb = allBetaIter->currentItem() ;
    if (!bb.isFixed) {
      ++paramNumber ;
      latexFile << paramNumber << " & " ; 
      latexFile << generateLatexRow(bb,stdErr,robustStdErr) ;
    }
 }

 latexFile << "\\hline" << endl ;

 patBoolean relevantSection(patFALSE);
 patBoolean footnote(patFALSE) ;
 patIterator<patBetaLikeParameter>* scaleIter = createScaleIterator() ;
 for (scaleIter->first() ;
      !scaleIter->isDone() ;
      scaleIter->next()) {
   patBetaLikeParameter bb = scaleIter->currentItem() ;
   if (!bb.isFixed) {
     ++paramNumber ;
     latexFile << paramNumber << " & " ; 
     latexFile << generateLatexRow(bb,stdErr,robustStdErr,1.0) ;
     footnote = patTRUE ;
     relevantSection = patTRUE ;
   }
   
 }
 if (relevantSection) {
   latexFile << "\\hline" << endl ;
 }

 patIterator<patBetaLikeParameter>* modelIter = createAllModelIterator() ;
 relevantSection = patFALSE ;
 for (modelIter->first() ;
      !modelIter->isDone() ;
      modelIter->next()) {
   patBetaLikeParameter bb = modelIter->currentItem() ;
   if (!bb.isFixed) {
     ++paramNumber ;
     latexFile << paramNumber << " & " ; 
     latexFile << generateLatexRow(bb,stdErr,robustStdErr,1.0) ;
     footnote = patTRUE ;
     relevantSection = patTRUE ;
   }
 }
 if (relevantSection) {
   latexFile << "\\hline" << endl ;
 }

 if (!mu.isFixed) {
   ++paramNumber  ;
   latexFile << paramNumber << " & " ; 
   latexFile << generateLatexRow(mu,stdErr,robustStdErr,1.0) ;
   footnote = patTRUE ;
   latexFile << "\\hline" << endl ;
 }

 latexFile << "" << endl ;
 latexFile << "\\end{tabular}" << endl ;
 latexFile << "\\\\" << endl ; 
  latexFile << "\\begin{tabular}{rcl}" << endl ;
  latexFile << "\\multicolumn{3}{l}{\\bf Summary statistics}\\\\" << endl ;
  latexFile << "\\multicolumn{3}{l}{ Number of observations = $"<< estimationResults.numberOfObservations <<"$} \\\\" << endl ;
  latexFile << " $\\mathcal{L}(0)$ &=&  $"<<theNumber.formatStats(estimationResults.nullLoglikelihood)<<"$ \\\\" << endl ;
  latexFile << " $\\mathcal{L}(c)$ &=& " ;
  if (allAlternativesAlwaysAvail) {
    latexFile << "$" << theNumber.formatStats(estimationResults.cteLikelihood) << "$" ;
  } 
  else {
    latexFile << "???" ;
  }
  latexFile << "\\\\" <<endl ;
  latexFile << " $\\mathcal{L}(\\hat{\\beta})$ &=& $"<<theNumber.formatStats(estimationResults.loglikelihood)<<" $  \\\\" << endl ;
  latexFile << " $-2[\\mathcal{L}(0) -\\mathcal{L}(\\hat{\\beta})]$ &=& $"<<theNumber.formatStats( -2.0 * (estimationResults.nullLoglikelihood - estimationResults.loglikelihood))<<"$ \\\\" << endl ;
  latexFile << "    $\\rho^2$ &=&   $"<<theNumber.formatStats(1.0 - (estimationResults.loglikelihood / estimationResults.nullLoglikelihood))<<"$ \\\\" << endl ;
  latexFile << "    $\\bar{\\rho}^2$ &=&    $"<< theNumber.formatStats(1.0 - ((estimationResults.loglikelihood-getNbrNonFixedParameters()) / estimationResults.nullLoglikelihood))<<"$ \\\\" << endl ;
  latexFile << "\\end{tabular}" << endl ;
  latexFile << "\\end{tabular}" << endl ;
 if (footnote) {
   latexFile << "\\footnotetext[1]{$t$-test against 1} " << endl ;
 }
  


  if (estimationResults.varCovarMatrix == NULL) {
    estimationResults.isVarCovarAvailable = patFALSE ;
  }
  if (estimationResults.robustVarCovarMatrix == NULL) {
    estimationResults.isRobustVarCovarAvailable = patFALSE ;
  }



 latexFile << "\\end{document}" << endl ;

 latexFile << "%%% Another joint format" << endl ; 
 latexFile << "%%%" << endl ; 


 latexFile.close() ;
 patOutputFiles::the()->addUsefulFile(fileName,"Estimation results in LaTeX format");
 return ;

}

void patModelSpec::writeSpecFile(patString fileName, patError*& err) {
  DEBUG_MESSAGE("Write " << fileName) ;
  ofstream specres(fileName.c_str()) ;
 
  patAbsTime now ;
  now.setTimeOfDay() ;
  specres << "// This file has automatically been generated." << endl ;
  specres << "// " << now.getTimeString(patTsfFULL) << endl ;
  specres << "// " << patVersion::the()->getCopyright() << endl ;
  specres << "// " << patVersion::the()->getVersionInfoDate() << endl ;
  specres << "// " << patVersion::the()->getVersionInfoAuthor() << endl ;
  specres << endl ;
//   specres << "[DataFile]" << endl ;
//   specres << "// Specify the number of columns that must be read in the data file" << endl ;
//   specres << "// It is used to check if the data file is read correctly." << endl ;
//   specres << "$COLUMNS = " ;
//   for (vector<unsigned long>::iterator i = columnData.begin() ;
//        i != columnData.end() ;
//        ++i) {
//     if (i != columnData.begin()) {
//       specres << " ; " ;
//     }
//     specres << *i ;
//   }
//   specres << endl ;
  
  specres << "[Choice]" << endl ;

  patArithNode* ptr = getChoiceExpr(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    specres << "??????????" << endl ;
  }
  else if (ptr == NULL) {
    specres << "??????????" << endl ;
  }
  else {
    specres << ptr->getExpression(err) << endl ;
  }

  specres << endl ;
  specres << "[Weight]" << endl ;

  ptr = getWeightExpr(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    specres << "??????????" << endl ;
  }
  else if (ptr == NULL) {
    specres << "$NONE" << endl ;
  }
  else {
    //    specres << setprecision(15) ;
    specres << ptr->getExpression(err) << endl ;
  }

  specres << endl ;

  specres << "[PanelData]" << endl ;
  specres << "// First, the attribute in the file containing the ID of the individual" << endl ;
  specres << "// Then the list of random parameters which are constant for all " << endl ;
  specres << "// observations of the same individual" << endl ;
  specres << "// The syntax for a random paramter with mean BETA and std err SIGMA is" << endl ;
  specres << "// BETA_SIGMA" << endl ;

  ptr = getPanelExpr(err) ;
  if ((err == NULL) && (ptr != NULL)) {
    specres << ptr->getExpression(err) << endl ;
    for (list<patString>::iterator i = panelVariables.begin() ;
	 i != panelVariables.end() ;
	 ++i) {
      specres << *i << endl ;
    }
    
  }
  else {
    specres << "$NONE" << endl ;
  }
  specres << endl ;

  //  specres << setprecision(7) << setiosflags(ios::scientific|ios::showpos) ;
  specres << endl ;
  specres << "[Beta]" << endl ;
  specres << "// Name Value  LowerBound UpperBound  status (0=variable, 1=fixed)" << endl ;
  for (map<patString,patBetaLikeParameter>::const_iterator i = 
	 betaParam.begin() ;
       i != betaParam.end() ;
       ++i) {
    if (!i->second.hasDiscreteDistribution) {
      specres << i->second.name << '\t' ;
      specres << i->second.estimated << '\t' ;
      specres << i->second.lowerBound << '\t' ;
      specres << i->second.upperBound << '\t' ;
      if (i->second.isFixed) {
	specres << "1" << endl ;
      }
      else {
	specres << "0" << endl ;
      }
    }
  }  
  specres << endl ;

  specres << "[LaTeX]" << endl ;
  if (latexNames.empty()) {
    specres << "$NONE" << endl ;
  }
  else {
    for (map<patString,patString>::iterator i = latexNames.begin() ;
	 i != latexNames.end() ;
	 ++i) {
      specres << i->first << "\t\"" << i->second << "\"" << endl ;
    }
  }
  specres << endl ;

  specres << "[OrdinalLogit]" << endl ;
  if (ordinalLogitThresholds.empty() && 
      ordinalLogitLeftAlternative == patBadId) {
    specres << "$NONE" << endl ;
  }
  else {
    specres << ordinalLogitLeftAlternative << " $NONE" << endl ;
    for (map<unsigned long, patString>::iterator i = 
	   ordinalLogitThresholds.begin() ;
	 i != ordinalLogitThresholds.end() ;
	 ++i) {
      specres << i->first << '\t' 
	      << i->second  << endl ;
    }
  }
  
  specres << endl ;
  specres << "[Mu]" << endl ;
  specres << "// In general, the value of mu must be fixed to 1. For testing purposes, you" << endl ;
  specres << "// may change its value or let it be estimated." << endl ;
  specres << "// Value LowerBound UpperBound Status" << endl ;
  specres << mu.estimated << '\t' ;
  specres << mu.lowerBound << '\t' ;
  specres << mu.upperBound << '\t' ;
  if (mu.isFixed) {
    specres << "1" << endl ; 
  }
  else {
    specres << "0" << endl ;
  }
  specres << endl ;

  specres << "[IIATest]" << endl ;
  specres << "// Relevant for biosim only" << endl ;
  specres << "// Description of the choice subsets to compute the new variable for McFadden's IIA test" << endl ;
  if (patModelSpec::the()->iiaTests.empty()) {
    specres << "$NONE" << endl ;
  }
  else {
    for (map<patString, list<long> >::iterator i = patModelSpec::the()->iiaTests.begin() ;
	 i != patModelSpec::the()->iiaTests.end() ;
	 ++i) {
      specres << i->first ;
      for (list<long>::iterator j = i->second.begin() ;
	   j != i->second.end(); 
	   ++j) {
	specres << '\t' << *j ;
      }
      specres << endl ;
    }
  }
  specres << endl ;
  specres << "[SampleEnum]" << endl ;
  specres << "// Relevant for biosim only" << endl ;
  specres << "// Number of simulated choices to include in the sample enumeration file" << endl ;
  specres << sampleEnumeration << endl ;
  specres << "" << endl ;


  specres << "[Utilities]" << endl ;
  specres << "// Id Name  Avail  linear-in-parameter expression (beta1*x1 + beta2*x2 + ... )" << endl ;
  unsigned long J = getNbrAlternatives() ;
  for (unsigned long alt = 0 ; alt < J ; ++alt) {
    unsigned long userId = getAltId(alt,err) ;
    specres << userId << '\t' ;
    specres << getAltName(userId,err) << '\t' ;
    specres << getAvailName(userId,err) << '\t' ;
    patUtilFunction* util = 
      patModelSpec::the()->getUtilFormula(userId,err) ;
    if (util->begin() == util->end()) {
      specres << "$NONE" ;
    }
    else {
      for (list<patUtilTerm>::iterator ii = util->begin() ;
	   ii != util->end() ;
	   ++ii) {
	if (ii != util->begin()) {
	  specres << " + " ;
	}
	if (ii->random) {
	  specres << ii->randomParameter->getOperatorName() << " * " << ii->x;
	}
	else {
	  specres << ii->beta << " * " << ii->x ;
	}
      }
    }
    specres << endl ;
  }
  
  specres << endl ;
  specres << "[GeneralizedUtilities]" << endl ;
  if (nonLinearUtilities.empty()) {
    specres << "$NONE" << endl ;
  }
  else {
    for (map<unsigned long, patArithNode*>::iterator 
	   i = nonLinearUtilities.begin();
	 i != nonLinearUtilities.end() ;
	 ++i) {
      specres << i->first << '\t' << *(i->second) << endl ;
    }
  }
  specres << endl ;

  if (!derivatives.empty()) {
    specres << "[Derivatives]" << endl ;
    for (map<unsigned long, map<patString, patArithNode*> >::iterator iter =
	   derivatives.begin() ;
	 iter != derivatives.end() ;
	 ++iter) {
      for (map<patString, patArithNode*>::iterator reiter = iter->second.begin() ;
	   reiter != iter->second.end() ;
	   ++reiter) {
	specres << iter->first << '\t' << reiter->first << '\t' << reiter->second->getExpression(err) << endl ;
      }
    }
    specres << endl ;
  }
  specres << "[SNP]" << endl ;
  if (listOfSnpTerms.empty()) {
    specres << "$NONE" << endl ;
  }
  else {
    specres << snpBaseParameter << endl ;
    for (list<pair<unsigned short,patString> >::iterator i =
	   listOfSnpTerms.begin() ;
	 i != listOfSnpTerms.end() ;
	 ++i) {
      specres << i->first << '\t' << i->second << endl ;
    }
  }

  specres << endl ;

  specres << "[AggregateLast]" << endl ;

  ptr = getAggLastExpr(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    specres << "??????????" << endl ;
  }
  else if (ptr == NULL) {
    specres << "$NONE" << endl ;
  }
  else {
    //    specres << setprecision(15) ;
    specres << ptr->getExpression(err) << endl ;
  }

  specres << endl ;

  specres << "[AggregateWeight]" << endl ;

  ptr = getAggWeightExpr(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    specres << "??????????" << endl ;
  }
  else if (ptr == NULL) {
    specres << "$NONE" << endl ;
  }
  else {
    //    specres << setprecision(15) ;
    specres << ptr->getExpression(err) << endl ;
  }

  specres << endl ;

  specres << "[SelectionBias]" << endl ;
  if (selectionBiasParameters.empty()) {
    specres << "$NONE" << endl ;
  }
  else {
    for (map<unsigned long, patString>::const_iterator i = selectionBiasParameters.begin() ;
	 i != selectionBiasParameters.end() ;
	 ++i) {
      specres << i->first << '\t' << i->second << endl ;
    }
  }
  specres << endl ;

  specres << "[ParameterCovariances]" << endl ;
  specres << "// Par_i Par_j Value  LowerBound UpperBound  status (0=variable, 1=fixed)" << endl ;
  if (covarParameters.empty()) {
    specres << "$NONE" << endl  ;
  }
  else {
    for (map<pair<patString,patString>,patBetaLikeParameter*>::iterator i = covarParameters.begin() ;
	 i != covarParameters.end() ;
	 ++i) {
      specres << i->first.first << '\t'
	      << i->first.second << '\t' 
	      << i->second->estimated << '\t'
	      << i->second->lowerBound << '\t'
	      << i->second->upperBound << '\t' ;
      if (i->second->isFixed) {
	specres << "1" << endl ; 
      }
      else {
	specres << "0" << endl ;
      }
    }
  }

  specres << "[Expressions] " << endl ;
  specres << "// Define here arithmetic expressions for name that are not directly " << endl ;
  specres << "// available from the data" << endl ;
  specres << endl ;

  printVarExpressions(specres,patFALSE,err) ;
  
  specres << endl ;

  specres << "[Draws]" << endl ;
  specres << numberOfDraws << endl << endl ;


  specres << "[Group]" << endl ;
  ptr = getGroupExpr(err) ;
  if (ptr != NULL) {
    specres << ptr->getExpression(err) << endl ;
  }
  specres << endl ;
  specres << "[Exclude]" << endl ;
  ptr = getExcludeExpr(err) ;
  if (ptr != NULL) {
    specres << ptr->getExpression(err) << endl ;
  }
  else {
    specres << "$NONE" << endl ;
  }
  specres << endl ;
  specres << "[Model]" << endl ;
  specres << "// Currently, the following models are available" << endl ;
  specres << "// Uncomment exactly one of them" << endl ;
  if (!patModelSpec::the()->isBP()) {
    specres << "//" ;
  }
  specres << "$BP  // Binary Probit Model" << endl ;
  if (!patModelSpec::the()->isOL()) {
    specres << "//"  ;
  }
  specres << "$OL // Ordinal logit" << endl ;
  specres << endl ;
  if (!patModelSpec::the()->isMNL()) {
    specres << "//" ;
  }
  specres << "$MNL  // Logit Model" << endl ;
  if (!patModelSpec::the()->isNL()) {
    specres << "//"  ;
  }
  specres << "$NL  // Nested Logit Model" << endl ;
  if (!patModelSpec::the()->isCNL()) {
    specres << "//"  ;
  }
  specres << "$CNL  // Cross-Nested Logit Model" << endl ;
  if (!patModelSpec::the()->isNetworkGEV()) {
    specres << "//"  ;
  }
  specres << "$NGEV // Network GEV Model" << endl ;
  specres << endl ;


  specres << endl ;
  specres << "[Scale]" << endl ;
  specres << "// The sample can be divided in several groups of individuals. The" << endl ;
  specres << "//utility of an individual in a group will be multiplied by the scale factor" << endl ;
  specres << "//associated with the group." << endl ;
  specres << "" << endl ;
  specres << "// Group_number  scale LowerBound UpperBound status" << endl ;
  specres << "" << endl ;
  patIterator<patBetaLikeParameter>* scaleIter = createScaleIterator() ;
  for (scaleIter->first() ;
       !scaleIter->isDone() ;
       scaleIter->next()) {
    patBetaLikeParameter b = scaleIter->currentItem() ;
    specres << getIdFromScaleName(b.name) << '\t' ;
    specres << b.estimated << '\t' ;
    specres << b.lowerBound << '\t' ;
    specres << b.upperBound << '\t' ;
    if (b.isFixed) {
      specres << "1" << endl ;
    }
    else {
      specres << "0" << endl ;
    }
  }

  specres << "[NLNests]" << endl ;
  specres << "// Name paramvalue  LowerBound UpperBound  status list of alt" 
	  << endl ;
  if (nlNestParam.empty()) {
    specres << "$NONE" << endl ;
  }
  else {
    for (map<patString,patNlNestDefinition>::const_iterator 
	   nestIter = nlNestParam.begin() ;
	 nestIter != nlNestParam.end() ;
	 ++nestIter) {
      specres << nestIter->second.nestCoef.name << '\t' ;
      specres << nestIter->second.nestCoef.estimated << '\t' ;
      specres << nestIter->second.nestCoef.lowerBound << '\t' ;
      specres << nestIter->second.nestCoef.upperBound << '\t' ;
      if (nestIter->second.nestCoef.isFixed) {
	specres << "1" ;
      }
      else {
	specres << " 0" ;
      }
      for (list<long>::const_iterator i = nestIter->second.altInNest.begin() ;
	   i != nestIter->second.altInNest.end() ;
	   ++i) {
	specres << " " << *i ;
      }
      specres << endl ;
    }
  }

  specres << "[CNLNests]" << endl ;
  specres << "// Name paramvalue LowerBound UpperBound  status " << endl ;
  if (!cnlNestParam.empty()) {
    for (map<patString,patBetaLikeParameter>::const_iterator
	   cnlIter = cnlNestParam.begin() ;
	 cnlIter != cnlNestParam.end() ;
	 ++cnlIter) {
      specres << cnlIter->second.name << '\t' ;
      specres << cnlIter->second.estimated << '\t' ;
      specres << cnlIter->second.lowerBound << '\t' ;
      specres << cnlIter->second.upperBound << '\t' ;
      if (cnlIter->second.isFixed) {
	specres << "1" << endl ;
      }
      else {
	specres << "0" << endl ;
      }
    }
  }
  else {
    specres << "$NONE" << endl ; 
  }
  specres << endl ;
  specres << "[CNLAlpha]" << endl ;
  specres << "// Alt Nest value LowerBound UpperBound  status" << endl ;
  if (!cnlNestParam.empty()) {  
    for (map<patString,patCnlAlphaParameter>::const_iterator 
	   alphaIter = cnlAlphaParam.begin() ;
	 alphaIter != cnlAlphaParam.end() ;
	 ++alphaIter) {
      specres << alphaIter->second.altName << '\t' ;
      specres << alphaIter->second.nestName << '\t' ;
      specres << alphaIter->second.alpha.estimated << '\t' ;
      specres << alphaIter->second.alpha.lowerBound << '\t' ;
      specres << alphaIter->second.alpha.upperBound << '\t' ;
      if (alphaIter->second.alpha.isFixed) {
	specres << "1" << endl ;
      }
      else {
	specres << "0" << endl ;
      }
    }
  }
  else {
    specres << "$NONE" << endl ;
  }

  specres << endl ;
  specres << "[Ratios] " << endl ;
  specres << "// List of ratios of estimated coefficients that must be produced in" << endl ;
  specres << "// the output. The most typical is the value-of-time." << endl ;
  specres << "// Numerator   Denominator  Name" << endl ;
  
  if (ratios.empty()) {
    specres << "$NONE" << endl ;
  }
  else {
    for (map<patString,pair<patString, patString> >::const_iterator i =
	   ratios.begin() ;
	 i != ratios.end() ;
	 ++i) {
      specres << i->second.first  << '\t' ;
      specres << i->second.second  << '\t' ;
      specres << i->first << endl ;
    }
  }
  specres << endl ;

  if (!discreteParameters.empty()) {
    specres << "[DiscreteDistributions]" << endl ;
    for (vector<patDiscreteParameter>::const_iterator i = discreteParameters.begin();
	 i != discreteParameters.end() ;
	 ++i) {
      specres << i->name ;
      specres << " < " ;
      for (vector<patDiscreteTerm>::const_iterator j = i->listOfTerms.begin() ;
	   j != i->listOfTerms.end() ;
	   ++j) {
	specres << j->massPoint->name ;
	specres << " ( " ;
	specres << j->probability->name ;
	specres << " ) " ;
      }
      specres << " >"  << endl ;
    }
  }

  specres << endl ;
  
  specres << "[LinearConstraints]" << endl ;

  if (listOfLinearConstraints.empty()) {
    specres << "$NONE" << endl ;
  }
  else {
    for (patListLinearConstraint::iterator i = 
	   listOfLinearConstraints.begin() ;
	 i != listOfLinearConstraints.end() ;
	 ++i) {
      specres << *i << endl ;
    }
  }
  specres << endl ;

  specres << "[NonLinearEqualityConstraints]" << endl ;
  if (equalityConstraints != NULL) {
    for (patListNonLinearConstraints::iterator i = equalityConstraints->begin() ;
	 i != equalityConstraints->end() ;
	 ++i) {
      specres << *i << endl ;
      
    }
  }
  else {
    specres << "$NONE" << endl ;
  }

  specres << endl ;
  specres << "[NonLinearInequalityConstraints]" << endl ;

  specres << "// At this point, BIOGEME is not able to handle nonlinear inequality" << endl ;
  specres << "// constraints yet. It should be available in a future version." << endl ;

  if (inequalityConstraints != NULL) {
    for (patListNonLinearConstraints::iterator i = inequalityConstraints->begin() ;
	 i != inequalityConstraints->end() ;
	 ++i) {
      specres << *i << endl ;
      
    }
  }
  else {
    specres << "$NONE" << endl ;
  }

  specres << endl ;
  specres << "[NetworkGEVNodes] " << endl ;
  specres << "// All nodes of the Network GEV model, except the root," << endl ;
  specres << "// must be listed here, with their associated parameter." << endl ;
  specres << "// If the nodes corresponding to alternatives are not listed," 
	  << endl ;
  specres << "// the associated parameter is constrained to 1.0 by default" 
	  << endl ;
  specres << "// Name  mu_param_value\tLowerBound\tUpperBound\tstatus" << endl ;
  if (networkGevNodes.empty()) {
    specres << "$NONE" << endl ;
  }
  else {
    for (map<patString, patBetaLikeParameter>::iterator i = 
	   networkGevNodes.begin() ;
	 i != networkGevNodes.end() ;
	 ++i) {
      specres << i->second.name << '\t' ;
      specres << i->second.estimated << '\t' ;
      specres << i->second.lowerBound << '\t' ;
      specres << i->second.upperBound << '\t' ;
      if (i->second.isFixed) {
	specres << "1" << endl ;
      }
      else {
	specres << "0" << endl ;
      }
    }
  }
  specres << endl ;
  specres << "[NetworkGEVLinks]" << endl ;
  specres << "// There is a line for each link of the network. " << endl ;
  specres << "// The root node is denoted by " << rootNodeName << endl ;
  specres << "// All other nodes must be either an alternative or a node listed in" << endl ;
  specres << "// the section [NetworkGEVNodes]" << endl ;
  specres << "// Note that an alternative cannot be the a-node of any link," << endl ;
  specres << "// and the root node cannot be the b-node of any link." << endl ;
  specres << "// a-node  b-node alpha_param_value LowerBound UpperBound  status" << endl ;
  if (networkGevLinks.empty()) {
    specres << "$NONE" << endl ;
  }
  else {
    for (map<patString, patNetworkGevLinkParameter>::iterator i =
	   networkGevLinks.begin() ;
	 i != networkGevLinks.end() ;
	 ++i) {
      specres << i->second.aNode << '\t' ;
      specres << i->second.bNode << '\t' ;
      specres << i->second.alpha.estimated << '\t' ;
      specres << i->second.alpha.lowerBound << '\t' ;
      specres << i->second.alpha.upperBound << '\t' ;
      if (i->second.alpha.isFixed) {
	specres << "1" << endl ;
      }
      else {
	specres << "0" << endl ;
      }
    }
  }

  specres << endl ;
  specres << "[ZhengFosgerau]" << endl ;
  specres << "// This section is used only by biosim for simulation, not by biogeme for estimation" << endl ;
  specres << "// Syntax: expression  bandwith lb  ub name" << endl ;
  specres << "// Expression must be a probability ($P) or an expression from the data ($E)" << endl ;
  specres << "// Examples:" << endl ;
  specres << "// $P { Alt1 } 1 0 1 \"P1\"" << endl ;
  specres << "// $E { x31 } 1 -1000 1000 \"x31\"  " << endl ;
  specres << "//" << endl ;
  if (zhengFosgerau.empty()) {
    specres << "$NONE" << endl ;
  }
  else {
    for (vector<patOneZhengFosgerau>::iterator i = zhengFosgerau.begin() ;
	 i != zhengFosgerau.end() ;
	 ++i) {
      if (i->isProbability()) {
	specres << "$P { " << i->getAltName() << " } " ;
      }
      else {
	specres << "$E { " << *(i->expression) << " } " ;
      }
      specres << i->bandwidth << " " << i->getLowerBound() << " " << i->getUpperBound() << " \"" << i->getTheName() << "\"" << endl ;
    }

  }
  specres.close() ;
  patOutputFiles::the()->addUsefulFile(fileName,"Model specification in Bison syntax");
}

void patModelSpec::writePythonSpecFile(patString fileName, patError*& err) {
  DEBUG_MESSAGE("Write " << fileName) ;
  ofstream pyfile(fileName.c_str()) ;
 
  patBoolean useWeights(patFALSE) ;
  patBoolean useScale(patFALSE) ;
  patBoolean isMixed(patFALSE) ;

  patAbsTime now ;
  now.setTimeOfDay() ;
  pyfile << "# This file has automatically been generated." << endl ;
  pyfile << "# " << now.getTimeString(patTsfFULL) << endl ;
  pyfile << "# " << patVersion::the()->getCopyright() << endl ;
  pyfile << "# " << patVersion::the()->getVersionInfoDate() << endl ;
  pyfile << "# " << patVersion::the()->getVersionInfoAuthor() << endl ;
  pyfile << endl ;
  pyfile << "#####################################################" << endl ;
  pyfile << "# This file complies with the syntax of pythonbiogeme" << endl ;
  pyfile << "# In general, it may require to be edited by hand before being operational" << endl ;
  pyfile << "# It is meant to help users translating their models from the previous version of biogeme to the python version." << endl ;
  pyfile << "#####################################################" << endl ;
  pyfile << endl ;

  pyfile << "from biogeme import *" << endl ;
  pyfile << "from headers import *" << endl ;
  if (patModelSpec::the()->isNL()) {
    pyfile << "from nested import *" << endl ;
  }
  if (patModelSpec::the()->isCNL()) {
    pyfile << "from cnl import *" << endl ;
  }
  pyfile << "from loglikelihood import *" << endl ;
  pyfile << "from statistics import *" << endl ;
  pyfile << "  " << endl ;
  
  pyfile << "# [Choice]" << endl ;
  pyfile << "__chosenAlternative = " ;

  patArithNode* ptr = getChoiceExpr(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    pyfile << "??????????" << endl ;
  }
  else if (ptr == NULL) {
    pyfile << "??????????" << endl ;
  }
  else {
    pyfile << ptr->getExpression(err) << endl ;
  }
  pyfile << endl ;

  pyfile << "# [Weight]" << endl ;

  ptr = getWeightExpr(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
  }
  else if (ptr == NULL) {
    pyfile << "# NONE" << endl ;
  }
  else {
    useWeights = patTRUE ;
    pyfile << "__weight = " << ptr->getExpression(err) << endl ;
  }

  pyfile << endl ;





  pyfile << "#[Beta]" << endl ;
  pyfile << "#Parameters to be estimated" << endl ;
  pyfile << "# Arguments:" << endl ;
  pyfile << "#   1  Name for report. Typically, the same as the variable" << endl ;
  pyfile << "#   2  Starting value" << endl ;
  pyfile << "#   3  Lower bound" << endl ;
  pyfile << "#   4  Upper bound" << endl ;
  pyfile << "#   5  0: estimate the parameter, 1: keep it fixed" << endl ;

  for (map<patString,patBetaLikeParameter>::const_iterator i = 
	 betaParam.begin() ;
       i != betaParam.end() ;
       ++i) {
    if (!i->second.hasDiscreteDistribution) {
      pyfile << i->second.name << '\t' ;
      pyfile << " = Beta('" << i->second.name << "'," ;
      if (patParameters::the()->getgevPythonFileWithEstimatedParam() > 0) {
	pyfile << i->second.estimated << ',' ;
      }
      else {
	pyfile << i->second.defaultValue << ',' ;
      }
      pyfile << i->second.lowerBound << ',' ;
      pyfile << i->second.upperBound << ',' ;
      if (i->second.isFixed) {
	pyfile << "1"  ;
      }
      else {
	pyfile << "0"  ;
      }
      map<patString,patString>::iterator found = latexNames.find(i->second.name) ;
      if (found != latexNames.end()) {
	pyfile << ",\"" << found->second << "\"" ;
      }
      pyfile << ")" << endl;
    }
  }  
  pyfile << endl ;

  // Random parameters

  patBoolean panel(patFALSE) ;
  ptr = getPanelExpr(err) ;
  if ((err == NULL) && (ptr != NULL)) {
    panel = patTRUE ;
    // Removing the last character which is a space
    patString columnName = ptr->getExpression(err) ;
    pyfile << "__individualId = '" << columnName.substr(0,columnName.size()-1) << "'" <<  endl ;

    pyfile << "#[PanelData]" << endl ;
    pyfile << "# The automatic translation of panel models has not been implemented yet" << endl ;
    pyfile << "# The panel data section contained the following entries:" << endl ;
    pyfile << "# " << ptr->getExpression(err) << endl ;
    for (list<patString>::iterator i = panelVariables.begin() ;
	 i != panelVariables.end() ;
	 ++i) {
      pyfile << "# " << *i << endl ;
    }
    
  }

  if (!randomParameters.empty()) {
    for (map<patString,pair<patRandomParameter,patArithRandom*> >::const_iterator i = 
	   randomParameters.begin() ;
	 i != randomParameters.end() ;
	 ++i) {
      pyfile << i->first << " = " << i->second.second->getLocationParameter() << " + " << i->second.second->getScaleParameter() << " * "; 
      if (i->second.first.type == NORMAL_DIST) {
	pyfile << "bioDraws" ; 
      }
      else if (i->second.first.type == UNIF_DIST) {
	pyfile << "bioDraws" ; 
      }
      else {
	pyfile << "UNKNOWN_DIST" ;
      }
      pyfile << "('" << i->first << "'" ;
      pyfile << ") " << endl ;
    }

    pyfile << "BIOGEME_OBJECT.DRAWS = {" ;
  patBoolean first = patTRUE ;
    for (map<patString,pair<patRandomParameter,patArithRandom*> >::const_iterator i = 
	   randomParameters.begin() ;
	 i != randomParameters.end() ;
	 ++i) {
      if (first) {
	first = patFALSE ;
      }
      else {
	pyfile << ", " ;
      }
      if (i->second.first.panel) {
	pyfile << "('" << i->first << "': " ;
	if (i->second.first.type == NORMAL_DIST) {
	  pyfile << "'NORMAL'" ; 
	}
	else if (i->second.first.type == UNIF_DIST) {
	  pyfile << "'UNIFORMSYM'" ; 
	}
	else {
	  pyfile << "'UNDEFINED'" ;
	}
	pyfile << ",__individualId)" ;
      }
      else {
	pyfile << i->first << ": " ;
	if (i->second.first.type == NORMAL_DIST) {
	  pyfile << "'NORMAL'" ; 
	}
	else if (i->second.first.type == UNIF_DIST) {
	  pyfile << "'UNIFORMSYM'" ; 
	}
	else {
	  pyfile << "'UNDEFINED'" ;
	}
      }
    }
    pyfile << "}" << endl ;
    isMixed = patTRUE ;
  }
  pyfile << endl ;

  pyfile << endl ;

  if (!latexNames.empty()) {
    pyfile << "#[LaTeX]" << endl ;
    for (map<patString,patString>::iterator i = latexNames.begin() ;
	 i != latexNames.end() ;
	 ++i) {
      pyfile << "# " << i->first << "\t\"" << i->second << "\"" << endl ;
    }
  }
  pyfile << endl ;

  if (ordinalLogitThresholds.empty() && 
      ordinalLogitLeftAlternative == patBadId) {
    //   pyfile << "$NONE" << endl ;
  }
  else {
    pyfile << "#[OrdinalLogit]" << endl ;
    pyfile << "# " << ordinalLogitLeftAlternative << " $NONE" << endl ;
    for (map<unsigned long, patString>::iterator i = 
	   ordinalLogitThresholds.begin() ;
	 i != ordinalLogitThresholds.end() ;
	 ++i) {
      pyfile << "# " << i->first << '\t' 
	     << i->second  << endl ;
    }
  }
  
  pyfile << endl ;
  patBoolean useMu(patFALSE) ;
  if (mu.estimated != 1.0 || !mu.isFixed) {
    useMu = patTRUE ;
    pyfile << "__Mu = Beta('__Mu'," ;
    if (patParameters::the()->getgevPythonFileWithEstimatedParam() > 0) {
      pyfile << mu.estimated << ',' ;
    }
    else {
      pyfile << mu.defaultValue << ',' ;
    }
    pyfile << mu.lowerBound << ',' ;
    pyfile << mu.upperBound << ',' ;
    if (mu.isFixed) {
      pyfile << "1)" << endl ; 
    }
    else {
      pyfile << "0)" << endl ;
    }
  }
  pyfile << endl ;

  if (patModelSpec::the()->iiaTests.empty()) {
    //    pyfile << "$NONE" << endl ;
  }
  else {
    pyfile << "# [IIATest]" << endl ;
    pyfile << "# Relevant for biosim only" << endl ;
    pyfile << "# Description of the choice subsets to compute the new variable for McFadden's IIA test" << endl ;
    for (map<patString, list<long> >::iterator i = patModelSpec::the()->iiaTests.begin() ;
	 i != patModelSpec::the()->iiaTests.end() ;
	 ++i) {
      pyfile << "# " << i->first ;
      for (list<long>::iterator j = i->second.begin() ;
	   j != i->second.end(); 
	   ++j) {
	pyfile << '\t' << *j ;
      }
      pyfile << endl ;
    }
  }
  pyfile << endl ;
  if (sampleEnumeration > 0) {
    pyfile << "#[SampleEnum]" << endl ;
    pyfile << "# Relevant for biosim only" << endl ;
    pyfile << "# Number of simulated choices to include in the sample enumeration file" << endl ;
    pyfile << "# " << sampleEnumeration << endl ;
    pyfile << "" << endl ;
  }


  pyfile << "# [Expressions] " << endl ;
  pyfile << "# Define here arithmetic expressions for name that are not directly " << endl ;
  pyfile << "# available from the data" << endl ;
  pyfile << endl ;

  printVarExpressions(pyfile,patTRUE,err) ;
  
  pyfile << endl ;

  ptr = getGroupExpr(err) ;
  if (ptr != NULL) {
    pyfile << "#[Group]" << endl ;
    patString gexpr = ptr->getExpression(err) ;
    if (gexpr != "1") {
      useScale = patTRUE ;
      pyfile << "__group = " << gexpr << endl ;
    }
  }
  pyfile << endl ;


  if (useScale) {
    pyfile << "#[Scale]" << endl ;
    pyfile << "# The sample can be divided in several groups of individuals. The" << endl ;
    pyfile << "#utility of an individual in a group will be multiplied by the scale factor" << endl ;
    pyfile << "#associated with the group." << endl ;
    pyfile << "" << endl ;
    patIterator<patBetaLikeParameter>* scaleIter = createScaleIterator() ;
    for (scaleIter->first() ;
	 !scaleIter->isDone() ;
	 scaleIter->next()) {
      patBetaLikeParameter b = scaleIter->currentItem() ;
      pyfile << "# Scale parameter for group " << getIdFromScaleName(b.name) << endl;
      pyfile << b.name << '\t' ;
      pyfile << " = Beta('" << b.name << "'," ;
      if (patParameters::the()->getgevPythonFileWithEstimatedParam() > 0) {
	pyfile << b.estimated << ',' ;
      }
      else {
	pyfile << b.defaultValue << ',' ;
      }
      pyfile << b.lowerBound << ',' ;
      pyfile << b.upperBound << ',' ;
      if (b.isFixed) {
	pyfile << "1)" << endl ;
      }
      else {
	pyfile << "0)" << endl ;
      }
    }
    pyfile << "__scale = " ;
    patBoolean first = patTRUE ;
    for (scaleIter->first() ;
	 !scaleIter->isDone() ;
	 scaleIter->next()) {
      patBetaLikeParameter b = scaleIter->currentItem() ;
      if (first) {
	first = patFALSE ;
      }
      else {
	pyfile << " + " ;
      }
      pyfile << "( __group == " << getIdFromScaleName(b.name) << ") * " << b.name ;
    }    
    pyfile << endl ;
  }


  pyfile << "#[Utilities]" << endl ;
  unsigned long J = getNbrAlternatives() ;
  for (unsigned long alt = 0 ; alt < J ; ++alt) {
    unsigned long userId = getAltId(alt,err) ;
    pyfile << "__" << getAltName(userId,err) << " = " ;
    //pyfile << getAvailName(userId,err) << '\t' ;
    patUtilFunction* util = 
      patModelSpec::the()->getUtilFormula(userId,err) ;
    if (util->begin() == util->end()) {
      pyfile << "0" ;
    }
    else {
      for (list<patUtilTerm>::iterator ii = util->begin() ;
	   ii != util->end() ;
	   ++ii) {
	if (ii != util->begin()) {
	  pyfile << " + " ;
	}
	if (ii->random) {
	  pyfile << ii->randomParameter->getCompactName() << " * " << ii->x;
	}
	else {
	  pyfile << ii->beta << " * " << ii->x ;
	}
      }
    }
    map<unsigned long, patArithNode*>::iterator found = nonLinearUtilities.find(userId) ;
    if (found != nonLinearUtilities.end()) {
      pyfile << " + " << *(found->second) ;
    }
    pyfile << endl ;
  }

  pyfile << "__V = {"  ;
  for (unsigned long alt = 0 ; alt < J ; ++alt) {
    unsigned long userId = getAltId(alt,err) ;
    if (alt > 0) {
      pyfile << "," ;
    }
    pyfile << userId << ": " ;
    if (useMu) {
      pyfile << "__Mu * " ;
    }
    if (useScale) {
      pyfile << "__scale * " ;
    }
    pyfile << "__" << getAltName(userId,err) ;
  }
  pyfile << "}" << endl ;


  pyfile << "__av = {"  ;
  for (unsigned long alt = 0 ; alt < J ; ++alt) {
    unsigned long userId = getAltId(alt,err) ;
    if (alt > 0) {
      pyfile << "," ;
    }
    pyfile << userId << ": " << getAvailName(userId,err) ;
  }
  pyfile << "}" << endl ;
  
  pyfile << endl ;
  if (listOfSnpTerms.empty()) {
    //    pyfile << "$NONE" << endl ;
  }
  else {
    pyfile << "# [SNP]" << endl ;
    pyfile << "# " << snpBaseParameter << endl ;
    for (list<pair<unsigned short,patString> >::iterator i =
	   listOfSnpTerms.begin() ;
	 i != listOfSnpTerms.end() ;
	 ++i) {
      pyfile << "# " << i->first << '\t' << i->second << endl ;
    }
  }

  pyfile << endl ;


  ptr = getAggLastExpr(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
  }
  else if (ptr == NULL) {
    //pyfile << "$NONE" << endl ;
  }
  else {
    pyfile << "# [AggregateLast]" << endl ;
    //    pyfile << setprecision(15) ;
    pyfile << "# " << ptr->getExpression(err) << endl ;
  }

  pyfile << endl ;


  ptr = getAggWeightExpr(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
  }
  else if (ptr == NULL) {
    //    pyfile << "$NONE" << endl ;
  }
  else {
    pyfile << "# [AggregateWeight]" << endl ;
    //    pyfile << setprecision(15) ;
    pyfile << "# " << ptr->getExpression(err) << endl ;
  }

  pyfile << endl ;

  if (selectionBiasParameters.empty()) {
    //    pyfile << "$NONE" << endl ;
  }
  else {
    pyfile << "# [SelectionBias]" << endl ;
    for (map<unsigned long, patString>::const_iterator i = selectionBiasParameters.begin() ;
	 i != selectionBiasParameters.end() ;
	 ++i) {
      pyfile << "# " << i->first << '\t' << i->second << endl ;
    }
  }
  pyfile << endl ;

  if (covarParameters.empty()) {
    //    pyfile << "$NONE" << endl  ;
  }
  else {
    pyfile << "# [ParameterCovariances]" << endl ;
    for (map<pair<patString,patString>,patBetaLikeParameter*>::iterator i = covarParameters.begin() ;
	 i != covarParameters.end() ;
	 ++i) {
      pyfile << "# " << i->first.first << '\t'
	     << i->first.second << '\t' 
	     << i->second->estimated << '\t'
	     << i->second->lowerBound << '\t'
	     << i->second->upperBound << '\t' ;
      if (i->second->isFixed) {
	pyfile << "1" << endl ; 
      }
      else {
	pyfile << "0" << endl ;
      }
    }
  }


  pyfile << "#[Draws]" << endl ;
  pyfile << "BIOGEME_OBJECT.PARAMETERS['NbrOfDraws'] = \""<< numberOfDraws<<"\"" << endl ;


  ptr = getExcludeExpr(err) ;
  if (ptr != NULL) {
    pyfile << "#[Exclude]" << endl ;
    pyfile << "BIOGEME_OBJECT.EXCLUDE = " << ptr->getExpression(err) << endl ;
  }
  else {
    //    pyfile << "$NONE" << endl ;
  }
  pyfile << endl ;

  if (nlNestParam.empty()) {
    //    pyfile << "$NONE" << endl ;
  }
  else {
    pyfile << "#[NLNests]" << endl ;
    pyfile << "# Name paramvalue  LowerBound UpperBound  status list of alt" 
	   << endl ;
    for (map<patString,patNlNestDefinition>::const_iterator 
	   nestIter = nlNestParam.begin() ;
	 nestIter != nlNestParam.end() ;
	 ++nestIter) {
      pyfile << nestIter->second.nestCoef.name << " = " ;
      pyfile << "Beta('" << nestIter->second.nestCoef.name << "'," ;
      if (patParameters::the()->getgevPythonFileWithEstimatedParam() > 0) {
	pyfile << nestIter->second.nestCoef.estimated << ',' ;
      }
      else {
	pyfile << nestIter->second.nestCoef.defaultValue << ',' ;
      }
      pyfile << nestIter->second.nestCoef.lowerBound << ',' ;
      pyfile << nestIter->second.nestCoef.upperBound << ',' ;
      if (nestIter->second.nestCoef.isFixed) {
	pyfile << "1)" ;
      }
      else {
	pyfile << " 0)" ;
      }
      pyfile << endl ;
      pyfile << "__" << nestIter->second.nestCoef.name << " = " ;
      pyfile << nestIter->second.nestCoef.name << " , [" ;
      for (list<long>::const_iterator i = nestIter->second.altInNest.begin() ;
	   i != nestIter->second.altInNest.end() ;
	   ++i) {
	if (i != nestIter->second.altInNest.begin()) {
	  pyfile << ", " ;
	}
	pyfile << " " << *i ;
      }
      pyfile << "]" << endl ;
      pyfile << endl ;
    }
    pyfile << "__nests = " ;
    for (map<patString,patNlNestDefinition>::const_iterator 
	   nestIter = nlNestParam.begin() ;
	 nestIter != nlNestParam.end() ;
	 ++nestIter) {
      if (nestIter != nlNestParam.begin()) {
	pyfile << ", "  ;
      }
      pyfile << "__" << nestIter->second.nestCoef.name ;
    }
    pyfile << endl ;
  }
  if (!cnlNestParam.empty()) {  
    pyfile << "#[CNLNests]" << endl ;
    pyfile << "# Name paramvalue LowerBound UpperBound  status " << endl ;
    for (map<patString,patBetaLikeParameter>::const_iterator
	   cnlIter = cnlNestParam.begin() ;
	 cnlIter != cnlNestParam.end() ;
	 ++cnlIter) {
      pyfile << cnlIter->second.name << " = " ;
      pyfile << "Beta('" << cnlIter->second.name << "'," ;
      if (patParameters::the()->getgevPythonFileWithEstimatedParam() > 0) {
	pyfile << cnlIter->second.estimated << ',' ;
      }
      else {
	pyfile << cnlIter->second.defaultValue << ',' ;
      }
      pyfile << cnlIter->second.lowerBound << ',' ;
      pyfile << cnlIter->second.upperBound << ',' ;
      if (cnlIter->second.isFixed) {
	pyfile << "1)" << endl ;
      }
      else {
	pyfile << "0)" << endl ;
      }
    }
    pyfile << endl ;
    pyfile << "#[CNLAlpha]" << endl ;
    for (map<patString,patCnlAlphaParameter>::const_iterator 
	   alphaIter = cnlAlphaParam.begin() ;
	 alphaIter != cnlAlphaParam.end() ;
	 ++alphaIter) {
      pyfile << alphaIter->second.alpha.name << " = " ;
      pyfile << "Beta('" << alphaIter->second.alpha.name << "'," ;
      if (patParameters::the()->getgevPythonFileWithEstimatedParam() > 0) {
	pyfile << alphaIter->second.alpha.estimated << ',' ;
      }
      else {
	pyfile << alphaIter->second.alpha.defaultValue << ',' ;
      }
      pyfile << alphaIter->second.alpha.lowerBound << ',' ;
      pyfile << alphaIter->second.alpha.upperBound << ',' ;
      if (alphaIter->second.alpha.isFixed) {
	pyfile << "1)" << endl ;
      }
      else {
	pyfile << "0)" << endl ;
      }
    }
    pyfile << endl ;

    // Loop on nests 
    for (map<patString,patBetaLikeParameter>::const_iterator
 	   cnlIter = cnlNestParam.begin() ;
 	 cnlIter != cnlNestParam.end() ;
 	 ++cnlIter) {
      pyfile << "__alpha_" << cnlIter->second.name << " = {" ;
      patBoolean first = patTRUE ;
      set<patULong> alphaAlt ;
      for (map<patString,patCnlAlphaParameter>::const_iterator 
	     alphaIter = cnlAlphaParam.begin() ;
	   alphaIter != cnlAlphaParam.end() ;
	   ++alphaIter) {
	if (alphaIter->second.nestName == cnlIter->second.name) {
	  if (first) {
	    first = patFALSE ;
	  }
	  else {
	    pyfile << ", " ;
 	  }
	  patULong altUserId = getAltUserId(alphaIter->second.altName) ;
	  alphaAlt.insert(altUserId) ;
	  pyfile << altUserId << ": " << alphaIter->second.alpha.name ;
	}
      }
      // Add alternatives not already included, with an alpha parameter = 0
      for (unsigned long alt = 0 ; alt < J ; ++alt) {
	unsigned long userId = getAltId(alt,err) ;
	set<patULong>::iterator found = alphaAlt.find(userId) ;
	if (found == alphaAlt.end()) {
	  if (first) {
	    first = patFALSE ;
	  }
	  else {
	    pyfile << ", " ;
	  }
	  pyfile << userId << ": 0" ;
	   
	}
      } 
      pyfile << "}" << endl ;
    }

    // Loop on nests 
    for (map<patString,patBetaLikeParameter>::const_iterator
 	   cnlIter = cnlNestParam.begin() ;
 	 cnlIter != cnlNestParam.end() ;
 	 ++cnlIter) {
      pyfile << "__nest_" << cnlIter->second.name << " = " <<  cnlIter->second.name << ", " <<  "__alpha_" << cnlIter->second.name << endl ;
      
    } 
    pyfile << "__nests = " ;
    // Loop on nests 
    for (map<patString,patBetaLikeParameter>::const_iterator
 	   cnlIter = cnlNestParam.begin() ;
 	 cnlIter != cnlNestParam.end() ;
 	 ++cnlIter) {
      if (cnlIter != cnlNestParam.begin()) {
	pyfile << ", " ;
      }
      pyfile << "__nest_" << cnlIter->second.name ;
    }     
    pyfile << endl ;
  }
  
  if (ratios.empty()) {
    //    pyfile << "$NONE" << endl ;
  }
  else {
    pyfile << "#[Ratios] " << endl ;
    pyfile << "# List of ratios of estimated coefficients that must be produced in" << endl ;
    pyfile << "# the output. The most typical is the value-of-time." << endl ;
    pyfile << "# Numerator   Denominator  Name" << endl ;
    for (map<patString,pair<patString, patString> >::const_iterator i =
	   ratios.begin() ;
	 i != ratios.end() ;
	 ++i) {
      pyfile << "# " << i->second.first  << '\t' ;
      pyfile << i->second.second  << '\t' ;
      pyfile << i->first << endl ;
    }
  }
  pyfile << endl ;

  if (!discreteParameters.empty()) {
    pyfile << "print('No automatic translation is available yet for discrete parameters')"  << endl ;
    pyfile << "print('The model must be specified by hand')" << endl ;
    pyfile << "sys.exit(0)" << endl ;
    pyfile << "# [DiscreteDistributions]" << endl ;
    for (vector<patDiscreteParameter>::const_iterator i = discreteParameters.begin();
	 i != discreteParameters.end() ;
	 ++i) {
      pyfile << "# " << i->name ;
      pyfile << " < " ;
      for (vector<patDiscreteTerm>::const_iterator j = i->listOfTerms.begin() ;
	   j != i->listOfTerms.end() ;
	   ++j) {
	pyfile << j->massPoint->name ;
	pyfile << " ( " ;
	pyfile << j->probability->name ;
	pyfile << " ) " ;
      }
      pyfile << " >"  << endl ;
    }
  }

  pyfile << endl ;
  

  if (listOfLinearConstraints.empty()) {
    //    pyfile << "#NONE" << endl ;
  }
  else {
    pyfile << "#[LinearConstraints]" << endl ;
    short cn(0) ;
    for (patListLinearConstraint::iterator i = 
	   listOfLinearConstraints.begin() ;
	 i != listOfLinearConstraints.end() ;
	 ++i) {
      ++cn ;
      if (i->theType == patLinearConstraint::patEQUAL){
	pyfile << "BIOGEME_OBJECT.CONSTRAINTS['Constraint_" << cn <<"'] = " << i->getFormForPython() << endl ;
      }
      else {
	pyfile << "#Cannot handle inequality constraints yet" << endl ;
	pyfile << "# " << *i << endl ;
      }
    }
  }
  pyfile << endl ;

  if (equalityConstraints != NULL) {
    pyfile << "#[NonLinearEqualityConstraints]" << endl ;
    for (patListNonLinearConstraints::iterator i = equalityConstraints->begin() ;
	 i != equalityConstraints->end() ;
	 ++i) {
      pyfile << "# " << *i << endl ;
      
    }
  }
  else {
    //    pyfile << "$NONE" << endl ;
  }

  pyfile << endl ;
  if (inequalityConstraints != NULL) {
    pyfile << "#[NonLinearInequalityConstraints]" << endl ;
    
    for (patListNonLinearConstraints::iterator i = inequalityConstraints->begin() ;
	 i != inequalityConstraints->end() ;
	 ++i) {
      pyfile << "# " << *i << endl ;
      
    }
  }
  else {
    //    pyfile << "$NONE" << endl ;
  }

  pyfile << endl ;
  if (networkGevNodes.empty()) {
    //    pyfile << "$NONE" << endl ;
  }
  else {
    pyfile << "#[NetworkGEVNodes] " << endl ;
    pyfile << "# All nodes of the Network GEV model, except the root," << endl ;
    pyfile << "# must be listed here, with their associated parameter." << endl ;
    pyfile << "# If the nodes corresponding to alternatives are not listed," 
	   << endl ;
    pyfile << "# the associated parameter is constrained to 1.0 by default" 
	   << endl ;
    pyfile << "# Name  mu_param_value\tLowerBound\tUpperBound\tstatus" << endl ;
    for (map<patString, patBetaLikeParameter>::iterator i = 
	   networkGevNodes.begin() ;
	 i != networkGevNodes.end() ;
	 ++i) {
      pyfile << i->second.name << " = " ;
      pyfile << "Beta('" << i->second.name << "'," ;
      if (patParameters::the()->getgevPythonFileWithEstimatedParam() > 0) {
	pyfile << i->second.estimated << ',' ;
      }
      else {
	pyfile << i->second.defaultValue << ',' ;
      }
      pyfile << i->second.lowerBound << ',' ;
      pyfile << i->second.upperBound << ',' ;
      if (i->second.isFixed) {
	pyfile << "1)" << endl ;
      }
      else {
	pyfile << "0)" << endl ;
      }
    }
  }
  pyfile << endl ;
  if (networkGevLinks.empty()) {
    //    pyfile << "$NONE" << endl ;
  }
  else {
    pyfile << "#[NetworkGEVLinks]" << endl ;
    pyfile << "# There is a line for each link of the network. " << endl ;
    pyfile << "# The root node is denoted by " << rootNodeName << endl ;
    pyfile << "# All other nodes must be either an alternative or a node listed in" << endl ;
    pyfile << "# the section [NetworkGEVNodes]" << endl ;
    pyfile << "# Note that an alternative cannot be the a-node of any link," << endl ;
    pyfile << "# and the root node cannot be the b-node of any link." << endl ;
    pyfile << "# a-node  b-node alpha_param_value LowerBound UpperBound  status" << endl ;
    for (map<patString, patNetworkGevLinkParameter>::iterator i =
	   networkGevLinks.begin() ;
	 i != networkGevLinks.end() ;
	 ++i) {
      //      pyfile << "# << i->second.aNode << '\t' ;
      //      pyfile << i->second.bNode << '\t' ;
      pyfile << i->second.alpha.name << " = " ;
      pyfile << "Beta('" << i->second.alpha.name << "'," ;
      if (patParameters::the()->getgevPythonFileWithEstimatedParam() > 0) {
	pyfile << i->second.alpha.estimated << ',' ;
      }
      else {
	pyfile << i->second.alpha.defaultValue << ',' ;
      }
      pyfile << i->second.alpha.lowerBound << ',' ;
      pyfile << i->second.alpha.upperBound << ',' ;
      if (i->second.alpha.isFixed) {
	pyfile << "1)" << endl ;
      }
      else {
	pyfile << "0)" << endl ;
      }
    }
  }

  pyfile << endl ;
  if (zhengFosgerau.empty()) {
    //    pyfile << "$NONE" << endl ;
  }
  else {
    pyfile << "#[ZhengFosgerau]" << endl ;
    pyfile << "# This section is used only by biosim for simulation, not by biogeme for estimation" << endl ;
    pyfile << "# Syntax: expression  bandwith lb  ub name" << endl ;
    pyfile << "# Expression must be a probability ($P) or an expression from the data ($E)" << endl ;
    pyfile << "# Examples:" << endl ;
    pyfile << "# $P { Alt1 } 1 0 1 \"P1\"" << endl ;
    pyfile << "# $E { x31 } 1 -1000 1000 \"x31\"  " << endl ;
    pyfile << "#" << endl ;
    for (vector<patOneZhengFosgerau>::iterator i = zhengFosgerau.begin() ;
	 i != zhengFosgerau.end() ;
	 ++i) {
      if (i->isProbability()) {
	pyfile << "# $P { " << i->getAltName() << " } " ;
      }
      else {
	pyfile << "# $E { " << *(i->expression) << " } " ;
      }
      pyfile << i->bandwidth << " " << i->getLowerBound() << " " << i->getUpperBound() << " \"" << i->getTheName() << "\"" << endl ;
    }

  }

  pyfile << "#[Model]" << endl ;
  if (patModelSpec::the()->isBP()) {
    pyfile << "# BP  // Binary Probit Model" << endl ;
  }
  if (patModelSpec::the()->isOL()) {
    pyfile << "# OL // Ordinal logit" << endl ;
  }
  if (patModelSpec::the()->isMNL()) {
    pyfile << "# MNL  // Logit Model" << endl ;
    pyfile << "# The choice model is a logit, with availability conditions" << endl ;
    pyfile << "prob = bioLogit(__V,__av,__chosenAlternative)" << endl ;
  }
  if (patModelSpec::the()->isNL()) {
    pyfile << "# NL  // Nested Logit Model" << endl ;
    if (useMu) {
      pyfile << "prob = nestedMevMu(__V,__av,__nests,__chosenAlternative,__Mu)" << endl;
    }
    else {
      pyfile << "prob = nested(__V,__av,__nests,__chosenAlternative)" << endl;
    }
  }
  if (patModelSpec::the()->isCNL()) {
    pyfile << "# CNL  // Cross-Nested Logit Model" << endl ;
    if (useMu) {
      pyfile << "prob = cnlmu_avail(__V,__av,__nests,__chosenAlternative,__Mu)"  << endl ;
    }
    else {
      pyfile << "prob = cnl_avail(__V,__av,__nests,__chosenAlternative)"  << endl ;
    }
  }
  if (patModelSpec::the()->isNetworkGEV()) {
    pyfile << "# NGEV // Network GEV Model" << endl ;
  }

  if (isMixed) {
    if (panel) {
      pyfile << "# Iterator on individuals, that is on groups of rows." << endl ;
      pyfile << "metaIterator('personIter','__dataFile__','panelObsIter',__individualId)" << endl ;
      pyfile << "" << endl ;
      pyfile << "# For each item of personIter, iterates on the rows of the group. " << endl ;
      pyfile << "rowIterator('panelObsIter','personIter')" << endl ;
      pyfile << "" << endl ;
      pyfile << "#Iterator on draws for Monte-Carlo simulation" << endl ;
      pyfile << "drawIterator('drawIter')" << endl ;
      pyfile << "" << endl ;
      pyfile << "#Conditional probability for the sequence of choices of an individual" << endl ;
      pyfile << "condProbIndiv = Prod(prob,'panelObsIter')" << endl ;
      pyfile << "" << endl ;
      pyfile << "# Integration by simulation" << endl ;
      pyfile << "probIndiv = Sum(condProbIndiv,'drawIter')" << endl ;
      pyfile << "" << endl ;
      pyfile << "__l = log(probIndiv)" << endl ;
    }
    else {
      pyfile << "drawIterator('drawIter')" << endl ;
      pyfile << "__l = Sum(prob,'drawIter')" << endl ;
    }
  }
  else {
    pyfile << "__l = log(prob)" << endl ;
  }

  if (useWeights) {
    pyfile << "__l = __l * __weight" << endl ;
  }

  pyfile << "" << endl ;
  if (!panel) {
    pyfile << "# Defines an itertor on the data" << endl ;
    pyfile << "rowIterator('obsIter') " << endl ;
    pyfile << "" << endl ;
    pyfile << "# Define the likelihood function for the estimation" << endl ;
    pyfile << "BIOGEME_OBJECT.ESTIMATE = Sum(__l,'obsIter')" << endl ;
  }
  else {
    pyfile << "BIOGEME_OBJECT.ESTIMATE = Sum(__l,'personIter')" << endl ;
    
  }

  pyfile << endl ;
  pyfile << "# The following parameters are imported from bison biogeme. You may want to remove them and prefer the default value provided by pythonbiogeme." << endl ;
  pyfile << endl ;

  pyfile << "BIOGEME_OBJECT.PARAMETERS['optimizationAlgorithm'] = \""<< patParameters::the()->getgevAlgo()<<"\"" << endl ;
  pyfile << "BIOGEME_OBJECT.PARAMETERS['stopFileName'] = \""<< patParameters::the()->getgevStopFileName()<<"\"" << endl ;
  

  pyfile << endl ;


  pyfile.close() ;
  patOutputFiles::the()->addCriticalFile(fileName,"Model specification in Python Bgiogeme syntax");
}

void patModelSpec::setEstimationResults(const patEstimationResult& res) {
  estimationResults = res ;
}

unsigned long patModelSpec::getNbrHeaders(unsigned short nh) const {
  return headers.size() ;
}
void patModelSpec::resetHeadersValues() {
  for (vector<patString>::iterator i = headers.begin() ;
       i != headers.end() ;
       ++i) {
    patValueVariables::the()->setValue(*i,
				       patParameters::the()->getgevMissingValue()) ;
    
  }
  
	    
}

void patModelSpec::setModelDescription(list<patString>* md) {
  if (md != NULL) {
    modelDescription = *md ;
  }
}

patBoolean patModelSpec::checkExpressionInFile(patArithNode* expression,
					       unsigned short fileId,
					       patError*& err) {
    // Check if the expression can be evaluated with the current file

  vector<patString> literals ;
  vector<patReal> valuesOfLiterals ;
  expression->getLiterals(&literals,&valuesOfLiterals,patFALSE,err) ;  
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patFALSE ;
  }

//   DEBUG_MESSAGE("List of literals") ;
//   for (vector<patString>::const_iterator i = literals.begin() ;
//        i != literals.end() ;
//        ++i) {
//     cout << *i << endl ;
//   }  
//   DEBUG_MESSAGE("End of list of literals") ;
//   DEBUG_MESSAGE("File headers") ;
//   for (vector<patString>::const_iterator i = headersPerFile[fileId].begin() ;
//        i != headersPerFile[fileId].end() ;
//        ++i) {
//     cout << *i << endl ;
//   }  
//   DEBUG_MESSAGE("End of file headers") ;


  for (vector<patString>::const_iterator i = literals.begin() ;
       i != literals.end() ;
       ++i) {
    
    vector<patString>::iterator found = 
      find(headersPerFile[fileId].begin(),
	   headersPerFile[fileId].end(),*i) ;

    if (found == headersPerFile[fileId].end() ) {      

      // The expression is not a header of the file. 
      // Check it recursively.
      
      patArithNode* literalExpression = getVariableExpr(*i,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patFALSE ;
      }
      if (literalExpression == NULL) {
	patString fileName = patFileNames::the()->getSamFile(fileId,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return patFALSE ;
	}
	DEBUG_MESSAGE("Expression " << *i << " not defined in file " << fileName) ;
	return patFALSE ;
      }
      
//       patBoolean result = checkExpressionInFile(literalExpression,fileId,err) ;
//       if (err != NULL) {
// 	WARNING(err->describe()) ;
// 	return patFALSE ;
//       }
//       if (result == patFALSE) {
// 	patString fileName = patFileNames::the()->getSamFile(fileId,err) ;
// 	if (err != NULL) {
// 	  WARNING(err->describe()) ;
// 	  return patFALSE ;
// 	}
// 	DEBUG_MESSAGE("Expression " << *i << " not defined in file " << fileName) ;
//       }
    }
  }
  return patTRUE ;
}


void patModelSpec::addMassAtZero(patString aName, patReal aThreshold) {
  listOfMassAtZero.push_back(pair<patString,patReal>(aName,aThreshold)) ;
}



unsigned long patModelSpec::getIdOfSelectionBiasParameter(unsigned long altIntId,patError*& err) {
  if (altIntId >= selectionBiasPerAlt.size()) {
    err = new patErrOutOfRange<unsigned long>(altIntId,0,selectionBiasPerAlt.size()-1) ;
    WARNING(err->describe()) ;
    return patBadId ;
  }
  if (selectionBiasPerAlt[altIntId] == NULL) {
    return patBadId ;
  }
  return selectionBiasPerAlt[altIntId]->id ;
}

patBoolean patModelSpec::isAggregateObserved() const {
  return (aggLastExpr != NULL) ;
}

void patModelSpec::addOrdinalLogitThreshold(unsigned long id,
					    patString paramName) {
  map<unsigned long, patString>::iterator found =
    ordinalLogitThresholds.find(id) ;
  if (found != ordinalLogitThresholds.end()) {
    WARNING("Interval " << id << " defined more than once in Section [OrdinalLogit]") ;
    return ;
  }
  ordinalLogitThresholds[id] = paramName ;
}

void patModelSpec::setOrdinalLogitLeftAlternative(unsigned long i) {
  ordinalLogitLeftAlternative = i ;
}

vector<patString>* patModelSpec::getHeaders(unsigned short fileId) {
  if (fileId >= headersPerFile.size()) {
    WARNING("No data file with ID " << fileId);
    return NULL;
  }
  return &(headersPerFile[fileId]) ;
}

unsigned long patModelSpec::getLargestAlternativeUserId() const {
  return largestAlternativeUserId ;
}

unsigned long patModelSpec::getLargestGroupUserId() const {
  return largestGroupUserId ;
}

void patModelSpec::generateCppCodeForAltId(ostream& cppFile,patError*& err) {
  for (  map<unsigned long,unsigned long>::iterator i = altIdToInternalId.begin() ;
	 i != altIdToInternalId.end() ;
	 ++i) {
    cppFile << "    altIndex["<<i->first<<"] = "<<i->second<<";" << endl ;
  }

}



void patModelSpec::generateCppCodeForGroupId(ostream& cppFile,patError*& err) {
  for (list<long>::iterator i = groupIds.begin() ;
       i != groupIds.end() ;
       ++i) {
    patBetaLikeParameter theScale = patModelSpec::the()->getScale(*i,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    cppFile << "      groupIndex[" << *i << "] = " << theScale.index << " ; " << endl ;
    cppFile << "      scalesPerGroup["<< *i<<"] = "<<theScale.defaultValue<<";" << endl ;
  }
  
}

void patModelSpec::setGeneralizedExtremeValueParameter(patString name) {
  generalizedExtremeValueParameterName = name ;
}
