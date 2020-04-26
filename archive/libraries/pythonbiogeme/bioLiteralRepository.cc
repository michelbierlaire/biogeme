//-*-c++-*------------------------------------------------------------
//
// File name : bioLiteralRepository.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Apr 28 16:28:07 2009
//
//--------------------------------------------------------------------

#include <algorithm>
#include <sstream>
#include "patDisplay.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "patErrOutOfRange.h"
#include "bioLiteralRepository.h"
#include "bioParameters.h" 
#include "bioPythonSingletonFactory.h"

bioLiteralRepository* bioLiteralRepository::the() {
  return bioPythonSingletonFactory::the()->bioLiteralRepository_the() ;
}

bioLiteralRepository::bioLiteralRepository() {

}

pair<patBoolean,pair<patULong,patULong> > bioLiteralRepository::addUserExpression(patString theName, patError*& err) {
 
  // Check if the variable is already defined
  pair<patULong,patULong>  theId = getVariable(theName,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return pair<patBoolean,pair<patULong,patULong> >() ;
  }
  if (theId.first != patBadId) {
    // If the name already exists as a literal, it may be because the
    // user expression is called recursively in python. In this case,
    // no error should be triggered.
    set<patString>::iterator found = userExpressions.find(theName) ;
    if (found == userExpressions.end()) {
      stringstream str ;
      str << "Variable " << theName << " cannot be redefined using DefineVariable as it is defined in the data set." ;
      err = new patErrMiscError(str.str());
      WARNING(err->describe()) ;
      return pair<patBoolean,pair<patULong,patULong> >() ;
    }
    else {
      return pair<patBoolean,pair<patULong,patULong> >(patFALSE,theId) ;
    }
  }
  theId = addVariable(theName,patBadId,err) ;
  userExpressions.insert(theName) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return pair<patBoolean,pair<patULong,patULong> >() ;
  }
  return pair<patBoolean,pair<patULong,patULong> >(patTRUE,theId) ;
}
pair<patULong,patULong> bioLiteralRepository::addVariable(patString theName, 
							  patULong colId,
							  patError*& err) {

  map<patString,patULong>::iterator found = 
    listOrganizedByNames.find(theName) ;
  if (found != listOrganizedByNames.end()) {
    pair<patULong,patULong> theIds = getVariable(theName, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
    }
    return theIds ;
  }

  // If colId is patBadId, it means that the variable does not
  // correspond to a "physical" column, but to a user-defined
  // expression. A virtual colId must be assigned.

  if (colId == patBadId) {
    patULong lastCol = getLastColumnId() ;
    if (lastCol == patBadId) {
      colId = 0 ;
    }
    else {
      colId = lastCol + 1 ;
    }
  }
  


  patULong uniqueId = bioParameters::the()->getValueInt("firstIdOfLiterals",err)  ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return pair<patULong,patULong>() ;
  }
  uniqueId += listOrganizedByNames.size() ;
  
  patULong varId = listOfVariables.size() ;

  bioVariable variable(theName,uniqueId, varId, colId) ;
  listOfVariables.push_back(variable) ;
  listOrganizedByNames[theName] = uniqueId ;
  theIdAndTypes[uniqueId] = pair<patULong,bioLiteral::bioLiteralType>(varId,bioLiteral::VARIABLE) ;
  return pair<patULong,patULong>(uniqueId,varId) ;
}

pair<patULong,patULong> bioLiteralRepository::addRandomVariable(patString theName, 
								patError*& err) {
  map<patString,patULong>::iterator found = 
    listOrganizedByNames.find(theName) ;
  if (found != listOrganizedByNames.end()) {
    pair<patULong,patULong> theIds = getRandomVariable(theName, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
    }
    return theIds ;

  }
  

  patULong uniqueId = bioParameters::the()->getValueInt("firstIdOfLiterals",err)  ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return pair<patULong,patULong>() ;
  }
  uniqueId += listOrganizedByNames.size() ;

  patULong rvId = listOfRandomVariables.size() ;
  bioRandomVariable variable(theName,uniqueId,rvId) ;
  listOfRandomVariables.push_back(variable) ;
  listOrganizedByNames[theName] = uniqueId ;
  theIdAndTypes[uniqueId] = pair<patULong,bioLiteral::bioLiteralType>(rvId,bioLiteral::RANDOM) ;
  return pair<patULong,patULong>(uniqueId,rvId) ;
}

pair<patULong,patULong> bioLiteralRepository::addFixedParameter(patString theName, 
								patReal val,
								patReal lowerBound,
								patReal upperBound, 
								patBoolean fixed, 
								patString latexName, 
								patError*& err) {

  map<patString,patULong>::iterator found = 
    listOrganizedByNames.find(theName) ;
  if (found != listOrganizedByNames.end()) {
    pair<patULong,patULong> theIds = getBetaParameter(theName, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
    }
    return theIds ;
  }

  patULong uniqueId = bioParameters::the()->getValueInt("firstIdOfLiterals",err)  ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return pair<patULong,patULong>() ;
  }
  uniqueId += listOrganizedByNames.size() ;

  patULong betaId = listOfFixedParameters.size() ;

  patULong estimatedId = patBadId ;
  if (!fixed) {
    estimatedId = parametersToBeEstimated.size() ;
    parametersToBeEstimated[betaId] = estimatedId ;
  }

  if (latexName == "") {
    latexName = theName ;
  }
 
  bioFixedParameter parameter(theName,uniqueId, betaId,estimatedId,val,lowerBound, upperBound, fixed, latexName) ;
  listOfFixedParameters.push_back(parameter) ;
  
  listOrganizedByNames[theName] = uniqueId ;
  flags[uniqueId] = patFALSE ;
  theIdAndTypes[uniqueId] = pair<patULong,bioLiteral::bioLiteralType>(betaId,bioLiteral::PARAMETER) ;
  return pair<patULong,patULong>(uniqueId,betaId) ;

}

pair<patULong,patULong>  bioLiteralRepository::addFixedParameter(patString theName, patReal val, patError*& err) {
  map<patString,patULong>::iterator found = 
    listOrganizedByNames.find(theName) ;
  if (found != listOrganizedByNames.end()) {
    pair<patULong,patULong> theIds = getBetaParameter(theName, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
    }
    return theIds ;
  }


  patULong uniqueId = bioParameters::the()->getValueInt("firstIdOfLiterals",err)  ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return pair<patULong,patULong>() ;
  }
  uniqueId += listOrganizedByNames.size() ;

  patULong betaId = listOfFixedParameters.size() ;

  bioFixedParameter parameter(theName,uniqueId, betaId, val) ;
  listOfFixedParameters.push_back(parameter) ;
  listOrganizedByNames[theName] = uniqueId ;
  flags[uniqueId] = patFALSE ;
  theIdAndTypes[uniqueId] = pair<patULong,bioLiteral::bioLiteralType>(betaId,bioLiteral::PARAMETER) ;
  return pair<patULong,patULong>(uniqueId,betaId) ;
}


patULong bioLiteralRepository::getLiteralId(patString theName,
					    patError*& err) const {
  
  map<patString,patULong>::const_iterator found = 
    listOrganizedByNames.find(theName) ;
  if (found != listOrganizedByNames.end()) {
    return found->second ;
  }
  else {
    return patBadId ;
  }
  
}

patULong bioLiteralRepository::getLiteralId(const bioLiteral* aLiteral,
					    patError*& err) const {
  if (aLiteral == NULL) {
    return patBadId ;
  }
  patULong result = getLiteralId(aLiteral->getName(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patBadId ;
  }
  return result ;
}

patString bioLiteralRepository::getName(patULong theId, patError*& err) {
  for (map<patString,patULong>::const_iterator i = listOrganizedByNames.begin() ;
       i != listOrganizedByNames.end() ;
       ++i) {
    if (i->second == theId) {
      return i->first ;
    }
  }
  stringstream str ;
  str << "Literal " << theId << " does not exist" ;
  err = new patErrMiscError(str.str()) ;
  WARNING(err->describe()) ;
  return patString() ;
 }


patULong bioLiteralRepository::getNumberOfVariables() const {
  return listOfVariables.size() ;
}
patULong bioLiteralRepository::getNumberOfRandomVariables() const {
  return listOfRandomVariables.size() ;
}
patULong bioLiteralRepository::getNumberOfParameters() const {
  return listOfFixedParameters.size() ;
}

patULong bioLiteralRepository::getNumberOfEstimatedParameters()  {
  return parametersToBeEstimated.size() ;
}

patReal bioLiteralRepository::getBetaValue(patString name, patError*& err) {
    for (vector<bioFixedParameter>::const_iterator i = listOfFixedParameters.begin() ;
	 i != listOfFixedParameters.end() ;
	 ++i) {
      if (i->getName() == name) {
	return i->currentValue ;
      }
    }
    stringstream str ;
    str << "Parameter " << name << " is unknown. List of "<< listOfFixedParameters.size() <<" parameters:" ;
    for (vector<bioFixedParameter>::const_iterator i = listOfFixedParameters.begin() ;
	 i != listOfFixedParameters.end() ;
	 ++i) {
      str << " " << i->getName() ;
    }
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return 999999 ;
  
}

patVariables bioLiteralRepository::getBetaValues(patBoolean all) const {
  if (all) {
    patVariables result ;
    for (vector<bioFixedParameter>::const_iterator i = listOfFixedParameters.begin() ;
	 i != listOfFixedParameters.end() ;
	 ++i) {
      result.push_back(i->currentValue) ;
    }
    return result ;
    
  }
  else {
    patVariables result(parametersToBeEstimated.size()) ;
    for (map<patULong,patULong>::const_iterator i = parametersToBeEstimated.begin() ;
	 i != parametersToBeEstimated.end() ;
	 ++i) {
      result[i->second] = listOfFixedParameters[i->first].currentValue ;
    }
  return result ;
  }
}

patVariables bioLiteralRepository::getLowerBounds() const {
  patVariables result ;
  for (vector<bioFixedParameter>::const_iterator i = 
	 listOfFixedParameters.begin() ;
       i != listOfFixedParameters.end() ;
       ++i) {
    if (!i->isFixed) {
      result.push_back(i->lowerBound) ;
    }
  }
  return result ;
  
}

patVariables bioLiteralRepository::getUpperBounds() const {
  patVariables result ;
  for (vector<bioFixedParameter>::const_iterator i = listOfFixedParameters.begin() ;
       i != listOfFixedParameters.end() ;
       ++i) {
    if (!i->isFixed) {
      result.push_back(i->upperBound) ;
    }
  }
  return result ;
  
}


vector<patULong> bioLiteralRepository::getBetaIds(patBoolean all,
						  patError*& err) const {
  vector<patULong> result ;
  for (vector<bioFixedParameter>::const_iterator i = listOfFixedParameters.begin() ;
       i != listOfFixedParameters.end() ;
       ++i) {
    patULong id = getLiteralId(&(*i),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return vector<patULong>() ;
    }
    if (all || !i->isFixed) {
      result.push_back(id) ;
    }
  }
  return result ;
}

void bioLiteralRepository::setBetaValues(patVariables b, patError*& err) {

  for (map<patULong,patULong>::iterator i = parametersToBeEstimated.begin() ;
       i != parametersToBeEstimated.end() ;
       ++i) {
    listOfFixedParameters[i->first].currentValue = b[i->second] ;
  }
  return ;
}

patString bioLiteralRepository::getBetaName(patULong betaId, patBoolean all,patError*& err) {
  if (all) {
    if (betaId >= listOfFixedParameters.size()) {
      err = new patErrOutOfRange<patULong>(betaId,0, listOfFixedParameters.size()-1) ;
      WARNING(err->describe()) ;
      return patString() ;
    }
    return listOfFixedParameters[betaId].name ;
  }
  else {
    for (map<patULong,patULong>::iterator i = parametersToBeEstimated.begin() ;
	 i != parametersToBeEstimated.end() ;
	 ++i) {
      if (i->second == betaId) {
	return listOfFixedParameters[i->first].name ;
      }
    }
    stringstream str ;
    str << "Estimated parameter " << betaId << " unknown" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patString() ;
  }
}

pair<patULong,patULong> bioLiteralRepository::getBetaParameter(patString name, patError*& err) {
  for (patULong i = 0 ; i < listOfFixedParameters.size() ; ++i) {
    if (listOfFixedParameters[i].getName() == name) {
      return pair<patULong,patULong>(listOfFixedParameters[i].getId(),i) ;
    }
  }
  stringstream str ;
  str << "Unknown fixed parameter " << name ;
  err = new patErrMiscError(str.str()) ;
  WARNING(err->describe()) ;
  return pair<patULong,patULong>() ;
}


void bioLiteralRepository::setFlag(patULong id) {
  flags[id] = patTRUE ;
}

void bioLiteralRepository::unsetFlag(patULong id) {
  flags[id] = patFALSE ;
}

patBoolean bioLiteralRepository::isFlagSet(patULong id) const {
  map<patULong,patBoolean>::const_iterator found = flags.find(id) ;
  if (found == flags.end()) {
    return patFALSE ;
  }
  return found->second ;
}

void bioLiteralRepository::resetAllFlags() {
  for (map<patULong,patBoolean>::iterator i = flags.begin() ;
       i != flags.end() ;
       ++i) {
    i->second = patFALSE ;
  }
}
ostream& operator<<(ostream &str, const bioLiteralRepository& x) {
  str << "Ids of variables" << endl ;
  str << "++++++++++++++++" << endl ;
  for (patULong i = 0 ; i < x.listOfVariables.size() ; ++i) {
    str << i << ": " << x.listOfVariables[i].getName() << " Unique ID: " << x.listOfVariables[i].getId() << endl ;

  }
  str << "Ids of random variables" << endl ;
  str << "+++++++++++++++++++++++" << endl ;
  for (patULong i = 0 ; i < x.listOfRandomVariables.size() ; ++i) {
    str << i << ": " << x.listOfRandomVariables[i].getName() << " Unique ID: " <<x.listOfRandomVariables[i].getId() << endl ;

  }
  str << "Ids of composite literals" << endl ;
  str << "+++++++++++++++++++++++++" << endl ;
  for (patULong i = 0 ; i < x.listOfCompositeLiterals.size() ; ++i) {
    str << i << ": " << x.listOfCompositeLiterals[i].getName() << " Unique ID: " <<x.listOfCompositeLiterals[i].getId() << endl ;

  }
  str << "Ids of  parameters" << endl ;
  str << "++++++++++++++++++" << endl ;
  for (patULong i = 0 ; i < x.listOfFixedParameters.size() ; ++i) {
    str << i << ": " << x.listOfFixedParameters[i].getName() << " Unique ID: " <<x.listOfFixedParameters[i].getId() << endl ;

  }
  str << "Ids of parameters to be estimated" << endl ;
  str << "+++++++++++++++++++++++++++++++++" << endl ;
  for (map<patULong,patULong>::const_iterator i = x.parametersToBeEstimated.begin() ;
       i != x.parametersToBeEstimated.end() ;
       ++i) {
    str << x.listOfFixedParameters[i->first].getName() << " Index optimization: " << i->second << endl ;
  }
  str << "Ids of literals" << endl ;
  str << "+++++++++++++++" << endl ;
  for (map<patString,patULong>::const_iterator i = x.listOrganizedByNames.begin() ;
       i != x.listOrganizedByNames.end() ;
       ++i) {
    str << i->first << ": " << i->second << endl ;
  }
  return str ;
}

patString bioLiteralRepository::printFixedParameters(patBoolean estimation) const {
  stringstream str ;
  for (vector<bioFixedParameter>::const_iterator i = listOfFixedParameters.begin() ;
       i != listOfFixedParameters.end() ;
       ++i) {
    str << i->printPythonCode(estimation) << endl ;
  }
  return patString(str.str()) ;
}


pair<patULong,patULong> bioLiteralRepository::addCompositeLiteral(patError*& err) {
  patULong uniqueId = bioParameters::the()->getValueInt("firstIdOfLiterals",err)  ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return pair<patULong,patULong>() ;
  }
  uniqueId += listOrganizedByNames.size() ;

  stringstream str ;
  str << "_bioKeyword_" << uniqueId ;
  patString name = patString(str.str()) ;
  pair<patULong,patULong> theIds = addCompositeLiteral(name,err) ;
  return theIds ;

}


pair<patULong,patULong> bioLiteralRepository::addCompositeLiteral(patString theName, patError*& err) {

  map<patString,patULong>::iterator found =
      listOrganizedByNames.find(theName) ;

  if (found != listOrganizedByNames.end()) {
    pair<patULong,patULong> theIds = getCompositeLiteral(theName, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
    }
    return theIds ;
  }
  patULong uniqueId = bioParameters::the()->getValueInt("firstIdOfLiterals",err)  ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return pair<patULong,patULong>() ;
  }
  uniqueId += listOrganizedByNames.size() ;

  patULong compId = listOfCompositeLiterals.size() ;

  bioCompositeLiteral theLiteral(theName, uniqueId, compId) ;
  listOfCompositeLiterals.push_back(theLiteral) ;
  listOrganizedByNames[theName] = uniqueId ;
  theIdAndTypes[uniqueId] = pair<patULong,bioLiteral::bioLiteralType>(compId,bioLiteral::COMPOSITE) ;

  return pair<patULong,patULong>(uniqueId,compId) ;
}


// bioLiteral* bioLiteralRepository::getLiteral(patULong id, patError*& err) const {

//   if (id >= listOfLiterals.size()) {
//     err = new patErrOutOfRange<patULong>(id,0, listOfLiterals.size()-1) ;
//     WARNING(err->describe()) ;
//     return NULL ;
//   }
//   return listOfLiterals[id] ;
// }

// bioLiteral* bioLiteralRepository::getLiteral(patString theName,
// 					     patError*& err) const {
  
//   map<patString,patULong>::const_iterator found = 
//     listOrganizedByNames.find(theName) ;
//   if (found != listOrganizedByNames.end()) {
//     return listOfLiterals[found->second] ;
//   }
//   else {
//     stringstream str ;
//     str << "No literal " << theName ;
//     err = new patErrMiscError(str.str()) ;
//     WARNING(err->describe()) ;
//     return NULL ;
//   }
  
// }


pair<patULong,patULong>  bioLiteralRepository::getCompositeLiteral(patULong literalId, patError*& err)  {
  for (patULong i = 0 ; i < listOfCompositeLiterals.size() ; ++i) {
    if (listOfCompositeLiterals[i].getId() == literalId) {
      return pair<patULong,patULong>(literalId,i) ;
    }
  }
  return pair<patULong,patULong>(literalId,patBadId) ;
}

pair<patULong,patULong> bioLiteralRepository::getVariable(patString name, patError*& err) const {
  for (patULong i = 0 ; i < listOfVariables.size() ; ++i) {
    if (listOfVariables[i].getName() == name) {
      return pair<patULong,patULong>(listOfVariables[i].getId(),i) ;
    }
  }
  return pair<patULong,patULong>(patBadId,patBadId) ;
}


pair<patULong,patULong> bioLiteralRepository::getCompositeLiteral(patString name, patError*& err) {
  for (patULong i = 0 ; i < listOfCompositeLiterals.size() ; ++i) {
    if (listOfCompositeLiterals[i].getName() == name) {
      return pair<patULong,patULong>(listOfCompositeLiterals[i].getId(),i) ;
    }
  }
  stringstream str ;
  str << "Composite literal " << name << " is unknown" ;
  err = new patErrMiscError(str.str()) ;
  WARNING(err->describe()) ;
  return pair<patULong,patULong>() ;
}

  

pair<patULong,patULong>  bioLiteralRepository::getRandomVariable(patString name, patError*& err) {
  for (patULong i = 0 ; i < listOfRandomVariables.size() ; ++i) {
    if (listOfRandomVariables[i].getName() == name) {
      return pair<patULong,patULong>(listOfRandomVariables[i].getId(),i) ;
    }
  }
  return pair<patULong,patULong>(patBadId,patBadId) ;
}

patIterator<bioFixedParameter*>* bioLiteralRepository::getIteratorFixedParameters() {
  patVectorIterator<bioFixedParameter>* theIter = new patVectorIterator<bioFixedParameter>(&listOfFixedParameters) ;
  return theIter ;
}

patIterator<bioFixedParameter*>* bioLiteralRepository::getSortedIteratorFixedParameters() {
  sortedListOfFixedParameters = listOfFixedParameters ;
  sort(sortedListOfFixedParameters.begin(),sortedListOfFixedParameters.end()) ;
  patVectorIterator<bioFixedParameter>* theIter = new patVectorIterator<bioFixedParameter>(&sortedListOfFixedParameters) ;
  return theIter ;
}



patULong bioLiteralRepository::getLiteralId(patULong typeSpecificId, bioLiteral::bioLiteralType type) {
  switch (type) {
  case bioLiteral::VARIABLE:
    if (typeSpecificId >= listOfVariables.size()) {
      stringstream str ;
      str << "Variable id " << typeSpecificId << " out of range [0," <<listOfVariables.size()-1 ;
      WARNING(str.str()) ;
      return patBadId ;
    }
    return listOfVariables[typeSpecificId].getId() ;
  case bioLiteral::PARAMETER:
    if (typeSpecificId >= listOfFixedParameters.size()) {
      stringstream str ;
      str << "Variable id " << typeSpecificId << " out of range [0," <<listOfFixedParameters.size()-1 ;
      WARNING(str.str()) ;
      return patBadId ;
    }
    return listOfFixedParameters[typeSpecificId].getId() ;
  case bioLiteral::RANDOM:
    if (typeSpecificId >= listOfRandomVariables.size()) {
      stringstream str ;
      str << "Variable id " << typeSpecificId << " out of range [0," <<listOfRandomVariables.size()-1 ;
      WARNING(str.str()) ;
      return patBadId ;
    }
    return listOfRandomVariables[typeSpecificId].getId() ;
  case bioLiteral::COMPOSITE:
    if (typeSpecificId >= listOfCompositeLiterals.size()) {
      stringstream str ;
      str << "Variable id " << typeSpecificId << " out of range [0," <<listOfCompositeLiterals.size()-1 ;
      WARNING(str.str()) ;
      return patBadId ;
    }
    return listOfCompositeLiterals[typeSpecificId].getId() ;
  }
  return patBadId ;
  
}

const bioVariable* bioLiteralRepository::theVariable(patULong typeSpecificId) const {
  if (typeSpecificId >= listOfVariables.size()) {
    return NULL ;
  }
  return &(listOfVariables[typeSpecificId]) ;

}

const bioCompositeLiteral* bioLiteralRepository::theComposite(patULong typeSpecificId) const {
  if (typeSpecificId >= listOfCompositeLiterals.size()) {
    return NULL ;
  }
  return &(listOfCompositeLiterals[typeSpecificId]) ;

}

const bioFixedParameter* bioLiteralRepository::theParameter(patULong typeSpecificId) const {
  if (typeSpecificId >= listOfFixedParameters.size()) {
    return NULL ;
  }
  return &(listOfFixedParameters[typeSpecificId]) ;
  
}

const bioRandomVariable* bioLiteralRepository::theRandomVariable(patULong typeSpecificId) const {
 if (typeSpecificId >= listOfRandomVariables.size()) {
    return NULL ;
  }
  return &(listOfRandomVariables[typeSpecificId]) ;
 
}

const bioLiteral* bioLiteralRepository::getLiteral(patULong id, patError*& err) const {
  pair<patULong,bioLiteral::bioLiteralType> theSpecificLiteral = getIdAndType(id,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  switch (theSpecificLiteral.second){
  case bioLiteral::VARIABLE:
    return theVariable(theSpecificLiteral.first) ;
  case bioLiteral::PARAMETER:
    return theParameter(theSpecificLiteral.first) ;
  case bioLiteral::RANDOM:
    return theRandomVariable(theSpecificLiteral.first) ;
  case bioLiteral::COMPOSITE:
    return theComposite(theSpecificLiteral.first) ;
  default:
    stringstream str ;
    str << "Literal " << id << " is of unknown type" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL ;
  }
    
}


pair<patULong,bioLiteral::bioLiteralType> bioLiteralRepository::getIdAndType(patULong uniqueId,patError*& err) const {
  map<patULong,pair<patULong,bioLiteral::bioLiteralType> >::const_iterator found = theIdAndTypes.find(uniqueId) ;
  if (found == theIdAndTypes.end()) {
    stringstream str ;
    str << "No literal with ID " << uniqueId << " in the repository" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return pair<patULong,bioLiteral::bioLiteralType>() ;
  }
  return found->second ;
}


patReal bioLiteralRepository::getCompositeValue(patULong litId, patULong threadId, patError*& err) {
  if (theCompositeValues.empty()) {
    err = new patErrMiscError("No memory for random variables") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }

  if (threadId >= theCompositeValues.size()) {
    if (theCompositeValues.empty()) {
      err = new patErrMiscError("Empty container") ;
    }
    else {
      err = new patErrOutOfRange<patULong>(threadId,0,theCompositeValues.size()-1) ;
    }
    WARNING(err->describe()) ;
    return patReal() ;
  }
  if (litId >= theCompositeValues[threadId].size()) {
    if (theCompositeValues[threadId].empty() ) {
      err = new patErrMiscError("Empty container") ;
    }
    else {
      err = new patErrOutOfRange<patULong>(litId,0,theCompositeValues[threadId].size()-1) ;
    }
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return theCompositeValues[threadId][litId] ;
}

void bioLiteralRepository::setCompositeValue(patReal v, patULong litId, patULong threadId, patError*& err) {
  if (theCompositeValues.empty()) {
    err = new patErrMiscError("No memory for random variables") ;
    WARNING(err->describe()) ;
    return ;
  }
  if (threadId >= theCompositeValues.size()) {
    if (theCompositeValues.empty()) {
      err = new patErrMiscError("Empty container") ;
    }
    else {
      err = new patErrOutOfRange<patULong>(threadId,0,theCompositeValues.size()-1) ;
    }
    WARNING(err->describe()) ;
    return ;
  }
  if (litId >= theCompositeValues[threadId].size()) {
    if (theCompositeValues[threadId].empty() ) {
      err = new patErrMiscError("Empty container") ;
    }
    else {
      err = new patErrOutOfRange<patULong>(litId,0,theCompositeValues[threadId].size()-1) ;
    }
    WARNING(err->describe()) ;
    return ;
  }
  theCompositeValues[threadId][litId] = v ;
}


patReal bioLiteralRepository::getRandomVariableValue(patULong litId, patULong threadId, patError*& err) {
  if (theRandomValues.empty()) {
    err = new patErrMiscError("No memory for random variables") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  if (threadId >= theRandomValues.size()) {
    if (theRandomValues.empty()) {
      err = new patErrMiscError("Empty container") ;
    }
    else {
      err = new patErrOutOfRange<patULong>(threadId,0,theRandomValues.size()-1) ;
    }
    WARNING(err->describe()) ;
    return patReal() ;
  }
  if (litId >= theRandomValues[threadId].size()) {
    if (theRandomValues[threadId].empty() ) {
      err = new patErrMiscError("Empty container") ;
    }
    else {
      err = new patErrOutOfRange<patULong>(litId,0,theRandomValues[threadId].size()-1) ;
    }
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return theRandomValues[threadId][litId] ;
}

void bioLiteralRepository::setRandomVariableValue(patReal v, patULong litId, patULong threadId, patError*& err) {
  if (theRandomValues.empty()) {
    err = new patErrMiscError("No memory for random variables") ;
    WARNING(err->describe()) ;
    return  ;
  }
  if (threadId >= theRandomValues.size()) {
    if (theRandomValues.empty()) {
      err = new patErrMiscError("Empty container") ;
    }
    else {
      err = new patErrOutOfRange<patULong>(threadId,0,theRandomValues.size()-1) ;
    }
    WARNING(err->describe()) ;
    return ;
  }
  if (litId >= theRandomValues[threadId].size()) {
    if (theRandomValues[threadId].empty() ) {
      err = new patErrMiscError("Empty container") ;
    }
    else {
      err = new patErrOutOfRange<patULong>(litId,0,theRandomValues[threadId].size()-1) ;
    }
    WARNING(err->describe()) ;
    return ;
  }
  theRandomValues[threadId][litId] = v ;
}

void bioLiteralRepository::prepareMemory() {

    theCompositeValues.
      resize(bioParameters::the()->getValueInt("numberOfThreads"),
	     vector<patReal>(listOfCompositeLiterals.size(),0.0)) ;
  

    theRandomValues.
      resize(bioParameters::the()->getValueInt("numberOfThreads"),
	     vector<patReal>(listOfRandomVariables.size(),0.0)) ;
}

///
// This is an inefficient implementation. But, in principle, this
// should be called only once, when the user expressions are built.
// @return patBadId if there is no variables, or the id of the last
// column if there are variables.
patULong bioLiteralRepository::getLastColumnId() const {
  if (listOfVariables.empty()) {
    return patBadId ;
  }
  patULong lastCol = 0 ;
  for (vector<bioVariable>::const_iterator i = listOfVariables.begin() ;
       i != listOfVariables.end() ;
       ++i) {
    if (i->getColumnId() > lastCol) {
      lastCol = i->getColumnId() ;
    }
  }
  return lastCol ;
}

patULong bioLiteralRepository::getColumnIdOfVariable(patString name,
						     patError*& err) const {
  pair<patULong,patULong> varIds = getVariable(name,err)  ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patBadId ;
  }
  const bioVariable* theVar = theVariable(varIds.second) ;
  if (theVar == NULL) {
    stringstream str ;
    str << "The ID of variable " << name << " could not be found" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patBadId ;
  }
  return theVar->getColumnId() ;
}

void bioLiteralRepository::setBetaValue(patString name, 
					patReal value, 
					patError*& err) {
  patBoolean found = patFALSE ;
  for (patULong i = 0 ; i < listOfFixedParameters.size() ; ++i) {
    if (listOfFixedParameters[i].getName() == name) {
      listOfFixedParameters[i].setValue(value) ;
      found = patTRUE ;
    }
  }
  if (!found) {
    stringstream str ;
    str << "Parameter " << name << " is unknown" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
}

int bioLiteralRepository::size(){
  int size=0;

  size+=listOfVariables.size() ;
  size+=listOfFixedParameters.size() ;
  // The sorted list is used only for output and created by the
  // getSortedIteratorFixedParameters() method.
  size+=sortedListOfFixedParameters.size() ;
  size+=listOfRandomVariables.size() ;
  // Composite literals are intermediate variables designed to simplify expressions
  size+=listOfCompositeLiterals.size() ;
  size+=flags.size() ;
  size+=listOrganizedByNames.size() ;
//   map<patString,patULong> idsOfVariables ;
//   map<patString,patULong> idsOfRandomVariables ;
//   map<patString,patULong> idsOfParameters ;
//   map<patString,patULong> idsOfCompositeLiterals ;

  // Given a unique id, store thr type ans type specific id of the literal
 size+=theIdAndTypes.size() ;

  // First: index in the vector listOfFixedParameters
  // Second: index in the set of decision variables for the optimization problem
  size+=parametersToBeEstimated.size() ;

  size+=theCompositeValues.size() ;
  // Used for integration
  size+=theRandomValues.size() ;

  size+=userExpressions.size() ;

  return size;
}

void bioLiteralRepository::reset() {
  listOfVariables.erase(listOfVariables.begin(),listOfVariables.end()) ;
  listOfFixedParameters.erase(listOfFixedParameters.begin(),listOfFixedParameters.end()) ;
  sortedListOfFixedParameters.erase(sortedListOfFixedParameters.begin(),sortedListOfFixedParameters.end()) ;
  listOfRandomVariables.erase(listOfRandomVariables.begin(),listOfRandomVariables.end()) ;
  listOfCompositeLiterals.erase(listOfCompositeLiterals.begin(),listOfCompositeLiterals.end()) ;
  flags.erase(flags.begin(),flags.end()) ;
  listOrganizedByNames.erase(listOrganizedByNames.begin(),listOrganizedByNames.end()) ;
  theIdAndTypes.erase(theIdAndTypes.begin(),theIdAndTypes.end()) ;
  parametersToBeEstimated.erase(parametersToBeEstimated.begin(),parametersToBeEstimated.end()) ;
  theCompositeValues.erase(theCompositeValues.begin(),theCompositeValues.end()) ;
  theRandomValues.erase(theRandomValues.begin(),theRandomValues.end()) ;
  userExpressions.erase(userExpressions.begin(),userExpressions.end()) ;

}
