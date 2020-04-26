//-*-c++-*------------------------------------------------------------
//
// File name : patValueVariables.cc
// Author :    Michel Bierlaire
// Date :      Thu Nov 23 14:43:48 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <cassert>
#include <sstream>
#include "patValueVariables.h"
#include "patErrMiscError.h"
#include "patErrOutOfRange.h"
#include "patErrNullPointer.h"
#include "patBisonSingletonFactory.h"
patValueVariables::patValueVariables() : x(NULL), 
					 attributes(NULL), 
					 randomDraws(NULL) {

}

patValueVariables* patValueVariables::the() {
  return patBisonSingletonFactory::the()->patValueVariables_the() ;
}

void patValueVariables::setValue(patString variable, patReal value) {
  values[variable] = value ;
}

patReal patValueVariables::getValue(patString variable,
				    patError*& err) {

  map<patString,patReal>::const_iterator i = values.find(variable) ;

  if (i == values.end()) {
    stringstream str ;
    str << "Unknown variable " << variable ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return i->second ;
}


void patValueVariables::setVariables(patVariables* y) {
  x = y ;
}

patReal patValueVariables::getVariable(unsigned long index, 
				       patError*& err) {

  
  if (x == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
//   DEBUG_MESSAGE("Variables of size " << x->size()) ;
//   for (unsigned long ii =  0 ;
//        ii < x->size() ;
//        ++ii) {
//     DEBUG_MESSAGE("** x[" << ii << "]=" << (*x)[ii]) ;
//   } 

  if (index >= x->size()) {
    err = new patErrOutOfRange<unsigned long>(index,0,x->size()-1) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return (*x)[index] ;
}

ostream& operator<<(ostream &str, const patValueVariables& x) {
  str << "Variables" << endl ;
  for (unsigned long i = 0 ; i < x.x->size() ; ++i) {
    str << "  x[" << i << "]=" << (*x.x)[i] << endl ;
  }
  str << "Values" << endl ;
  for (map<patString,patReal>::const_iterator i = x.values.begin() ;
       i != x.values.end() ;
       ++i) {
    str << i->first << "=" << i->second << endl ;
  }
  return str ;
}

patReal patValueVariables::getAttributeValue(unsigned long attrId,
					     patError*& err) {
  if (attributes == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  if (attrId >= attributes->size()) {
    err = new patErrOutOfRange<unsigned long>(attrId,0,attributes->size()-1) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return (*attributes)[attrId].value ;
}

patReal patValueVariables::getRandomDrawValue(unsigned long attrId,
					      patError*& err) {

  if (randomDraws == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  if (attrId >= randomDraws->size()) {
    err = new patErrOutOfRange<unsigned long>(attrId,0,randomDraws->size()-1) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return (*randomDraws)[attrId] ;
}


void patValueVariables::setAttributes(vector<patObservationData::patAttributes>* y) {
  attributes = y ;
}

void patValueVariables::setRandomDraws(patVariables* y) {
  randomDraws = y ;
}

patBoolean patValueVariables::areAttributesAvailable() {
  return (attributes != NULL) ;
}
  
patBoolean patValueVariables::areVariablesAvailable() {
  return(x != NULL) ;
}
