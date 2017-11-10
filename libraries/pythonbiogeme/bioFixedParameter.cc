//-*-c++-*------------------------------------------------------------
//
// File name : bioFixedParameter.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Apr 28 10:54:02 2009
//
//--------------------------------------------------------------------

#include <sstream>
#include "bioFixedParameter.h"
#include "patFormatRealNumbers.h"
#include "patMath.h"
#include "patPValue.h"

patReal bioFixedParameter::getValue() const {
  return currentValue ;
}
  
void bioFixedParameter::setValue(patReal val) {
  currentValue = val ;
}

bioFixedParameter::bioFixedParameter(patString theName, 
				     patULong uniqueId,
				     patULong pId,
				     patULong eId,
				     patReal def, 
				     patReal lb, 
				     patReal ub, 
				     patBoolean f,
				     patString desc) :
  bioLiteral(theName,uniqueId),
  theParameterId(pId),
  theEstimatedId(eId),
  defaultValue(def),
  currentValue(def),
  lowerBound(lb),
  upperBound(ub),
  isFixed(f),
  latexName(desc) {
  
}

bioFixedParameter::bioFixedParameter(patString theName, 
				     patULong uniqueId,
				     patULong id,
				     patULong eId,
				     patReal def):
  bioLiteral(theName,uniqueId),
  theParameterId(id),
  theEstimatedId(eId),
  defaultValue(def),
  currentValue(def),
  lowerBound(-1.0e20),
  upperBound(1.0e20),
  isFixed(patFALSE),
  latexName(theName) {

}

bioLiteral::bioLiteralType bioFixedParameter::getType() const {
  return PARAMETER ;
}

ostream& operator<<(ostream &str, const bioFixedParameter& x) {
  str << "x[" ;
  str << x.theParameterId << "]=" << x.name << " (" << x.defaultValue << "," 
      <<  x.currentValue << ") [" << x.lowerBound << "," << x.upperBound << "] " ;
  if (x.isFixed) {
    str << "FIXED " ;
  }
  else {
    str << "FREE " ;
  }
  str << "ID=" << x.uniqueId ;

  return str ;
}

patReal bioFixedParameter::getLowerBound() const {
  return lowerBound ;
}

patReal bioFixedParameter::getUpperBound() const {
  return upperBound ;
}

patBoolean bioFixedParameter::isFixedParam() const {
  return (isFixed != 0) ;
}

patString bioFixedParameter::printPythonCode(patBoolean estimation) const {
  stringstream str ;
  if (estimation) {
    str << name << " = Beta(";
    str << "'" << name << "'," ;
    str << currentValue << "," ; 
    str << lowerBound << "," ; 
    str << upperBound << "," ; 
    if (isFixed) {
      str << "1" ; 
    }
    else {
      str << "0" ; 
    }
    str << "," ;
    str << "'" << latexName << "'" ;
    str << " )" << endl ;
  }
  else {
    str << name << " = " << currentValue << endl ;
  }
  return patString(str.str());
}

patULong bioFixedParameter::getParameterId() const {
  return theParameterId ;
}

patULong bioFixedParameter::getEstimatedParameterId() const {
  return theEstimatedId ;
}

patString bioFixedParameter::getLaTeXRow(patReal stdErr, patError*& err) const {
  patFormatRealNumbers theNumber ;
  stringstream latexFile ;
  latexFile << latexName << " & " ;
  
  // Estimated valued
  patString anumber = theNumber.formatParameters(currentValue) ;
  replaceAll(&anumber,patString("."),patString("&")) ;
  latexFile << anumber << " & "  ;

  // Std error
  anumber = theNumber.formatParameters(stdErr) ;
  replaceAll(&anumber,patString("."),patString("&")) ;
  latexFile << anumber << " & "  ;

  // t test
  patReal ttest = currentValue / stdErr ;
  anumber = theNumber.formatTTests(ttest) ;
  replaceAll(&anumber,patString("."),patString("&")) ;
  latexFile << anumber  << " & " ;

  // p value
  patReal pvalue = patPValue(patAbs(ttest),err) ;
  anumber= theNumber.formatTTests(pvalue) ;
  replaceAll(&anumber,patString("."),patString("&")) ;
  latexFile << anumber ;

  return patString(latexFile.str()) ;
  

}
