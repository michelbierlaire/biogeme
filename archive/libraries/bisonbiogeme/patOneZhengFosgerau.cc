//-*-c++-*------------------------------------------------------------
//
// File name : patOneZhengFosgerau.cc
// Author :    Michel Bierlaire
// Date :      Tue Dec 11 14:47:24 2007
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include "patModelSpec.h"
#include "patOneZhengFosgerau.h"
#include "patFormatRealNumbers.h"

patOneZhengFosgerau::patOneZhengFosgerau(patReal b, 
				   patReal lb, 
				   patReal ub, 
				   patString an, 
				   patString name) : 
  bandwidth(b),
  expression(NULL),
  lowerBound(lb),
  upperBound(ub),
  altName(an),
  theName(name)
{
  
  
}

patOneZhengFosgerau::patOneZhengFosgerau(patReal b, 
				   patReal lb, 
				   patReal ub, 
				   patString name,
				   patArithNode* expr) :
  bandwidth(b),
  expressionIndex(patBadId),
  expression(expr),
  lowerBound(lb),
  upperBound(ub),
  theName(name)
{

}

patBoolean patOneZhengFosgerau::isProbability() const {
  return (expression == NULL) ;
}

patString patOneZhengFosgerau::getAltName() const {
  return altName ; 
}

patULong patOneZhengFosgerau::getAltInternalId() const {
  return patModelSpec::the()->getAltInternalId(altName) ;
}


patBoolean patOneZhengFosgerau::trim(patReal x)  {
  if (x < lowerBound) {
    ++trimDown ;
    return patTRUE ;
  }
  if (x > upperBound) {
    ++trimUp ;
    return patTRUE ;
  }
  ++noTrim ;
  return patFALSE ;
}

void patOneZhengFosgerau::resetTrimCounter() {
  trimDown = 0 ;
  trimUp = 0 ;
  noTrim = 0 ;
}

patString patOneZhengFosgerau::describeTrimming() const {
  patFormatRealNumbers theNumber;
  stringstream str ;
  patULong total = trimDown + trimUp + noTrim ;
  str << " [" << lowerBound 
      << ":" << upperBound << "]: " 
      << trimDown << " < " << noTrim << " > " << trimUp << " <==> " 
      << theNumber.format(patFALSE,patFALSE,2,100.0 * trimDown/total) 
      << "% < " 
      << theNumber.format(patFALSE,patFALSE,2,100.0 * noTrim/total) 
      << "% > " 
      << theNumber.format(patFALSE,patFALSE,2,100.0 * trimUp/total) 
      << "%" ;
  return patString (str.str()) ;
}

patString patOneZhengFosgerau::describeTrimmingLatex() const {
  patFormatRealNumbers theNumber;
  stringstream str ;
  patULong total = trimDown + trimUp + noTrim ;
  str << " [" << lowerBound 
      << ":" << upperBound << "]: " 
      << trimDown << " $\\langle$ " << noTrim << " $\\rangle$ " << trimUp << " $\\Leftrightarrow$ " 
      << theNumber.format(patFALSE,patFALSE,2,100.0 * trimDown/total) 
      << "\\% $\\langle$ " 
      << theNumber.format(patFALSE,patFALSE,2,100.0 * noTrim/total) 
      << "\\% $\\rangle$ " 
      << theNumber.format(patFALSE,patFALSE,2,100.0 * trimUp/total) 
      << "\\%" ;
  return patString (str.str()) ;
}

patString patOneZhengFosgerau::latexDescription() const {
  return describeVariable() ;
}
patString patOneZhengFosgerau::describeVariable() const {
  stringstream str ;
  if (isProbability()) {
    str << theName << " = Proba(" << getAltName() << ")" ;
  }
  else {
    str << theName << " = " << *expression ;
  }
  return patString(str.str()) ;
}

ostream& operator<<(ostream &str, const patOneZhengFosgerau& x) {
  str << x.describeVariable() ;
  return str ;
}

patString patOneZhengFosgerau::getDefaultExpressionName() {
  stringstream str ;
  str << "ZhengFosgerauExpr" << expressionIndex ;
  return patString(str.str()) ;
}

patReal patOneZhengFosgerau::getLowerBound() const {
  return lowerBound ;
}

patReal patOneZhengFosgerau::getUpperBound() const {
  return upperBound ;
}

patString patOneZhengFosgerau::getTheName() const {
  return theName ;
}
