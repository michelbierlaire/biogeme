//-*-c++-*------------------------------------------------------------
//
// File name : patOneZhengFosgerau.h
// Author :    \URL[Michel Bierlaire]{http://transp-or2.epfl.ch}
// Date :      Tue Dec 11 14:40:07 2007
//
//--------------------------------------------------------------------

#ifndef patOneZhengFosgerau_h
#define patOneZhengFosgerau_h

#include "patType.h"
#include "patArithNode.h"

/**
   A Zheng-Fosgerau test can be either a probability or an expression
 */

class patOneZhengFosgerau {
 public:
  patOneZhengFosgerau(patReal b, 
		   patReal lb, 
		   patReal up, 
		   patString altName, 
		   patString aName) ;
  patOneZhengFosgerau(patReal b, 
		   patReal lb, 
		   patReal up, 
		   patString aName, 
		   patArithNode* expr) ;

  patBoolean isProbability() const ;
  patString getAltName() const ;
  patULong getAltInternalId() const ;
  patBoolean trim(patReal x)  ;
  patReal bandwidth ;
  void resetTrimCounter() ;
  patString latexDescription() const ;
  patString describeVariable() const ;
  patString describeTrimming() const ;
  patString describeTrimmingLatex() const ;
  patULong expressionIndex ;
  patArithNode* expression ;
  patString getDefaultExpressionName() ;
  patReal getLowerBound() const ;
  patReal getUpperBound() const ;
  patString getTheName() const ;
 private :
  patReal lowerBound ;
  patReal upperBound ;
  // irrelevant if expression != NULL
  patString altName ;
  patString theName ;
  unsigned long trimDown ;
  unsigned long trimUp ;
  unsigned long noTrim ;
};

ostream& operator<<(ostream &str, const patOneZhengFosgerau& x) ;

#endif
