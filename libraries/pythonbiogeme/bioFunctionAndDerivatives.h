//-*-c++-*------------------------------------------------------------
//
// File name : bioFunctionAndDerivatives.h
// Author :    Michel Bierlaire
// Date :      Sun May 15 16:02:39 2011
//
//--------------------------------------------------------------------

#ifndef bioFunctionAndDerivatives_h
#define bioFunctionAndDerivatives_h

#include <iostream>
#include "patType.h"
#include "trHessian.h"

class bioFunctionAndDerivatives {
  
public:

  bioFunctionAndDerivatives(patULong n = 0) ;
  ~bioFunctionAndDerivatives() ;
  bioFunctionAndDerivatives& operator=(bioFunctionAndDerivatives& obj) ;
  patReal theFunction ;
  vector<patReal> theGradient ;
  trHessian *theHessian ;
  patBoolean empty() const ;
  patString printSize() ;
  void resize(patULong s) ;
  void resize(patULong s,patReal r) ;
  //Return  the maximum difference across all quantities
  patReal compare(const bioFunctionAndDerivatives& x, patError*& err) ;

};

ostream& operator<<(ostream &str, const bioFunctionAndDerivatives& x) ;


#endif
