//-*-c++-*------------------------------------------------------------
//
// File name : bioLiteralValues.h
// Author :    Michel Bierlaire
// Date :      Sun Oct 17 11:41:42 2010
//
//--------------------------------------------------------------------

#ifndef bioLiteralValues_h
#define bioLiteralValues_h

#include <map>
#include "patError.h"

/*!
Stores numerical values for literals. Used when the expression are used to compute numerical values and not to generate code.
*/

class bioLiteralValues {

  friend class bioPythonSingletonFactory ;
  friend ostream& operator<<(ostream &str, const bioLiteralValues& x) ;  
public:
  static bioLiteralValues* the() ;
  void setValue(patString l, patReal v, patError*& err) ;
  patReal getValue(patString l, patError*& err) ;
  void eraseValues() ;
private:
  bioLiteralValues() ;
  map<patString,patReal> values ;
};

#endif
