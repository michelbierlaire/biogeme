//-*-c++-*------------------------------------------------------------
//
// File name : bioRandomVariable.h
// Author :    Michel Bierlaire
// Date :      Thu Apr  1 12:20:21 2010
//
//--------------------------------------------------------------------

#ifndef bioRandomVariable_h
#define bioRandomVariable_h

#include "bioLiteral.h"

/*!
Class defining a random variable of the model, used in numerical integration
*/

class bioRandomVariable: public bioLiteral {

  friend class bioLiteralRepository ;
  friend class bioArithRandomVariable ;
public:
  bioLiteralType getType() const ;
protected:
  /*!
    Only the repository can create a variable
  */
  bioRandomVariable(patString theName, patULong uniqueId, patULong rvId) ;

  //  bioRandomVariable(const bioRandomVariable&) ;
protected:

  // Consecutive ID for variables
  patULong theRandomVariableId ;
};

#endif 
