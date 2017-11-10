//-*-c++-*------------------------------------------------------------
//
// File name : bioVariable.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Mon Apr 27 16:17:59 2009
//
//--------------------------------------------------------------------

#ifndef bioVariable_h
#define bioVariable_h

#include "bioLiteral.h"

/*!
Class defining a variable of the model
*/

class bioVariable: public bioLiteral {

  friend class bioLiteralRepository ;

  friend class bioArithVariable ;
  
public:
  bioLiteralType getType() const ;
  patULong getColumnId() const ;
protected:
  /*!
    Only the repository can create a parameter
  */
  bioVariable(patString theName, patULong uniqueId, patULong vId, patULong theColumnId = patBadId) ;

protected:

  // Consecutive ID for variables
  patULong theVariableId ;
  // Id of the column in the datafile
  patULong columnId ;

};

#endif 
