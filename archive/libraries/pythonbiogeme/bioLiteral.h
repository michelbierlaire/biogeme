//-*-c++-*------------------------------------------------------------
//
// File name : bioLiteral.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Mon Apr 27 15:54:09 2009
//
//--------------------------------------------------------------------

#ifndef bioLiteral_h
#define bioLiteral_h

/*!
Class defining a literal, that is a parameter (fixed or random) or a
variable of the model
*/

#include "patError.h"
#include "patString.h"
#include "patConst.h"
#include "patType.h"



class bioLiteral {
  
  friend class bioLiteralRepository ;
  friend ostream& operator<<(ostream &str, const bioLiteral& x) ;
 public:

  virtual patBoolean operator<(const bioLiteral& x) const ;

  typedef enum  {
    VARIABLE,
    PARAMETER,
    RANDOM,
    COMPOSITE
  } bioLiteralType ;
  
  virtual bioLiteralType getType() const = PURE_VIRTUAL ;
  patString getName() const ;
  patULong getId() const ; 

 protected:
  /*!
    Only the repository can create a parameter
  */
  bioLiteral(patString theName, patULong theId) ;
  
 protected:
  patString name ;
  patULong uniqueId ;

};

#endif
