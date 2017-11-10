//-*-c++-*------------------------------------------------------------
//
// File name : bioCompositeLiteral.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Apr  5 11:28:44  2011
//
//--------------------------------------------------------------------

#ifndef bioCompositeLiteral_h
#define bioCompositeLiteral_h

/*!
This literal represents an intermediate computation. The associated expression must be defined before its use by the user.
*/

#include "bioLiteral.h"

class bioCompositeLiteral : public bioLiteral {

  friend class bioLiteralRepository ;
  friend ostream& operator<<(ostream &str, const bioCompositeLiteral& x) ;

public:

  patReal getValue(patError*& err) const ;
  void setValue(patReal val, patError*& err) ;
  bioLiteralType getType() const ;
  patULong getCompositeId() const ;

protected:
  /*!
    Only the repository can create a composite literal
  */
  bioCompositeLiteral(patString theName, patULong uniqueId, patULong clId) ;

  patReal currentValue ;
  patULong theCompositeLiteralId ;
};

#endif
