//-*-c++-*------------------------------------------------------------
//
// File name : bioArithIterator.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Jun 16 12:41:39 2009
//
//--------------------------------------------------------------------

#ifndef bioArithIterator_h
#define bioArithIterator_h


#include "bioLiteral.h"
#include "bioIteratorInfo.h"
#include "bioArithUnaryExpression.h"
#include "bioIteratorSpan.h"

class bioSample ;
/*!
Class implementing a node of the tree representing a sum expression
*/
class bioArithIterator : public bioArithUnaryExpression {

public:
  
  bioArithIterator(bioExpressionRepository* rep,
		   patULong par,
		   patULong left,
		   patString anIterator,
		   patError*& err) ;
  
  ~bioArithIterator() ;

public:
  virtual patString getExpression(patError*& err) const ;
  virtual patString theIterator() const ;


  virtual patBoolean isSum() const = PURE_VIRTUAL ;
  virtual patBoolean isProd() const = PURE_VIRTUAL ;
  virtual patBoolean containsAnIterator() const ;
  virtual patBoolean containsAnIteratorOnRows() const ;
protected:

  patString theIteratorName ;
  bioIteratorType theIteratorType ;


};

#endif
