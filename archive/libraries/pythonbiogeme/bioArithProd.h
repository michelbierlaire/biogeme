//-*-c++-*------------------------------------------------------------
//
// File name : bioArithProd.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Wed Jul  8 15:06:48  2009
//
//--------------------------------------------------------------------

#ifndef bioArithProd_h
#define bioArithProd_h


#include "bioArithIterator.h"


/*!
Class implementing a node of the tree representing a product expression
*/
class bioArithProd : public bioArithIterator {

public:
  
  bioArithProd(bioExpressionRepository* rep,
	       patULong par,
	       patULong left,
	       patString anIterator,
               patBoolean isPositive,
	       patError*& err) ;
  ~bioArithProd() ;

public:
  virtual patString getOperatorName() const;
  virtual bioExpression* getDerivative(patULong aLiteralId, patError*& err) const ;

  virtual bioArithProd* getDeepCopy(bioExpressionRepository* rep,
				    patError*& err) const ;
  virtual bioArithProd* getShallowCopy(bioExpressionRepository* rep,
				    patError*& err) const ;
  virtual patBoolean isSum() const  ;
  virtual patBoolean isProd() const  ;
  virtual patString getExpressionString() const ;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  ;
  virtual patULong getNumberOfOperations() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian, 				  patBoolean debugDerivatives,
patError*& err) ;

private:
  vector<patULong> theCompositeLiteralsIds ;
  patBoolean allEntriesArePositive ;
  patULong accessToFirstRow ;
};

#endif
