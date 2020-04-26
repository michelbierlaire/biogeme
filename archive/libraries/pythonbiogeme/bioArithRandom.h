//-*-c++-*------------------------------------------------------------
//
// File name : bioArithRandom.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Thu Jul 30 17:13:52 2009
//
//--------------------------------------------------------------------

#ifndef bioArithRandom_h
#define bioArithRandom_h

#include "bioRandomDraws.h"
#include "bioArithUnaryExpression.h"

/*!
Class defining an interface for a random parameter uusing draws
*/

// The child computes the number of the individual or the observation. Typically, it is read in the file.
class bioArithRandom : public bioArithUnaryExpression {

public:

  bioArithRandom(bioExpressionRepository* rep,
		 patULong par, 
		 patULong individual, 
		 patULong id, 
		 bioRandomDraws::bioDrawsType  type,
		 patError*& err) ;

  ~bioArithRandom() ;
  patString getOperatorName() const  ;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)   ;
  bioExpression* getDerivative(patULong aLiteralId, patError*& err) const ;
  bioArithRandom* getDeepCopy(bioExpressionRepository* rep, 
			      patError*& err) const ;
  bioArithRandom* getShallowCopy(bioExpressionRepository* rep, 
			      patError*& err) const ;
  patULong getNumberOfOperations() const ;
  patString getExpressionString() const  ;
  patString getExpression(patError*& err) const  ;
  patBoolean dependsOf(patULong aLiteralId) const  ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian, 				  patBoolean debugDerivatives,
 patError*& err) ;
  vector<patULong> getListOfDraws(patError*& err) const ;
  
  void checkMonteCarlo(patBoolean insideMonteCarlo, patError*& err) ;

protected:
  patULong variableIdForDraws ;
  bioRandomDraws::bioDrawsType type ;

};


#endif
