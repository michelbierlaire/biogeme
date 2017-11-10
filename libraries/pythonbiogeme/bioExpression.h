//-*-c++-*------------------------------------------------------------
//
// File name : bioExpression.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Mon Apr 27 16:31:01 2009
//
//--------------------------------------------------------------------

#ifndef bioExpression_h
#define bioExpression_h

#include "patError.h"
#include <map>
#include <set>
#include <list>
#include <ostream>

#include "bioIteratorInfo.h"
#include "bioIteratorSpan.h"
#include "patVariables.h" 
#include "bioFunctionAndDerivatives.h"

#include "patLap.h"
class bioExpressionRepository ;
class bioArithListOfExpressions ;
class bioArithLikelihoodFctAndGrad ;
class bioArithNamedExpression ;
class bioSample ;
class bioReporting ;
class trHessian ;
/*!
 This virtual class represents a generic arithmetic expression.

*/

class bioLiteral ;

class bioExpression {

  friend ostream& operator<<(ostream& str, const bioExpression& x) ;
  friend class bioExpressionRepository ;
 public:

//   typedef  enum {
//     UNARY_OP,
//     BINARY_OP,
//     LITERALS_OP,
//     RANDOM_OP,
//     CONSTANT_OP,
//     SUM_OP,
//     PROD_OP,
//     ENUM_OP,
//     MERGED_OP,
//     SEQUENCE_OP,
//     DICTIONARY_OP,
//     INTEGRAL_OP,
//     DERIVATIVE_OP,
//     UNDEFINED_OP,
//   } bioOperatorType ;


  bioExpression(bioExpressionRepository* rep, patULong par) ;

  virtual patBoolean operator==(const bioExpression& x) ;
  virtual patBoolean operator!=(const bioExpression& x) ;

  virtual ~bioExpression() ;

  /*!
    \return patTRUE if the node has no parent
  */
  virtual patBoolean isTop() const  ;

  virtual void setTop(patBoolean isTop);
  
  /*!  return patTRUE if the expression is a sum iterator. Typically
    used to identify a loglikelihood function.
  */
  virtual patBoolean isSumIterator() const ;

  /*!
    return patTRUE if the expression is structurally 0 so that there is no need to evaluate it. In the base class, it return patFALSE.
  */
  virtual patBoolean isStructurallyZero() const ;
  
  /*!
    return patTRUE if the expression is structurally 1. In the base class, it return patFALSE.
  */
  virtual patBoolean isStructurallyOne() const ;
  
  virtual vector<patULong> getListOfDraws(patError*& err) const ;
  
  virtual patString check(patError*& err) const = PURE_VIRTUAL ;
  
  virtual patBoolean containsMonteCarlo() const ;
  /**
     \return pointer to the iterator information if the top level node is an iterator, NULL otherwise. Not recursive. 
   */
  virtual patString theIterator() const ;

  

  /*!
    \return name of the operator
  */
  virtual patString getOperatorName() const = PURE_VIRTUAL ;
  
  /*!
    \return value of the expression
    \param err ref. of the pointer to the error object.
  */
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  = PURE_VIRTUAL ;


  /*!
    \return value and gradient of the expression
    \param err ref. of the pointer to the error object.
  */
  virtual bioFunctionAndDerivatives* 
  getNumericalFunctionAndGradient(vector<patULong> literalIds, 
				  patBoolean computeHessian, 
				  patBoolean debugDerivatives,
				  patError*& err) = PURE_VIRTUAL  ;

  /*!
    \return value and gradient of the expression, using finite differences
    \param err ref. of the pointer to the error object.
  */
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndFinDiffGradient(vector<patULong> literalIds, patError*& err)  ;


  /*!
    \return value of the derivative w.r.t literal
    \param index of the literal involved in the derivative
    \param err ref. of the pointer to the error object.
  */
  virtual bioExpression* getDerivative(patULong aLiteralId, 
				       patError*& err) const 
    = PURE_VIRTUAL ;



  /*!
    \return A sequence of expressions containing the expression itself, followed by the derivative of it w.r.t. each literal is the list. Designed to recycle intermediary expressions.
  */
  // virtual bioArithListOfExpressions* getFunctionAndDerivatives(vector<patULong> literalIds, 
  // 							       patError*& err) const = PURE_VIRTUAL ;

  /*!
    \return In the special case of a loglikelihood function, we create an object for function and derivatives that maintain the iterator structure. 
  */
  virtual bioArithLikelihoodFctAndGrad* getLikelihoodFunctionAndDerivatives(vector<patULong> literalIds, 
									    patError*& err) const ;

  /*!
     \return printed expression
   */
  virtual patString getExpression(patError*& err) const  = PURE_VIRTUAL ;

 /*!
    Mainly for debugging
   */
  virtual patString getInfo() const ;

  /*!
     Create a deep copy of the expression and returns a pointer to it.
     It means that new instances of the children are created.
   */
  virtual bioExpression* getDeepCopy(bioExpressionRepository* rep, 
				     patError*& err) const = PURE_VIRTUAL ;
  

  /*!
     Create a shallow copy of the expression and returns a pointer to
     it.  It means that no new instance of the children are
     created. It is typically called by the repository
   */
  virtual bioExpression* getShallowCopy(bioExpressionRepository* rep, 
				     patError*& err) const = PURE_VIRTUAL ;
  
  /*!
   */
  virtual patBoolean dependsOf(patULong aLiteralId) const  = PURE_VIRTUAL ;

  /*!
   */
  patULong getId() const;
  /*!
    The name is "bio" + Id
   */
  patString getUniqueName() const;

  /*!
   */
  virtual patBoolean containsAnIterator() const  = PURE_VIRTUAL ;

  /*!
   */
  virtual patBoolean containsAnIteratorOnRows() const  = PURE_VIRTUAL ;

  /*!
   */
  virtual patBoolean containsAnIntegral() const  = PURE_VIRTUAL ;

  /*!
   */
  virtual patBoolean containsASequence() const  = PURE_VIRTUAL ;


  /*
    Count the number of operations
   */
  virtual patULong getNumberOfOperations() const  = PURE_VIRTUAL ;

  /*!
   */
  virtual void simplifyZeros(patError*& err) = PURE_VIRTUAL ;

  /**
   */
  virtual patBoolean isConstant() const ;

  /**
   */
  virtual patBoolean isSimulator() const ;

  /**
   */
  virtual patBoolean isBayesian() const ;

  /**
   */
  virtual patBoolean isSequence() const ;
  /**
   */
  virtual patBoolean isLiteral() const ;
  /**
   */
  bioExpression* getParent() ;

  void setVariables(const patVariables* x)  ;

  void setDraws(pair<patReal**, patReal**> d) ;
  

  void setSample(bioSample* s) ;
  void setReport(bioReporting* s) ;
  
  void setCurrentSpan(bioIteratorSpan aSpan)  ;
  
  void setThreadSpan(bioIteratorSpan aSpan) ;

  // This function returns a non null pointer only for the top sum iterator.
  virtual trHessian* getBhhh() ;

  patULong getThreadId() const ;
  /**
     Compute a string that represents the expression. It is designed to replace the expression itself when used only for comparison purposes. 
     Code:
     +{expr1}{expr2}: binary plus
     -{expr1}{expr2}: binary minus
     *{expr1}{expr2}: multiplication
     /{expr1}{expr2}: division
     ^{expr1}{expr2}: power
     &{expr1}{expr2}: and
     |{expr1}{expr2}: or
     ={expr1}{expr2}: equal
     !={expr1}{expr2}: not equal
     <{expr1}{expr2}: lesser than
     <={expr1}{expr2}: lesser or equal to
     >{expr1}{expr2}: greater than
     >={expr1}{expr2}: greater or equal to 
     $A{expr}: abs
     $D[expr][{expr1}...{exprN}]: dictionary (bioArithElem)
     $E{expr}: exp
     $L{expr}: log
     $M{expr}: Unary minus
     $Piterator_name{expr}: prod
     $Q{string1}{string2}: sequence
     $Siterator_name{expr}: sum
     $Ziterator_name[{expr1}...{exprN}]: merged sum
     //{expr1}{expr2}...{exprN}//: list of expressions
     number: constant
     #id: literal
     &id: random
   */
  virtual patString getExpressionString() const = PURE_VIRTUAL ;

  virtual void collectExpressionIds(set<patULong>* s) const = PURE_VIRTUAL ;

  virtual patBoolean isNamedExpression() const ;

  virtual void checkMonteCarlo(patBoolean insideMonteCarlo, patError*& err) ;

protected:
  patBoolean performCheck ;
  patULong parent ;
  patBoolean top;

  patULong lastComputedLap ;
  patReal lastValue ;

  patULong theId ;
  patString additionalCode ;

  patBoolean isInsideMonteCarlo ;

  const patVariables* __x ;
  //  const vector< vector<patReal> >* __draws ;
  patReal** __draws ;
  patReal** __unifdraws ;
  bioSample* theSample ;

  bioIteratorSpan theCurrentSpan ;
  bioIteratorSpan theThreadSpan ;

  //  vector<patULong> theGradientLiteralIds ;

  // These are the expressions such that we need to propagate
  // information such as threadId or the data sample
  vector<bioExpression*> relatedExpressions ;

  bioExpressionRepository* theRepository ;

  bioFunctionAndDerivatives result ;
  bioFunctionAndDerivatives findiff ;

  bioReporting* theReport ;

  

};


#endif
