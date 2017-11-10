//-*-c++-*------------------------------------------------------------
//
// File name : patArithNode.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Nov 22 16:27:47 2000
//
//--------------------------------------------------------------------

#ifndef patArithNode_h
#define patArithNode_h

#include "patError.h"
#include <map>

/**
   @doc This  class implements a node of the tree representing an
   arithmetic expression. It contains purely virtual methods. 
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Nov 22 16:27:47
   2000) 
   @see patArithExpression 
*/

class patArithNode {

/**
 */
friend ostream& operator<<(ostream& str, const patArithNode& x) ;

public:


  typedef  enum {
    UNARY_OP,
    BINARY_OP,
    VARIABLE_OP,
    CONSTANT_OP,
    RANDOM_OP,
    UNIRANDOM_OP,
    ATTRIBUTE_OP,
    UNDEFINED_OP,
  } patOperatorType ;


  /**
   */
  patArithNode(patArithNode* par,
	       patArithNode* left, 
	       patArithNode* right) ;

  /**
   */
  virtual ~patArithNode() ;

  /**
   */
  virtual patOperatorType getOperatorType() const = PURE_VIRTUAL ;

  /**
     @return patTRUE if the node has no parent
   */
  virtual patBoolean isTop() const  ;

  /**
     @return pointer to the node representing the left side of the expression
   */
  virtual patArithNode* getLeftChild() const ;
  /**
     @return pointer to the node representing the right side of the expression
   */
  virtual patArithNode* getRightChild() const;

  /**
     @param par pointer to the parent. 
  */
   void setParent(patArithNode* par) ;

  /**
     @return pointer to the root of the current expression
   */
  virtual patArithNode* getRoot() const ;

  /**
     @return pointer to the parent node
   */
  virtual patArithNode* getParent() const  ;

  /**
     @return name of the operator
   */
  virtual patString getOperatorName() const = PURE_VIRTUAL ;

  /**
     @return value of the expression
     @param err ref. of the pointer to the error object.
   */
  virtual patReal getValue(patError*& err) const 
			   = PURE_VIRTUAL ;

  /**
     @return value of the derivative w.r.t variable x[index]
     @param index index of the variable involved in the derivative
     @param err ref. of the pointer to the error object.
   */
  virtual patReal getDerivative(unsigned long index, patError*& err) const 
			   = PURE_VIRTUAL ;


  /**
   */
  virtual patBoolean isDerivativeStructurallyZero(unsigned long index, patError*& err)  ;

  /**
     @return printed expression
   */

  virtual patString getExpression(patError*& err) const ;

  /**
     Expand an expression with expressions already defined
   */
  virtual void expand(patError*& err) ;

  /**
     @return GNUPLOT syntax
   */
  virtual patString getGnuplot(patError*& err) const ;

  /**
     Identify literal NAME as the variable for derivation, and specify its index.
   */
  virtual void setVariable(const patString& s, unsigned long i)  ;

  /**
     Identify literal NAME as an attribute, and specify its index.
   */
  virtual void setAttribute(const patString& s, unsigned long i)  ;

  /**
     Compute the id of the parameter in the derivative node. 
     For other nodes, the call is just transferred
   */
  virtual void computeParamId(patError*& err) ;

  /**
     @return a pointer to a vector containing literals used in the expression. It is a recursive function. If needed, the values of the literals are stored.
     @param listOfLiterals pointer to the data structures where literals are stored. Must be non NULL
     @param valuesOfLiterals pointer to the data structure where the current values of the literals are stored (NULL is not necessary)
     @param withRandom patTRUE if random coefficients are considered as literals, patFALSE otherwise.
   */
  virtual vector<patString>* getLiterals(vector<patString>* listOfLiterals,
					 vector<patReal>* valuesOfLiterals,
					 patBoolean withRandom,
					 patError*& err) const ;

  /**
     Replace a subchain by another in each literal
   */
  virtual void replaceInLiterals(patString subChain, patString with) ;

  /**
   */
  virtual patString getInfo() const ;

  /**
   */
  ostream& printLiterals(ostream& str,patError*& err) const ;

  /**
     Get a deep copy of the expression. The user is in charge of
     releasing the memory.
   */
  virtual patArithNode* getDeepCopy(patError*& err) = PURE_VIRTUAL ;
  
  /**
   */
  virtual patString getCppCode(patError*& err) = PURE_VIRTUAL ;

  /**
   */
  virtual patString getCppDerivativeCode(unsigned long index, patError*& err) = PURE_VIRTUAL  ;

protected:

  patArithNode* leftChild ;
  patArithNode* rightChild ;
  patArithNode* parent ;
  map<unsigned long,patBoolean> derivStructurallyZero ;
};


#endif

