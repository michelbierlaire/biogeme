//-*-c++-*------------------------------------------------------------
//
// File name : patArithDeriv.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Mon Feb 20 14:44:10 2006
//
//--------------------------------------------------------------------

#ifndef patArithDeriv_h
#define patArithDeriv_h

#include "patArithNode.h" 

/**
  @doc This class implements a node of the tree representing the
  derivative of another expression with respect to a given parameter.
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Mon Feb 20 14:44:10 2006) 
   @see patArithExpression, patArithNode 
*/

class patArithDeriv: public patArithNode {

public:
  /**
     @param par  parent in the expression tree
     @param mainExpression expression to be derived
     @param param derive with respect to this parameter
   */
  patArithDeriv(patArithNode* par,
		patArithNode* mainExpression,
		patString param) ;

  /**
   */
  ~patArithDeriv() ;

  /**
   */
  virtual patOperatorType getOperatorType() const ;

  /**
     @return '$DERIV'
   */
  virtual patString getOperatorName() const;

  /**
     Compute the id of the parameter in the derivative node. 
     For other nodes, the call is just transferred
   */
  virtual void computeParamId(patError*& err) ;



  /**
     @return value of the expression
   */
  virtual patReal getValue(patError*& err) const ;
  
  /**
     @return value of the derivative w.r.t variable x[index]
     @param index index of the variable involved in the derivative
     @param err ref. of the pointer to the error object.
  */
  patReal getDerivative(unsigned long index, patError*& err) const ;
  
  /**
     @return printed expression
  */
  
  virtual patString getExpression(patError*& err) const ;
  
  /**
     Get a deep copy of the expression
  */
  virtual patArithDeriv* getDeepCopy(patError*& err) ;
  
  /**
   */
  patString getCppCode(patError*& err) ;
  /**
   */
  patString getCppDerivativeCode(unsigned long index, patError*& err)  ;
private:
  patString parameter ;
  patBoolean first ;
  unsigned long index ;
};
#endif
