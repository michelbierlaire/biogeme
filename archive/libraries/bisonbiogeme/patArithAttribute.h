//-*-c++-*------------------------------------------------------------
//
// File name : patArithAttribute.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Mar  5 13:29:12 2003
//
//--------------------------------------------------------------------

#ifndef patArithAttribute_h
#define patArithAttribute_h

#include "patArithNode.h"

/**
   @doc This class implements a node of the tree representing a variable
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Mar  5 13:29:12 2003) 
*/

class patArithAttribute : public patArithNode {

public:
  
  /**
   */
  patArithAttribute(patArithNode* par) ;

  /**
   */
  virtual ~patArithAttribute() ;

  /**
   */
  virtual patOperatorType getOperatorType() const ;

  /**
     @return name of the variable
   */
  virtual patString getOperatorName() const;

  /**
     @return value of the expression
   */
   patReal getValue(patError*& err) const  ;

  /**
     @return value of the derivative w.r.t variable x[index]
     @param index index of the variable involved in the derivative
     @param err ref. of the pointer to the error object.
   */
  virtual patReal getDerivative(unsigned long index, patError*& err) const ;
    
  /**
     @return printed expression
   */

  virtual patString getExpression(patError*& err) const ;


  /**
     Expand an expression with expressions already defined
   */
  virtual void expand(patError*& err) ;

  /**
     Replace a subchain by another in each literal
   */
  virtual void replaceInLiterals(patString subChain, patString with) ;


  /**
     Identify literal NAME as an attribute, and specify its index.
   */
  void setAttribute(const patString& s, unsigned long i)  ;

  /**
   */
  void setId(unsigned long id) ;

  /**
   */
  void setName(const patString& n) ;

  /**
     Get a deep copy of the expression
   */
  virtual patArithAttribute* getDeepCopy(patError*& err) ;

  /**
   */
  patString getCppCode(patError*& err) ;
  /**
   */
  patString getCppDerivativeCode(unsigned long index, patError*& err)  ;

private:

  patString name ;
  unsigned long index ;
  patArithNode* calculatedExpression ; 
};
#endif

