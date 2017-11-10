//-*-c++-*------------------------------------------------------------
//
// File name : patArithVariable.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Nov 22 22:26:58 2000
//
//--------------------------------------------------------------------

#ifndef patArithVariable_h
#define patArithVariable_h

#include "patArithNode.h"

/**
   @doc This class implements a node of the tree representing a variable
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Nov 22 22:26:58 2000) 
*/

class patArithVariable : public patArithNode {

public:
  
  /**
   */
  patArithVariable(patArithNode* par) ;

  /**
   */
  virtual ~patArithVariable() ;

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
   */
  patBoolean isDerivativeStructurallyZero(unsigned long index, patError*& err)  ;
    
  /**
     @return printed expression
   */

  virtual patString getExpression(patError*& err) const ;

  /**
     Expand an expression with expressions already defined
   */
  virtual void expand(patError*& err) ;


  /**
   */
  void setName(patString n) ;

  /**
     Identify literal NAME as the variable for derivation, and specify its index.
   */
   void setVariable(const patString& s, unsigned long i) ;

  /**
     Replace a subchain by another in each literal
   */
  virtual void replaceInLiterals(patString subChain, patString with) ;

  /**
     Get a deep copy of the expression
   */
  virtual patArithVariable* getDeepCopy(patError*& err) ;

  /**
   */
  patString getCppCode(patError*& err) ;
  /**
   */
  patString getCppDerivativeCode(unsigned long index, patError*& err)  ;
private:

  patString name ;
  unsigned long index ;
  // Index in the vector of unknown parameters. Needed for the cppcode only.
  unsigned long xIndex ;
  patReal value ;
  patArithNode* calculatedExpression ;
};
#endif

