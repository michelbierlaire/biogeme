//-*-c++-*------------------------------------------------------------
//
// File name : patArithUnaryMinus.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Nov 22 16:45:45 2000
//
//--------------------------------------------------------------------

#ifndef patArithUnaryMinus_h
#define patArithUnaryMinus_h

#include "patArithNode.h"

/**
   @doc This class implements a node of the tree representing a unary minus ($-$) operation 
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Nov 22 16:45:45 2000) 
   @see patArithExpression, patArithNode 
*/

class patArithUnaryMinus : public patArithNode {

public:
  
  /**
   */
  patArithUnaryMinus(patArithNode* par,
		     patArithNode* left) ;

  /**
   */
  ~patArithUnaryMinus() ;

  /**
   */
  virtual patOperatorType getOperatorType() const ;

  /**
     @return '-'
   */
  virtual patString getOperatorName() const;



  /**
     @return value of the expression
   */
  virtual patReal getValue(patError*& err) const  ;
    
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
  virtual patArithUnaryMinus* getDeepCopy(patError*& err) ;

  /**
   */
   patString getCppCode(patError*& err) ;
  /**
   */
  patString getCppDerivativeCode(unsigned long index, patError*& err)  ;

};
#endif

