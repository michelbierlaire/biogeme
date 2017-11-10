//-*-c++-*------------------------------------------------------------
//
// File name : patArithPower.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Nov 23 15:54:37 2000
//
//--------------------------------------------------------------------

#ifndef patArithPower_h
#define patArithPower_h

#include "patArithNode.h"

/**
   @doc This class implements a node of the tree representing a power operation 
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Thu Nov 23 15:54:37 2000)
*/

class patArithPower : public patArithNode {

public:
  
  /**
   */
  patArithPower(patArithNode* par,
		 patArithNode* left, 
		 patArithNode* right) ;

  /**
   */
  ~patArithPower() ;

  /**
   */
  virtual patOperatorType getOperatorType() const ;

  /**
     @return '^'
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
     
     \[
     \frac{\partial }{\partial x_i} u^v = u^{v-1} \left \frac{\partial v}{\partial x_i} u \ln u + v \frac{\partial u}{\partial x_i}\right)
     \]

   */
  patReal getDerivative(unsigned long index, patError*& err) const ;

  /**
     @return GNUPLOT syntax
   */
  virtual patString getGnuplot(patError*& err) const ;


  /**
     @return printed expression
   */

  virtual patString getExpression(patError*& err) const ;

  /**
     Get a deep copy of the expression
   */
  virtual patArithPower* getDeepCopy(patError*& err) ;
   /**
    */
   patString getCppCode(patError*& err) ;
  /**
   */
  patString getCppDerivativeCode(unsigned long index, patError*& err)  ;
};
#endif

