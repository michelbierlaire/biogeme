//-*-c++-*------------------------------------------------------------
//
// File name : patArithMult.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Nov 23 15:47:43 2000
//
//--------------------------------------------------------------------

#ifndef patArithMult_h
#define patArithMult_h

#include "patArithNode.h"

/**
   @doc This class implements a node of the tree representing a multiplication  operation 
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Thu Nov 23 15:47:43 2000) 
*/

class patArithMult : public patArithNode {

public:
  
  /**
   */
  patArithMult(patArithNode* par,
	       patArithNode* left, 
	       patArithNode* right) ;

  /**
   */
  ~patArithMult() ;

  /**
   */
  virtual patOperatorType getOperatorType() const ;

  /**
     @return '*'
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
  virtual patArithMult* getDeepCopy(patError*& err) ;
  /**
   */
   patString getCppCode(patError*& err) ;
  /**
   */
  patString getCppDerivativeCode(unsigned long index, patError*& err)  ;

};
#endif

