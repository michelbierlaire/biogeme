//-*-c++-*------------------------------------------------------------
//
// File name : patArithInt.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Nov 23 15:40:48 2000
//
//--------------------------------------------------------------------

#ifndef patArithInt_h
#define patArithInt_h

#include "patArithNode.h"

/**
   @doc This class implements a node of the tree representing an rounding  operation 
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Thu Nov 23 15:40:48 2000) 
*/

class patArithInt : public patArithNode {

public:
  
  /**
   */
  patArithInt(patArithNode* par,
	      patArithNode* left) ;

  /**
   */
  ~patArithInt() ;

  /**
   */
  virtual patOperatorType getOperatorType() const ;

  /**
     @return 'int'
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
  virtual patArithInt* getDeepCopy(patError*& err) ;
   /**
    */
   patString getCppCode(patError*& err) ;
  /**
   */
  patString getCppDerivativeCode(unsigned long index, patError*& err)  ;
};
#endif

