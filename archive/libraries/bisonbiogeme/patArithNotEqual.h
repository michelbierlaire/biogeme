//-*-c++-*------------------------------------------------------------
//
// File name : patArithNotEqual.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri Nov 24 09:09:09 2000
//
//--------------------------------------------------------------------

#ifndef patArithNotEqual_h
#define patArithNotEqual_h

#include "patArithNode.h"

/**
   @doc This class implements a node of the tree representing a comparison (!=) operation 
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Fri Nov 24 09:09:09 2000)
*/

class patArithNotEqual : public patArithNode {

public:
  
  /**
   */
  patArithNotEqual(patArithNode* par,
		 patArithNode* left, 
		 patArithNode* right) ;

  /**
   */
  ~patArithNotEqual() ;

  /**
   */
  virtual patOperatorType getOperatorType() const ;

  /**
     @return '!='
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
  virtual patArithNotEqual* getDeepCopy(patError*& err) ;

  /**
   */
   patString getCppCode(patError*& err) ;
  /**
   */
  patString getCppDerivativeCode(unsigned long index, patError*& err)  ;
};
#endif

