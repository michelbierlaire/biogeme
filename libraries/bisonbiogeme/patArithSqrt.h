//-*-c++-*------------------------------------------------------------
//
// File name : patArithSqrt.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Nov 23 15:15:39 2000
//
//--------------------------------------------------------------------

#ifndef patArithSqrt_h
#define patArithSqrt_h

#include "patArithNode.h"

/**
   @doc This class implements a node of the tree representing a square root operation 
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Thu Nov 23 15:15:39 2000) 
   @see patArithExpression, patArithNode 
*/

class patArithSqrt : public patArithNode {

public:
  
  /**
   */
  patArithSqrt(patArithNode* par,
	       patArithNode* left) ;

  /**
   */
  ~patArithSqrt() ;

  /**
   */
  virtual patOperatorType getOperatorType() const ;

  /**
     @return 'sqrt'
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
  patReal getDerivative(unsigned long index, patError*& err) const  ;

  /**
     @return printed expression
   */

  virtual patString getExpression(patError*& err) const ;

  /**
     Get a deep copy of the expression
   */
  virtual patArithSqrt* getDeepCopy(patError*& err) ;


  /**
   */
   patString getCppCode(patError*& err) ;
  /**
   */
  patString getCppDerivativeCode(unsigned long index, patError*& err)  ;
};
#endif

