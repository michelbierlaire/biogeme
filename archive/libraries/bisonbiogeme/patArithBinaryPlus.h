//-*-c++-*------------------------------------------------------------
//
// File name : patArithBinaryPlus.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Nov 22 17:28:58 2000
//
//--------------------------------------------------------------------

#ifndef patArithBinaryPlus_h
#define patArithBinaryPlus_h

#include "patArithNode.h"

/**
   @doc This class implements a node of the tree representing an addition ($+$) operation 
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Nov 22 17:28:58 2000) 
   @see patArithExpression, patArithNode 
*/

class patArithBinaryPlus : public patArithNode {

public:
  
  /**
   */
  patArithBinaryPlus(patArithNode* par,
		     patArithNode* left, 
		     patArithNode* right) ;

  /**
   */
  ~patArithBinaryPlus() ;

  /**
   */
  virtual patOperatorType getOperatorType() const ;

  /**
     @return '+'
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
  virtual patArithBinaryPlus* getDeepCopy(patError*& err) ;
  /**
   */
   patString getCppCode(patError*& err) ;
  /**
   */
  patString getCppDerivativeCode(unsigned long index, patError*& err)  ;
};
#endif

