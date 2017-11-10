//-*-c++-*------------------------------------------------------------
//
// File name : patArithConstant.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Nov 22 22:16:54 2000
//
//--------------------------------------------------------------------

#ifndef patArithConstant_h
#define patArithConstant_h

#include "patArithNode.h"

/**
   @doc This class implements a node of the tree representing a constant 
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Nov 22 22:16:54 2000) 
   @see patArithNode 
*/

class patArithConstant : public patArithNode {

public:
  
  /**
   */
  patArithConstant(patArithNode* par) ;

  /**
   */
  virtual ~patArithConstant() ;

  /**
   */
  virtual patOperatorType getOperatorType() const ;

  /**
     @return value of the constant
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
   */
  void setValue(patReal v) ;

  /**
     Get a deep copy of the expression
   */
  virtual patArithConstant* getDeepCopy(patError*& err) ;

  /**
   */
  patString getCppCode(patError*& err) ;
  /**
   */
  patString getCppDerivativeCode(unsigned long index, patError*& err)  ;
private:

  patReal value ;
};
#endif

