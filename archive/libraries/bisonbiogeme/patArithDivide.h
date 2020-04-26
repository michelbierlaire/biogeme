//-*-c++-*------------------------------------------------------------
//
// File name : patArithDivide.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Nov 23 15:50:12 2000
//
//--------------------------------------------------------------------

#ifndef patArithDivide_h
#define patArithDivide_h

#include "patArithNode.h"

/**
   @doc This class implements a node of the tree representing a division operation 
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Thu Nov 23 15:50:12 2000)
*/

class patArithDivide : public patArithNode {

public:
  
  /**
   */
  patArithDivide(patArithNode* par,
		 patArithNode* left, 
		 patArithNode* right) ;

  /**
   */
  ~patArithDivide() ;

  /**
   */
  virtual patOperatorType getOperatorType() const ;

  /**
     @return '/'
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
  virtual patArithDivide* getDeepCopy(patError*& err) ;
  /**
   */
   patString getCppCode(patError*& err) ;
  /**
   */
  patString getCppDerivativeCode(unsigned long index, patError*& err)  ;

};
#endif

