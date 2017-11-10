//-*-c++-*------------------------------------------------------------
//
// File name : patArithLess.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri Nov 24 09:17:33 2000
//
//--------------------------------------------------------------------

#ifndef patArithLess_h
#define patArithLess_h

#include "patArithNode.h"

/**
   @doc This class implements a node of the tree representing a comparison (<) operation 
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Fri Nov 24 09:17:33 2000)
*/

class patArithLess : public patArithNode {

public:
  
  /**
   */
  patArithLess(patArithNode* par,
		 patArithNode* left, 
		 patArithNode* right) ;

  /**
   */
  ~patArithLess() ;

  /**
   */
  virtual patOperatorType getOperatorType() const ;

  /**
     @return '<'
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
  virtual patArithLess* getDeepCopy(patError*& err) ;
  /**
   */
   patString getCppCode(patError*& err) ;
  /**
   */
  patString getCppDerivativeCode(unsigned long index, patError*& err)  ;
};
#endif

