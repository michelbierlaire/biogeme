//-*-c++-*------------------------------------------------------------
//
// File name : patArithLog.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Nov 23 15:18:44 2000
//
//--------------------------------------------------------------------

#ifndef patArithLog_h
#define patArithLog_h

#include "patArithNode.h"

/**
   @doc This class implements a node of the tree representing a natural logarith operation 
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Thu Nov 23 15:18:44 2000) 
   @see patArithExpression, patArithNode 
*/

class patArithLog : public patArithNode {

public:
  
  /**
   */
  patArithLog(patArithNode* par,
	      patArithNode* left) ;

  /**
   */
  ~patArithLog() ;

  /**
   */
  virtual patOperatorType getOperatorType() const ;

  /**
     @return 'log'
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
  virtual patArithLog* getDeepCopy(patError*& err) ;
   /**
    */
   patString getCppCode(patError*& err) ;
  /**
   */
  patString getCppDerivativeCode(unsigned long index, patError*& err)  ;
};
#endif

