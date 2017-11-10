//-*-c++-*------------------------------------------------------------
//
// File name : patArithNormalRandom.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Tue Mar  4 09:21:57 2003
//
//--------------------------------------------------------------------

#ifndef patArithNormalRandom_h
#define patArithNormalRandom_h

#include "patArithRandom.h"
#include "patArithNode.h"
#include "patUtilFunction.h"

/**
   @doc This class implements a node of the tree representing a random variable
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Tue Mar  4 09:21:57 2003) 
*/

class patArithNormalRandom : public patArithRandom {

public:
  
  /**
   */
  patArithNormalRandom(patArithNode* par) ;

  /**
   */
  virtual ~patArithNormalRandom() ;

  /**
   */
  virtual patOperatorType getOperatorType() const ;

  /**
     @return name of the variable
   */
  virtual patString getOperatorName() const;

  /**
     @return value of the expression
   */
   patReal getValue(patError*& err) const  ;

  /**
     @return value of the derivative w.r.t variable x[index]
     @param index index of the variable involved in the derivative
     @param err ref. of the pointer to the error object.
   */
  virtual patReal getDerivative(unsigned long index, patError*& err) const ;
    
  /**
     @return printed expression
   */

  virtual patString getExpression(patError*& err) const ;

  /**
   */
  void addCovariance(patString c) ;

  /**
   */
  void setLinearExpression(const patUtilFunction& util) ;

  /**
   */
  patDistribType getDistribution()  ;

  /**
   */
  void addTermToLinearExpression(patUtilTerm term) ;

  /**
     Replace a subchain by another in each literal
   */
  virtual void replaceInLiterals(patString subChain, patString with) ;

  /**
     Get a deep copy of the expression
   */
  virtual patArithNormalRandom* getDeepCopy(patError*& err) ;

  /**
   */
  patUtilFunction* getLinearExpression() ;

  /**
     @return GNUPLOT syntax
   */
  virtual patString getGnuplot(patError*& err) const ;
  
  /**
   */
  patString getCppCode(patError*& err) ;
  /**
   */
  patString getCppDerivativeCode(unsigned long index, patError*& err)  ;

  /**
   */
  patBoolean isDerivativeStructurallyZero(unsigned long index, patError*& err)  ;
private:
  patUtilFunction linearExpression ;
  vector<patString> covariance ;
    
};


#endif
