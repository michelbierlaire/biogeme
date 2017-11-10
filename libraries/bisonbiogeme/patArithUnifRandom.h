//-*-c++-*------------------------------------------------------------
//
// File name : patArithUnifRandom.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Tue Dec 23 13:56:35 2003
//
//--------------------------------------------------------------------

#ifndef patArithUnifRandom_h
#define patArithUnifRandom_h

#include "patArithRandom.h"
#include "patUtilFunction.h"

/**
   @doc This class implements a node of the tree representing a
   distributed random variable, defined by two parameters $\beta$ and
   $\lambda$. The random variable is uniformly distributed between
   $\beta-\lambda$ and $\beta+\lambda$.

   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Tue Dec 23 13:56:35 2003) 
*/

class patArithUnifRandom : public patArithRandom {

public:
  
  /**
   */
  patArithUnifRandom(patArithNode* par) ;

  /**
   */
  virtual ~patArithUnifRandom() ;

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
  void setLinearExpression(const patUtilFunction& util) ;

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
  virtual patArithUnifRandom* getDeepCopy(patError*& err) ;

  /**
   */
  patDistribType getDistribution()  ;

  /**
   */
  patUtilFunction* getLinearExpression() ;

  /**
   */
  void addCovariance(patString c) ;
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
private:
  patUtilFunction linearExpression ;
};


#endif
