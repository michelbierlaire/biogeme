//-*-c++-*------------------------------------------------------------
//
// File name : patArithRandom.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Tue Dec 23 14:16:05 2003
//
//--------------------------------------------------------------------

#ifndef patArithRandom_h
#define patArithRandom_h

#include "patArithNode.h"
#include "patUtilFunction.h"
#include "patDistribType.h"

/**
   @doc This class defines an interface for a random parameter
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Tue Dec 23 14:16:05 2003) 
*/

class patArithRandom : public patArithNode {

public:

  /**
   */
  patArithRandom(patArithNode* par) ;

  /**
   */
  virtual ~patArithRandom() {} ;

  /**
   */
  virtual patOperatorType getOperatorType() const = PURE_VIRTUAL ;

  /**
     @return name of the variable
   */
  virtual patString getOperatorName() const = PURE_VIRTUAL ;

  /**
     @return value of the expression
   */
  patReal getValue(patError*& err) const = PURE_VIRTUAL ;

  /**
     @return value of the derivative w.r.t variable x[index]
     @param index index of the variable involved in the derivative
     @param err ref. of the pointer to the error object.
   */
  virtual patReal getDerivative(unsigned long index, patError*& err) const = PURE_VIRTUAL ;
    
  /**
     @return printed expression
   */

  virtual patString getExpression(patError*& err) const = PURE_VIRTUAL ;

  /**
   */
  virtual void setLinearExpression(const patUtilFunction& util) = PURE_VIRTUAL ;
  /**
   */
  virtual patDistribType getDistribution() = PURE_VIRTUAL ;
  
  /**
   */
  virtual void addTermToLinearExpression(patUtilTerm term) = PURE_VIRTUAL ;

  /**
     Replace a subchain by another in each literal
   */
  virtual void replaceInLiterals(patString subChain, patString with) = PURE_VIRTUAL ;

  /**
     Get a deep copy of the expression
   */
  virtual patArithRandom* getDeepCopy(patError*& err) = PURE_VIRTUAL ;

  /**
   */
  virtual patUtilFunction* getLinearExpression() = PURE_VIRTUAL ;

  /**
   */
  virtual void addCovariance(patString c) = PURE_VIRTUAL ;

  /**
   */
  virtual void setLocationParameter(patString m) ;

  /**
   */
  virtual void setScaleParameter(patString s) ;

  /**
   */
  virtual patString getLocationParameter() ;

  /**
   */
  virtual patString getScaleParameter() ;

  patString getCompactName() const ;

  /**
     TRUE if the random variable captures individual specific effects
     in panel data analysis
   */
  patBoolean isPanel ;

  /**
     @return a pointer to a vector containing literals used in the expression. It is a recursive function. If needed, the values of the literals are stored.
     @param listOfLiterals pointer to the data structures where literals are stored. Must be non NULL
     @param valuesOfLiterals pointer to the data structure where the current values of the literals are stored (NULL is not necessary)
     @param withRandom patTRUE if random coefficients are considered as literals, patFALSE otherwise.
  */
  virtual vector<patString>* getLiterals(vector<patString>* listOfLiterals,
					 vector<patReal>* valuesOfLiterals,
					 patBoolean withRandom,
					 patError*& err) const ;


protected:
  patString locationParameter ;
  patString scaleParameter ;





};


#endif
