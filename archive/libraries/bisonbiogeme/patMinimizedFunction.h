//-*-c++-*------------------------------------------------------------
//
// File name : patMinimizedFunction.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Aug  9 01:30:01 2000
//
//--------------------------------------------------------------------


#ifndef patMinimizedFunction_h
#define patMinimizedFunction_h

#include "trFunction.h"
#include "patSecondDerivatives.h"
#include "trParameters.h"

class patLikelihood ;
class patSecondDerivatives ;

/** @doc This object implements the abstract trFunction for the maximum likelihood
    estimation of the GEV model.
    @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Aug  9 01:30:01 2000)
*/

class patMinimizedFunction : public trFunction {

public :

  /**
     Sole constructor.  
     @param aLikelihood pointer to the object in charge of
     computing the log-likelihood
  */
  patMinimizedFunction(patLikelihood* aLikelihood, trParameters theTrParameters) ;
  /**
     Dtor
   */
  virtual ~patMinimizedFunction() ;

  /**
     @return  value of the function to minimize. 
     @param x the variariables stored in x correspond to thr non fixed 
     parameters of the model ($\beta$, $\mu$, $\mu_m$,$\alpha$, 
     scale parameters, etc.)  
     @param err ref. of the pointer to the error object.
   */
  virtual patReal computeFunction(trVector* x,
			      patBoolean* success,
			      patError*& err) ;

  /**
     @param x vector of $\mathbb{R}^n$ where the function is evaluated
     @param grad pointer to the vector where the gradient will be stored
     @param err ref. of the pointer to the error object.
     @return value of the function
   */
  virtual patReal computeFunctionAndDerivatives(trVector* x,
						trVector* grad,
						trHessian* hessian,
						patBoolean* success,
						patError*& err); 
  /**
     This method is supposed to provide a cheap approximation of the
     hessian. Here, it is the BHHH approximation
  */
  virtual trHessian* computeCheapHessian(trHessian* theCheapHessian,
					 patError*& err) ;

  /**
   */
  virtual patBoolean isCheapHessianAvailable() ;

  /**
     Computes the product of the hessian and a vector.
     @return  value of the hessian of the function
     @param x the variariables stored in x correspond to the non fixed 
     parameters of the model ($\beta$, $\mu$, $\mu_m$,$\alpha$, 
     scale parameters, etc.)  
     @param err ref. of the pointer to the error object.
   */
  virtual trVector* computeHessianTimesVector(trVector* x,
					 const trVector* v,
					 trVector* r,
					 patBoolean* success,
					 patError*& err)  ;

  /**
     @param err ref. of the pointer to the error object.
   */
  virtual patBoolean isGradientAvailable() 
    const ;

  /**
     @param err ref. of the pointer to the error object.
   */
  virtual patBoolean isHessianAvailable() 
    const ;

  /**
     @param err ref. of the pointer to the error object.
   */
  virtual patBoolean 
  isHessianTimesVectorAvailable() const ;

  /**
     @return Number of variable for the minimization problem
     @param err ref. of the pointer to the error object.
   */
  virtual unsigned long getDimension() 
    const ;

  /**
     @return This function is designed to generate optimized code for
     the computation of the function and gradients
   */
  void generateCppCode(ostream& str, patError*& err) ;

protected:
  
  trVector* gatherGradient(trVector* grad,
			   patError*& err) ;
  /**
     Allocates memory for gradient computation
   */
  void allocateMemory(patError*& err) ;

protected:
  patLikelihood* likelihood ;

private :

  patBoolean firstTime ;

  vector<patBoolean> betaVariable ;
  patVariables betaDerivatives ;
  patBoolean muVariable;
  patReal muDerivative ;
  vector<patBoolean> paramVariable ;
  patVariables paramDerivative ;
  vector<patBoolean> scaleVariable ;
  patVariables scaleDerivative ;
  vector<patVariables> bhhh ;
  patBoolean useBhhh ;
  trVector previousTime ;
  patReal previousValue ;
  patSecondDerivatives* secondDeriv ;

  trParameters theTrParameters ;
};

#endif


