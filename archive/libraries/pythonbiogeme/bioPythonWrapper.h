//-*-c++-*------------------------------------------------------------
//
// File name : bioPythonWrapper.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Fri Jul 17 14:00:35 2009
//
//--------------------------------------------------------------------


#ifndef bioPythonWrapper_h
#define bioPythonWrapper_h





#include "trFunction.h"
#include "trParameters.h"

class bioSample ;
class bioSplit ;

/**! This object implements the abstract trFunction for the maximum
     likelihood estimation of the model specificed with a Python
     program. 
*/

class bioPythonWrapper : public trFunction {

public :

  /**
     Sole constructor.  
  */
  bioPythonWrapper() ;
  /**
     Dtor
   */
  ~bioPythonWrapper() ;

  /**
     @return  value of the function to minimize. 
     @param x the variariables stored in x correspond to thr non fixed 
     parameters of the model ($\beta$, $\mu$, $\mu_m$,$\alpha$, 
     scale parameters, etc.)  
     @param err ref. of the pointer to the error object.
   */
  patReal computeFunction(trVector* x,
			  patBoolean* success,
			  patError*& err) ;

  /**
     @param x vector of $\mathbb{R}^n$ where the function is evaluated
     @param grad pointer to the vector where the gradient will be stored
     @param err ref. of the pointer to the error object.
     @return value of the function
   */
  patReal computeFunctionAndDerivatives(trVector* x,
					trVector* grad,
					trHessian* hessian,
					patBoolean* success,
					patError*& err); 
  /**
     This method is supposed to provide a cheap approximation of the
     hessian. Here, it is the BHHH approximation
  */
  trHessian* computeCheapHessian(trHessian* hessian,
				 patError*& err) ;

  /**
   */
  patBoolean isCheapHessianAvailable() ;

  /**
     Computes the product of the hessian and a vector.
     @return  value of the hessian of the function
     @param x the variariables stored in x correspond to the non fixed 
     parameters of the model ($\beta$, $\mu$, $\mu_m$,$\alpha$, 
     scale parameters, etc.)  
     @param err ref. of the pointer to the error object.
   */
  trVector* computeHessianTimesVector(trVector* x,
				      const trVector* v,
				      trVector* r,
				      patBoolean* success,
				      patError*& err)  ;

  /**
     @param err ref. of the pointer to the error object.
   */
   virtual patBoolean isGradientAvailable()     const ;

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
     @return Must comply with the trFunction interface, but useless here.
   */
  void generateCppCode(ostream& str, patError*& err) ;

  /**
     @return Current value of the variables
     @param err ref. of the pointer to the error object.
   */
  trVector getCurrentVariables(patError*& err) const ;

  /**
     @return patTRUE if inherited, user-based function, patFALSE if base class.
   */
  patBoolean isUserBased() const ;


  /**
   */
  void setSample(bioSample* s, patError*& err) ;

 protected:
  bioSample* sample ;
  trParameters theTrParameters ;
};

#endif


