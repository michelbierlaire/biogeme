//-*-c++-*------------------------------------------------------------
//
// File name : bioConstraintWrapper.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Sun Oct 25 09:42:18 2009
//
//--------------------------------------------------------------------

#ifndef bioConstraintWrapper_h
#define bioConstraintWrapper_h

#include "trFunction.h"

/**! This object implements the abstract trFunction for the
     constraints on the parameters, as specificed with the Python code.
*/

class bioConstraintWrapper: public trFunction {

public :

  /**
     Sole constructor.  
  */
  bioConstraintWrapper() ;
  /**
     Dtor
   */
  ~bioConstraintWrapper() ;

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
  patBoolean isGradientAvailable() 
    const ;

  /**
     @param err ref. of the pointer to the error object.
   */
  patBoolean isHessianAvailable() 
    const ;

  /**
     @param err ref. of the pointer to the error object.
   */
  patBoolean 
  isHessianTimesVectorAvailable() const ;

  /**
     @return Number of variable for the minimization problem
     @param err ref. of the pointer to the error object.
   */
  unsigned long getDimension() const = PURE_VIRTUAL;

  /**
     @return Must comply with the trFunction interface, but useless here.
   */
  void generateCppCode(ostream& str, patError*& err) ;


};
#endif
