//-*-c++-*------------------------------------------------------------
//
// File name : trFunction.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri Jan 21 14:04:32 2000
//
//--------------------------------------------------------------------

#ifndef trFunction_h
#define trFunction_h

#include "patConst.h"
#include "patType.h"
#include "trVector.h"
#include "trHessian.h"

/**
  @doc  Defines a virtual interface for a non linear function
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Fri Jan 21 14:04:32 2000)
 */
class trFunction {
public:

  virtual ~trFunction() ;
  /**
     @param x vector of $\mathbb{R}^n$ where the function is evaluated
     @param err ref. of the pointer to the error object.
     @return value of the function.  
   */
  virtual patReal computeFunction(trVector* x,
				  patBoolean* success,
				  patError*& err) 
    = PURE_VIRTUAL ;
  /**
     Computes the gradient by forward finite difference.
     @param x vector of $\mathbb{R}^n$ where the gradient is evaluated
     @param err ref. of the pointer to the error object.
     @return gradient
   */
  virtual trVector* computeFinDiffGradient(trVector* x,
					   trVector* g,
					   patBoolean* success,
					   patError*& err) ;

  /**
     @param x vector of $\mathbb{R}^n$ where the function is evaluated
     @param grad pointer to the vector where the gradient will be stored
     @param err ref. of the pointer to the error object.
     @return vale of the function
   */
  virtual patReal computeFunctionAndDerivatives(trVector* x,
						trVector* g,
						trHessian* h,
						patBoolean* success,
						patError*& err) = PURE_VIRTUAL ;
  /**
rF     @param x vector of $\mathbb{R}^n$ where the hessian is evaluated
     @param v vector of $\mathbb{R}^n$ that will be pre-multiplied by
     the hessian @param err ref. of the pointer to the error object.
     @return hessian times v. It is assumed that the hessian has been
     computed already using "computeFunctionAndDerivatives"
  */
  virtual trVector* computeHessianTimesVector(trVector* x,
					      const trVector* v,
					      trVector* r,
					      patBoolean* success,
					      patError*& err)  = PURE_VIRTUAL ;
  
  /**
     This method is supposed to provide a cheap approximation of the
     hessian. It should be called after a call to
     "computeFunctionAndDerivatives"
  */
  virtual trHessian* computeCheapHessian(trHessian* theCheapHessian,
					 patError*& err) = PURE_VIRTUAL ;

  /**
   */
  virtual patBoolean isCheapHessianAvailable() = PURE_VIRTUAL ;

  /**
     Compute the hessian by finite difference
     @return  pointer to the hessian of the function (pointer to the hessian  object passed to the function) 
     @param x the variables stored in x correspond to the non fixed 
     parameters of the model ($\beta$, $\mu$, $\mu_m$,$\alpha$, 
     scale parameters, etc.)  
     @param hessian object where the hessian values will be stored
     @param err ref. of the pointer to the error object.
   */
  virtual trHessian* computeFinDiffHessian(trVector* x,
					   trHessian* h,
					   patBoolean* success,
					   patError*& err) ; 
 

  /**
     @param err ref. of the pointer to the error object.
   */
  virtual patBoolean isGradientAvailable() const 
    = PURE_VIRTUAL ;
  /**
     @param err ref. of the pointer to the error object.
   */
  virtual patBoolean isHessianTimesVectorAvailable() const
    = PURE_VIRTUAL ;
  /**
     @param err ref. of the pointer to the error object.
   */
  virtual patBoolean isHessianAvailable() const
    = PURE_VIRTUAL ;

  /**
     @return number of variables of the function
     @param err ref. of the pointer to the error object.
   */
  virtual unsigned long getDimension() const = PURE_VIRTUAL;

  /**
     @return This function is designed to generate optimized code for
     the computation of the function and gradients
   */
  virtual void generateCppCode(ostream& str, patError*& err) = PURE_VIRTUAL ;

  void setStopFileName(patString f) ; // patParameters::the()->getgevStopFileName()
protected:
  patString stopFileName ;
};

#endif
