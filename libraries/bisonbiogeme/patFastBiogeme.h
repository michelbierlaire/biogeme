//-*-c++-*------------------------------------------------------------
//
// File name : patFastBiogeme.h
// Author :    \URL[Michel Bierlaire]{http://transp-or2.epfl.ch}
// Date :      Tue Aug 14 10:39:56 2007
//
//--------------------------------------------------------------------


#ifndef patFastBiogeme_h
#define patFastBiogeme_h

#include "trFunction.h"

class patSample ;

/** @doc This object implements the abstract trFunction for the maximum likelihood estimation of the GEV model. Its purpose is to create optimized  C++ code, and to be superseded by the user-based implementation. 
    @author \URL[Michel Bierlaire]{http://transp-or2.epfl.ch}, EPFL (Tue Aug 14 10:39:56 2007)
*/

class patFastBiogeme : public trFunction {

public :

  /**
     Sole constructor.  
  */
  patFastBiogeme() ;
  /**
     Dtor
   */
  virtual ~patFastBiogeme() ;

  /**
     @return  value of the function to minimize. 
     @param x the variariables stored in x correspond to thr non fixed 
     parameters of the model ($\beta$, $\mu$, $\mu_m$,$\alpha$, 
     scale parameters, etc.)  
     @param err ref. of the pointer to the error object.
   */
  virtual patReal getFunction(trVector* x,
			      patBoolean* success,
			      patError*& err) ;

  /**
     @return  value of the gradient of the function
     @param x the variariables stored in x correspond to the non fixed 
     parameters of the model ($\beta$, $\mu$, $\mu_m$,$\alpha$, 
     scale parameters, etc.)  
     @param err ref. of the pointer to the error object.
   */
  virtual trVector* getGradient(trVector* x,
				trVector* grad,
				patBoolean* success,
				patError*& err)  ;
  
  /**
     @param x vector of $\mathbb{R}^n$ where the function is evaluated
     @param grad pointer to the vector where the gradient will be stored
     @param err ref. of the pointer to the error object.
     @return value of the function
   */
  virtual patReal getFunctionAndGradient(trVector* x,
					 trVector* grad,
					 patBoolean* success,
					 patError*& err); 
  /**
     @return  pointer to the hessian of the function (pointer to the hessian  object passed to the function) 
     @param x the variariables stored in x correspond to the non fixed 
     parameters of the model ($\beta$, $\mu$, $\mu_m$,$\alpha$, 
     scale parameters, etc.)  
     @param hessian object where the hessian values will be stored
     @param err ref. of the pointer to the error object.
   */
  virtual trHessian* computeHessian(patVariables* x,
				    trHessian& hessian,
				    patBoolean* success,
				    patError*& err) ; 
 
  /**
     This method is supposed to provide a cheap approximation of the
     hessian. Here, it is the BHHH approximation
  */
  virtual trHessian* getCheapHessian(trVector* x,
				     trHessian& hessian,
				     patBoolean* success,
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
  virtual trVector getHessianTimesVector(trVector* x,
					 const trVector* v,
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
     @return Must comply with the trFunction interface, but useless here.
   */
  virtual void generateCppCode(ostream& str, patError*& err) ;

  /**
     @return Current value of the variables
     @param err ref. of the pointer to the error object.
   */
  trVector getCurrentVariables(patError*& err) const ;

  /**
     @return patTRUE if inherited, user-based function, patFALSE if base class.
   */
  virtual patBoolean isUserBased() const ;


  /**
   */
  virtual void setSample(patSample* s) ;

 protected:
  patSample* sample ;

};

#endif


