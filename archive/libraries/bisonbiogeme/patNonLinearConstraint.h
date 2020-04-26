//-*-c++-*------------------------------------------------------------
//
// File name : patNonLinearConstraint.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Jan 20 09:10:52 2003
//
//--------------------------------------------------------------------

#ifndef patNonLinearConstraint_h
#define patNonLinearConstraint_h

#include <list>
#include "patArithNode.h"
#include "trFunction.h"

class patNonLinearConstraint :  public trFunction {

public:
  /**
   */
  friend ostream& operator<<(ostream &str, const patNonLinearConstraint& x) ;

  /**
     Ctor
   */
  patNonLinearConstraint(patArithNode* node) ;

  /**
     Dtor
   */
  virtual ~patNonLinearConstraint() ;

  /**
     @param x vector of $\mathbb{R}^n$ where the function is evaluated
     @param err ref. of the pointer to the error object.
     @return value of the function 
   */
  virtual patReal computeFunction(trVector* x,
				  patBoolean* success,
				  patError*& err) ;


  /**
     @param x vector of $\mathbb{R}^n$ where the function is evaluated
     @param grad pointer to the vector where the gradient will be stored
     @param err ref. of the pointer to the error object.
     @return vale of the function
   */
  virtual patReal computeFunctionAndDerivatives(trVector* x,
						trVector* grad,
						trHessian* hessian,
						patBoolean* success,
						patError*& err) ;



  /**
     @doc This function does not do anything, and is implemented to
     comply with the interface.  2
     param x vector of $\mathbb{R}^n$
     where the hessian is evaluated 
     @param v vector of $\mathbb{R}^n$
     that will be pre-multiplied by the hessian 
     @param err ref. of the  pointer to the error object.  
     @return trVector()
  */

  virtual trVector* computeHessianTimesVector(trVector* x,
					     const trVector* v,
					     trVector* r,
					     patBoolean* success,
					     patError*& err)  ;
  
  /**
     This method is supposed to provide a cheap approximation of the
     hessian. By default, it is set to the identity matrix.
  */
  virtual trHessian* computeCheapHessian(trHessian* theCheapHessian,
					 patError*& err) ;

  /**
   */
  virtual patBoolean isCheapHessianAvailable();

  virtual trHessian* getHessian() ;
  /**
     @param err ref. of the pointer to the error object.
     @return patTRUE
   */
  virtual patBoolean isGradientAvailable() const  ;

  /**
     @param err ref. of the pointer to the error object.
     @return patFALSE
   */
  virtual patBoolean isHessianTimesVectorAvailable() const ;

  /**
     @param err ref. of the pointer to the error object.
     @return patFALSE
   */
  virtual patBoolean isHessianAvailable() const ;

  /**
     @return number of variables of the function
     @param err ref. of the pointer to the error object.
   */
   virtual unsigned long getDimension() const ;

  /**
     @return This function is designed to generate optimized code for
     the computation of the function and gradients
   */
  virtual void generateCppCode(ostream& str, patError*& err) ;


  /**
     @param dim dimension of the problem
   */
  void setDimension(unsigned long dim) ;
  
  /**
     Identify literal NAME as the variable for derivation, and specify its index.
   */
  void setVariable(const patString& s, unsigned long i)  ;

  /**
     @return a pointer to a vector containing literals used in the expression. It is a recursive function. If needed, the values of the literals are stored.
     @param listOfLiterals pointer to the data structures where literals are stored. Must be non NULL
     @param valuesOfLiterals pointer to the data structure where the current values of the literals are stored (NULL is not necessary)
     @param withRandom patTRUE if random coefficients are considered as literals, patFALSE otherwise.
   */
  vector<patString>* getLiterals(vector<patString>* listOfLiterals,
				 vector<patReal>* valuesOfLiterals,
				 patBoolean withRandom,
				 patError*& err) const ;

  
private:

  patArithNode* expression ;
  unsigned long dimension ;

};

/**
 */
typedef list<patNonLinearConstraint> patListNonLinearConstraints ;


#endif
