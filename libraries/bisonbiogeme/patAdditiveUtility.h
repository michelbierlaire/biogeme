//-*-c++-*------------------------------------------------------------
//
// File name : patAdditiveUtility.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Mar  3 17:54:02 2003
//
//--------------------------------------------------------------------

#ifndef patAdditiveUtility_h
#define patAdditiveUtility_h

#include "patError.h"
#include "patUtility.h"
#include "patObservationData.h"

/**
   @doc This class is in charge of computing the value of the sum of utility functions for
   various value of characteristics and parameters.
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Mon Mar  3 17:54:02 2003)
 */

class patAdditiveUtility : public patUtility {

public:
  /**
     Sole constructor. Does nothing.
  */
  patAdditiveUtility() {} ;

  /**
     Destructor
  */
  virtual ~patAdditiveUtility() {} ;

  /**
     Add a new term in the sum of utility functions
  */
  void addUtility(patUtility* aUtil) ;
  
  /** Evaluate the function. 
      @param altId identifier of the alternative
      @param beta value of the beta parameters. 
      @param x    value of the characteristics
      @param err ref. of the pointer to the error object.
      @return value of the utility function
  */
  
  virtual patReal computeFunction(unsigned long observationId,
				  unsigned long drawNumber,
				  unsigned long altId,
				  patVariables* beta,
				  vector<patObservationData::patAttributes>* x,
				  patError*& err) ;
  

  /** Evaluate the derivative with regard to all parameters
      @param altId identifier of the alternative
      @param beta value of the beta parameters. 
      @param x    value of the characteristics
      @param err ref. of the pointer to the error object.
      @return vector of derivatives. Same length as beta.
  */

  virtual patVariables* computeBetaDerivative(unsigned long observationId,
					     unsigned long drawNumber,
					     unsigned long altId,
					     patVariables* beta,
					     vector<patObservationData::patAttributes>* x,
					      patVariables* betaDerivatives,
					     patError*& err) ;
  
  /** Evaluate the derivative with regard to all characteristics. This is used
      in the computation of the elasticities.

      @param altId identifier of the alternative
      @param beta value of the beta parameters. 
      @param x    value of the characteristics
      @param err ref. of the pointer to the error object.
      @return vector of derivatives. Same length as x.  
  */

  virtual patVariables computeCharDerivative(unsigned long observationId,
					     unsigned long drawNumber,
					     unsigned long altId,
					     patVariables* beta,
					     vector<patObservationData::patAttributes>* x,
					     patError*& err)  ;


  /**
     Provides the name of the utility function 
  */
  virtual patString getName() const ;

  /**
     A priori, an additive utility is nonlinear. 
   */
  patBoolean isLinear() const { return patFALSE ; }
  /**
   */
  void generateCppCode(ostream& cppFile,
		       unsigned long altId,
		       patError*& err) ;

  /**
   */
  void generateCppDerivativeCode(ostream& cppFile,
				 unsigned long altId,
				 unsigned long betaId,
				 patError*& err)  ;
  
private:
  vector<patUtility*> listOfUtilities ;

};

#endif 

