//-*-c++-*------------------------------------------------------------
//
// File name : patGeneralizedUtility.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Sun Mar  2 16:38:57 2003
//
//--------------------------------------------------------------------

#ifndef patGeneralizedUtility_h
#define patGeneralizedUtility_h

#include "patError.h"
#include "patVariables.h"
#include "patUtility.h"

class patArithNode ;

/**
   @doc This class is in charge of computing the value of the
     generalized utility functions. A generalized utility function is
     composed of 3 terms $V=V_1 + V_2 + V_3$ where $V_1$ is
     deterministic and linear-in-parameters, $V_2$ is deterministic
     and nonlinear, and $V_3$ is a nonlinear function of normally
     distributed parameters.
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Sun Mar  2 16:38:57 2003)
 */

class patGeneralizedUtility : public patUtility {

public:
  /**
     Sole constructor. Does nothing.
   */
  patGeneralizedUtility() ;

  /**
     Destructor
  */
  ~patGeneralizedUtility() ;

    /** Evaluate the function. 
      @param altId identifier of the alternative
      @param beta value of the beta parameters. 
      @param x    value of the characteristics
      @param err ref. of the pointer to the error object.
      @return value of the utility function
   */

  patReal computeFunction(unsigned long observationId,
			  unsigned long drawNumber,
			  unsigned long altId,
			  patVariables* beta,
			  vector<patObservationData::patAttributes>* x,
			  patError*& err)  ;
  

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
					     patError*& err)  ;
  
  
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
   */
  patBoolean isLinear() const { return patFALSE ; }

  /**
   */
  void generateCppCode(ostream& cppFile,
		       unsigned long altId,
		       patError*& err)  ;


  /**
   */
  void generateCppDerivativeCode(ostream& cppFile,
				 unsigned long altId,
				 unsigned long betaId, 
				 patError*& err)  ;

private:
  unsigned long userId ;
  patArithNode* expression  ;
  patReal result ;
  unsigned long i ;

  patBoolean analyticalDerivatives ;
  vector<patArithNode*> utilExpressions ;
  vector< vector<patArithNode*> > derivExpressions ;
  vector< vector<patBoolean> > derivFirst ;
};

#endif 

