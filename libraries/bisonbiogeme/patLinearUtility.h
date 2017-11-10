//-*-c++-*------------------------------------------------------------
//
// File name : patLinearUtility.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Jan 10 14:05:45 2001
//
//--------------------------------------------------------------------

#ifndef patLinearUtility_h
#define patLinearUtility_h

#include "patError.h"
#include "patVariables.h"
#include "patUtility.h"
#include "patAlternative.h"

/**
   @doc This class is in charge of computing the value of the utility functions for
   various value of characteristics and parameters.

   The current implementation is based on linear-in-parameters function, 
   that is $V_i=\sum_j \beta_j x_{ij}$.
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Jan 10 14:05:50 2001)
 */

class patLinearUtility : public patUtility {

public:
  /**
     Sole constructor. Does nothing.
   */
  patLinearUtility(patBoolean m) ;

  /**
     Destructor
  */
  ~patLinearUtility() ;

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
					      patVariables* betaDrawDerivatives,
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
  virtual patString getName() const {
    return patString("Linear-in-parameters") ;
  } ;

  /**
   */
  patBoolean isLinear() const { return patTRUE ; }

  /**
   */
  void printFileForDenis(unsigned long nbeta) ;

  /**
   */
  virtual void generateCppCode(ostream& cppFile,
		       unsigned long altId,
		       patError*& err)  ;

  /**
   */
  virtual void generateCppDerivativeCode(ostream& cppFile,
					 unsigned long altId,
					 unsigned long betaId,
					 patError*& err)  ;

protected:
  void initPointers(unsigned long altId) ;

  patReal computeLinearFunction(unsigned long observationId,
				unsigned long drawNumber,
				unsigned long altId,
				patVariables* beta,
				vector<patObservationData::patAttributes>* x,
				patError*& err)  ;

private:
  
  patReal outputVariable ;
  patBoolean completeCalculation ;
  patUtilFunction* theUtility ;
  patReal theAttr ;
  patReal determ ;
  patReal drawValue ;
  patUtilFunction::iterator start;

  patVariables oldDetermUtilities ;
  vector<unsigned long> previousObservation ;
  vector<unsigned long> previousDraw ;
  vector<unsigned long> previousDerivObservation ;
  vector<patUtilFunction*> utilities ;
  vector<patUtilFunction::iterator> startRandom ;



  unsigned long userId ;
  patUtilFunction::iterator i ;
  patUtilFunction::iterator term ;
};

#endif 

