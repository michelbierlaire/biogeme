//-*-c++-*------------------------------------------------------------
//
// File name : patUtility.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Tue Mar 13 09:32:33 2001
//
//--------------------------------------------------------------------

#ifndef patUtility_h
#define patUtility_h

#include "patError.h"
#include "patIndividualData.h"

/**
   @doc This class is in charge of computing the value of the utility functions for
   various value of characteristics and parameters.
   It is a pure virtual class defining an interface
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Tue Mar 13 09:32:33 2001)
 */

class patUtility {

public:
  /**
     Sole constructor. Does nothing.
   */
  patUtility() {} ;

  /**
     Destructor
  */
  virtual ~patUtility() {} ;

    /** Evaluate the function. 
      @param altId identifier of the alternative
      @param beta value of the beta parameters. 
      @param x    value of the characteristics
      @param err ref. of the pointer to the error object.
      @return value of the utility function
   */

  virtual patReal computeFunction(unsigned long individualId,
				  unsigned long drawNumber,
				  unsigned long altId,
				  patVariables* beta,
				  vector<patObservationData::patAttributes>* x,
				  patError*& err) 
    = PURE_VIRTUAL ;
  

  /** Evaluate the derivative with regard to all parameters
      @param altId identifier of the alternative
      @param beta value of the beta parameters. 
      @param x    value of the characteristics
      @param err ref. of the pointer to the error object.
      @return vector of derivatives. Same length as beta.
  */

  virtual patVariables* computeBetaDerivative(unsigned long individualId,
					      unsigned long drawNumber,
					      unsigned long altId,
					      patVariables* beta,
					      vector<patObservationData::patAttributes>* x,
					      patVariables* derivatives,
					      patError*& err) 
    = PURE_VIRTUAL ;

  /** Evaluate the derivative with regard to all characteristics. This is used
      in the computation of the elasticities.

      @param altId identifier of the alternative
      @param beta value of the beta parameters. 
      @param x    value of the characteristics
      @param err ref. of the pointer to the error object.
      @return vector of derivatives. Same length as x.  
  */

  virtual patVariables computeCharDerivative(unsigned long individualId,
					     unsigned long drawNumber,
					     unsigned long altId,
					     patVariables* beta,
					     vector<patObservationData::patAttributes>* x,
					     patError*& err) 
    = PURE_VIRTUAL ;


  /**
     Provides the name of the utility function 
   */
  virtual patString getName() const = PURE_VIRTUAL ;

  /**
   */
  virtual patBoolean isLinear() const = PURE_VIRTUAL ;

  /**
   */
  void genericCppCode(ostream& cppFile, 
		      patBoolean derivatives,
		      patError*& err) ;

  /**
   */
  virtual void generateCppCode(ostream& cppFile,
			       unsigned long altId,
			       patError*& err) = PURE_VIRTUAL ;
  /**
   */
  virtual void generateCppDerivativeCode(ostream& cppFile,
					 unsigned long altId,
					 unsigned long betaId,
					 patError*& err) = PURE_VIRTUAL ;

};

#endif 

