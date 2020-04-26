//-*-c++-*------------------------------------------------------------
//
// File name : patLikelihood.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Tue Aug  8 23:04:38 2000
//
//--------------------------------------------------------------------

#ifndef patLikelihood_h
#define patLikelihood_h

#include "patError.h"
#include "patIterator.h"
#include "patVariables.h"
#include "patBetaLikeParameter.h"
//#include "patDiscreteParameter.h"
//#include "patGenerateCombinations.h"

class patProbaModel ;
class patProbaPanelModel ;
class patSample ;
class patObservationData ;
class patAggregateObservationData ;
class patIndividualData ;
class patSecondDerivatives ;
class trHessian ;
class patDiscreteParameterProba ;

/**
   @doc This class implements a likelihood function, combining a model and a sample
   of observations.  
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Tue Aug  8 23:04:38 2000) 
 */

class patLikelihood {

public:
  /**
   */
  patLikelihood() ;
  /**
   */
  ~patLikelihood() ;
  /**
     @param aModel pointer to the probability model
     @param aPanelModel pointer to the probability model for panel data
     @param aSample pointer to the sample
     @doc Exactly one of aModel and aPanelModel must be non zero
   */
  patLikelihood(patProbaModel* aModel, 
		patProbaPanelModel* aPanelModel,
		patSample* aSample,
		patError*& err) ;
  /**
   */
  void setModel(patProbaModel* aModel) ;
  /**
   */
  void setModel(patProbaPanelModel* aModel) ;
  /**
   */
  patProbaModel* getModel() ;
  /**
   */
  patProbaPanelModel* getPanelModel() ;

  /**
   */
  void setSample(patSample* aSample) ;
  /**
   */
  patSample* getSample() ;

  /**
     Evaluate the likelihood function for given values of beta parameters
     (involved in the utilities), model parameters (like mu_m) and scale parameters
     @param betaParameters parameters from the utility function
     @param mu degree of homogeneity of the GEV model
     @param modelParameters parameters appearing in the GEV model
     @param scaleParameters group specific scale parameters
  */

  patReal evaluate(patVariables* betaParameters,
		   const patReal* mu,
		   const patVariables* modelParameters,
		   const patVariables* scaleParameters,
		   patBoolean* success,
		   patVariables* grad,
		   patError*& err)  ;
  
  /**
     Evaluate the loglikelihood function for given values of beta parameters
     (involved in the utilities) and model parameters (like scale parameters)
     @param betaParameters parameters from the utility function
     @param mu degree of homogeneity of the GEV model
     @param modelParameters parameters appearing in the GEV model
     @param scaleParameters group specific scale parameters
  */

  patReal evaluateLog(patVariables* betaParameters,
		      const patReal* mu,
		      const patVariables* modelParameters,
		      const patVariables* scaleParameters,
		      patBoolean noDerivative ,
		      const vector<patBoolean>& compBetaDerivatives,
		      const vector<patBoolean>& compParamDerivatives,
		      patBoolean compMuDerivative,
		      const vector<patBoolean>& compScaleDerivatives,
		      patVariables* betaDerivatives,
		      patVariables* paramDerivatives,
		      patReal* muDerivative,
		      patSecondDerivatives* secondDeriv,
		      patVariables* scaleDerivatives,
		      patBoolean* success,
		      patVariables* grad,
		      patBoolean computeBHHH,
		      vector<patVariables>* bhhh,
		      trHessian* trueHessian,
		      patError*& err)  ;

  
  /**
   */
  patString getModelName(patError*& err) ;

  /**
     Computes the log-likelihood of the sample for a trivial model giving equal probability to each available alternative.
   */
  patReal getTrivialModelLikelihood() ;


protected:
  patReal computeObservationProbability(patObservationData* observation,
					patAggregateObservationData* aggObservation,
					patVariables* betaParameters,
					const patReal* mu,
					const patVariables* modelParameters,
					const patVariables* scaleParameters,
					patBoolean noDerivative ,
					const vector<patBoolean>& compBetaDerivatives,
					const vector<patBoolean>& compParamDerivatives,
					patBoolean compMuDerivative,
					const vector<patBoolean>& compScaleDerivatives,
					patVariables* betaDerivatives,
					patVariables* paramDerivatives,
					patReal* muDerivative,
					patReal* scaleDerivPointer,
					patSecondDerivatives* secondDeriv,
					patBoolean* success,
					patVariables* grad,
					patBoolean computeBHHH,
					vector<patVariables>* bhhh,
					patError*& err) ;

  patReal computePanelIndividualProbability(patIndividualData* observation,
					    patVariables* betaParameters,
					   const patReal* mu,
					   const patVariables* modelParameters,
					   const patVariables* scaleParameters,
					   patBoolean noDerivative ,
					   const vector<patBoolean>& compBetaDerivatives,
					   const vector<patBoolean>& compParamDerivatives,
					   patBoolean compMuDerivative,
					   const vector<patBoolean>& compScaleDerivatives,
					   patVariables* betaDerivatives,
					   patVariables* paramDerivatives,
					   patReal* muDerivative,
					   patReal* scaleDerivPointer,
					   patBoolean* success,
					   patVariables* grad,
					   patBoolean computeBHHH,
					   vector<patVariables>* bhhh,
					   patError*& err) ;

public:
  /**
     @return These functions are designed to generate optimized code for
     the computation of the function and gradients
   */
  void generateCppCode(ostream& str,  patError*& err) ;
  void generateCppCodeOneObservation(ostream& str,
				     patBoolean derivatives, 
				     patBoolean secondDerivatives, 
				     patError*& err) ;
//   void generateScaleCppCode(ostream& cppFile,
// 			    patError*& err) ;
  
private :
  patProbaModel* model ;
  patProbaPanelModel* panelModel ;
  patSample*     sample ;
  patReal logLike ;
  unsigned long nAlt ;
  unsigned long observationCounter ;
  unsigned long aggObservationCounter ;
  unsigned long individualCounter ;
  unsigned long step ;
  patObservationData* observation ;
  patAggregateObservationData* aggObservation ;
  patBoolean compScale ;
  unsigned long groupIndex;
  patReal scale ;
  patBetaLikeParameter theScale ;
  unsigned long scaleIndex ;
  patBoolean areObservationsAggregate ;

//   patVariables singleBetaDerivatives ;
//   patVariables singleParamDerivatives ;
//   patReal singleMuDerivative ;
//   patReal singleScaleDeriv ;

  patReal tmp ;

  patDiscreteParameterProba* theDiscreteParamModel ;


  patIterator<patAggregateObservationData*>* theAggObsIterator ;
  patIterator<patObservationData*>* theObsIterator ;
  patIterator<patIndividualData*>* theIndIterator ;
};

#endif
