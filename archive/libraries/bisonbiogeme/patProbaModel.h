//-*-c++-*------------------------------------------------------------
//
// File name : patProbaModel.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Jun  4 14:52:32 2001
//
//--------------------------------------------------------------------

#ifndef patProbaModel_h
#define patProbaModel_h

#include "patError.h"

class patGEV ;
class patUtility ;
class patObservationData ;
class patAggregateObservationData ;
class patSecondDerivatives ;

#include "patVariables.h"

/**
   @doc Defines an interface for the probability model
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Mon Jun  4 14:52:32 2001)
*/

class patProbaModel {
  
public:

  /**
   */
  patProbaModel(patUtility* aUtility = NULL) ;

  /**
   */
  virtual ~patProbaModel() ;

  /**
     Evaluates the logarithm of the probability given by the model that the
     individiual chooses alternative 'index', knowing the utilities. If
     requested, the derivatives are evaluated as well. Note that the
     derivatives are cumulated to previous values. In order to have the value
     of the derivatives, the corresponding storage area must be initialized to
     zero before calling this method.  */
  virtual patReal evalProbaLog
  (patObservationData* observation,
   patAggregateObservationData* aggObservation,
   patVariables* beta,
   const patVariables* parameters,
   patReal scale,
   unsigned long scaleIndex,
   patBoolean noDerivative,
   const vector<patBoolean>& compBetaDerivatives,
   const vector<patBoolean>& compParamDerivatives,
   patBoolean compMuDerivative,
   patBoolean compScaleDerivative,
   patVariables* betaDerivatives,
   patVariables* paramDerivatives,
   patReal* muDerivative,
   patReal* scaleDerivative,
   const patReal* mu,
   patSecondDerivatives* secondDeriv,
   patBoolean* success,
   patVariables* grad,
   patBoolean computeBHHH,
   vector<patVariables>* bhhh,
   patError*& err) ;
  
  /**
     Evaluates the logarithm of the probability given by the model that the
     individiual chooses alternative 'index', knowing the utilities, for a given draw number. 
  */
    
  virtual patReal evalProbaPerDraw(patBoolean logOfProba, 
				   patObservationData* individual,
				   unsigned long drawNumber,
				   patVariables* beta,
				   const patVariables* parameters,
				   patReal scale,
				   patBoolean noDerivative ,
				   const vector<patBoolean>& compBetaDerivatives,
				   const vector<patBoolean>& compParamDerivatives,
				   patBoolean compMuDerivative,
				   patBoolean compScaleDerivative,
				   patVariables* betaDerivatives,
				   patVariables* paramDerivatives,
				   patReal* muDerivative,
				   patReal* scaleDerivative,
				   const patReal* mu,
				   patSecondDerivatives* secondDeriv,
				   patBoolean snpTerms,
				   patReal factorForDerivOfSnpTerms,
				   patBoolean* success,
				   patError*& err) = PURE_VIRTUAL;
  
  /**
   */
  virtual patString getModelName(patError*& err) = PURE_VIRTUAL ;

  /**
   */
  void setUtility(patUtility* aUtility) ;

  /**
   */
  patUtility* getUtility() ;

  /**
   */
  virtual patString getInfo() ;

  /**
   */
  void generateCppCode(ostream& str,
		       patBoolean derivatives, 
		       patBoolean secondDerivatives, 
		       patError*& err) ;

  /**
   */
  virtual void generateCppCodePerDraw(ostream& str,
				      patBoolean logOfProba,
				      patBoolean derivatives, 
				      patBoolean secondDerivatives, 
				      patError*& err) = PURE_VIRTUAL ;

  
protected:

  patUtility*    utility ;
  patVariables* betaDrawDerivatives ;
  patVariables* paramDrawDerivatives ;
  patReal* muDrawDerivative;
  patReal* scaleDrawDerivative ;

  patVariables* betaSingleDerivatives ;
  patVariables* paramSingleDerivatives ;
  patReal* muSingleDerivative;
  patReal* scaleSingleDerivative ;

  patVariables* betaAggregDerivatives ;
  patVariables* paramAggregDerivatives ;
  patReal* muAggregDerivative;
  patReal* scaleAggregDerivative ;


  vector<unsigned long> idOfSnpBetaParameters ;
  patReal normalizeSnpTerms ;
  // This vector contains the derivative of (1 / K) with respect to delta_k, 
  // where delta_k is the SNP parameter, and K the normalizing constant
  // that is K = 1 + \sum_k \delta_k^2.
  patVariables derivNormalizationConstant ;
  patReal qForSnp ;
  patReal snpCorrection ;
  patReal proba ;
  patReal tmp ;
};

#endif
