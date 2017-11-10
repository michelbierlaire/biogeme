//-*-c++-*------------------------------------------------------------
//
// File name : patProbaPanelModel.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Tue Mar 30 14:58:04 2004
//
//--------------------------------------------------------------------

#ifndef patProbaPanelModel_h
#define patProbaPanelModel_h

#include "patError.h"

class patGEV ;
class patUtility ;

#include "patVariables.h"
#include "patIndividualData.h" 

/**
   @doc Defines an interface for the probability model
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Tue Mar 30 14:58:04 2004)
*/

class patProbaPanelModel {
  
public:

  /**
   */
  patProbaPanelModel(patUtility* aUtility = NULL) ;

  /**
   */
  virtual ~patProbaPanelModel() ;

  /**
     Evaluates the logarithm of the probability given by the model that the
     individiual chooses alternative 'index', knowing the utilities. If
     requested, the derivatives are evaluated as well. Note that the
     derivatives are cumulated to previous values. In order to have the value
     of the derivatives, the corresponding storage area must be initialized to
     zero before calling this method. 
     
     Denoting by $i$ the draw index, and $R$ the number of draws, the simulated probability is
     \[
     P = \frac{1}{R} \sum_{i=1}^R  P_i
     \]
     and
     \[
     \log P = \log \frac{1}{R} \sum_{i=1}^R  P_i.
     \]
     The derivative with regard to any parameter $\theta$ is 
     \[
     \frac{\partial \log P}{\partial \theta} = \frac{R}{\sum_{i=1}^R  P_i} \frac{1}{R} \sum_{i=1}^R  \frac{\partial P_i}{\partial \theta}= \frac{1}{\sum_{i=1}^R  P_i} \sum_{i=1}^R  \frac{\partial P_i}{\partial \theta}
     \]
     with
     \[
     \frac{\partial P_i}{\partial \theta} = P_i \frac{\partial \log  P_i}{\partial \theta}
     \]
     @param indivId identifier of the individual
     @param index index of the chosen alternative
     @param utilities value of the utility of each alternative
     @param beta value of the beta parameters
     @param x value of the characteristics
     @param parameters value of the structural parameters of the GEV function
     @param scale scale parameter
     @param noDerivative patTRUE: no derivative is computed. 
     patFALSE: derivatives may be computed.
     @param compBetaDerivatives If entry k is patTRUE, the derivative with 
     respect to $\beta_k$ is computed
     @param compParamDerivatives If entry k is patTRUE, the derivative with 
     respect to parameter $k$ is computed
     @param compMuDerivative If patTRUE, the derivative with respect to $\mu$ 
     is computed.
     @param betaDerivatives pointer to the vector where the derivatives with respoect to $\beta$ will be accumulated. The result will be meaningful only if noDerivatives is patFALSE and for entries corresponding to patTRUE in compBetaDerivatives. All other cells are undefined. 
     @param Derivatives pointer to the vector where the derivatives with respect to the GEV parameters will be accumulated. The result will be meaningful only if noDerivatives is patFALSE and for entries corresponding to patTRUE in compParamDerivatives. All other cells are undefined. 
     @param muDerivative pointer to the patReal where the derivative with respect to mu will be accumulated. The result will be meaningful only if noDerivatives is patFALSE and  compMuDerivative is patTRUE.
     @param available describes which alternative are available
     @param mu value of the homogeneity parameter of the GEV function
     @param err ref. of the pointer to the error object. 
     @return pointer to the variable where the result will be stored. 
     Same as outputvariable
     
     The derivatives computed in this routine have been successfully checked
     through a comparison with finite difference computation (Wed Jan 24
     22:06:22 2001)
  */    


  virtual patReal evalProbaLog
  (patIndividualData* observation,
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
   patBoolean* success,
   patVariables* grad,
   patBoolean computeBHHH,
   vector<patVariables>* bhhh,
   patError*& err) ;

  /**  
   */
  
  virtual patReal evalProbaPerAggObs( patObservationData* observation,
				      patAggregateObservationData* aggObservation,
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
				      patBoolean* success,
				      patError*& err) ;
  /**  
   */
  
  virtual patReal evalProbaPerObs( patObservationData* individual,
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
				   patBoolean* success,
				   patError*& err) = PURE_VIRTUAL ;
  /**
   */
  virtual  patReal evalProbaPerDraw( patIndividualData* individual,
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
				     patBoolean snpTerms,
				     patReal factorForDerivOfSnpTerms,
				     patBoolean* success,
				     patError*& err) ;
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
  virtual void generateCppCode(ostream& cppFile,
			       patBoolean derivatives, 
			       patError*& err) ;

  /**
   */
  virtual void generateCppCodePerDraw(ostream& cppFile,
			       patBoolean derivatives, 
			       patError*& err) ;

  /**
   */
  virtual void generateCppCodePerObs(ostream& cppFile,
			       patBoolean derivatives, 
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
  patReal add ;
  patBoolean useAggregateObservation ;
  patVariables f_betaDerivatives ;
  patVariables f_paramDerivatives ;
};

#endif
