//-*-c++-*------------------------------------------------------------
//
// File name : patProbaOrdinalLogit.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Thu Jun 15 16:36:50 2006
//
//--------------------------------------------------------------------

#ifndef patProbaOrdinalLogit_h
#define patProbaOrdinalLogit_h

#include "patError.h"
#include "patObservationData.h"
class patGEV ;
class patUtility ;
struct patBetaLikeParameter ;

#include "patProbaModel.h"

/**
   @doc Defines the probability model for the ordinal logit model, based on the specification of the utility functions and a sample
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Thu Jun 15 16:36:50 2006 )
 */

class patProbaOrdinalLogit : public patProbaModel {

public:
  /**
     Sole constructor
   */
  patProbaOrdinalLogit(patUtility* aUtility) ;

  /**
     Destructor
  */
  ~patProbaOrdinalLogit() ;


  /**
     Evaluates the logarithm of the probability given by the model that the
     individual chooses alternative 'index', knowing the utilities, for a given draw number. If
     requested, the derivatives are evaluated as well. Note that the derivatives
     are cumulated to previous values. In order to have the value of the
     derivatives, the corresponding storage area must be initialized to zero
     before calling this method.
     
@param logOfProba if patTRUE, the log of the probability is returned
@param indivId identifier of the individual
    @param drawNumber number of the requested draw
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
    
    

  */
    
  patReal evalProbaPerDraw(patBoolean logOfProba, 
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
			   patError*& err) ;
  



  /**
   */
  patString getModelName(patError*& err) ;

  /**
     @return largest argument used in all calls to the exp() function
   */
  patReal getMaxExpArgument () ;

  /**
     @return smallest argument used in all calls to the exp() function
   */
  patReal getMinExpArgument () ;

  /**
   */
  unsigned long getUnderflow() ;
  /**
   */
  unsigned long getOverflow() ;

  /**
   */
  virtual patString getInfo() ;

  /**
   */
  void generateCppCodePerDraw(ostream& str,
			      patBoolean logOfProba,
			      patBoolean derivatives, 
			      patBoolean secondDerivatives, 
			      patError*& err) ;
private :

  
  patReal minExpArgument ;
  patReal maxExpArgument ;

  unsigned long overflow ;
  unsigned long underflow ;

  map<unsigned long, pair<patBetaLikeParameter*,patBetaLikeParameter*> >
  thresholds ;

  vector<patObservationData::patAttributes>* x ;
  unsigned long index ;
  patReal weight ;
  unsigned long nBeta ;
  unsigned long indivId ;
  unsigned short chosen ;
  unsigned short unchosen ;
  patVariables V ;
  patReal diff ;
  unsigned long i,k ;
  patReal proba ;
  patVariables utilDerivativesZero ;
  patVariables utilDerivativesOne ;

};

#endif
