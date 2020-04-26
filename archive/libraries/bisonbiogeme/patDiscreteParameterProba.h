//-*-c++-*------------------------------------------------------------
//
// File name : patDiscreteParameterProba.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Mon May 22 12:48:50 2006
//
//--------------------------------------------------------------------

#ifndef patDiscreteParameterProba_h
#define patDiscreteParameterProba_h

#include <vector>

#include "patError.h"
#include "patDiscreteParameter.h"
#include "patGenerateCombinations.h"
#include "patVariables.h"

class patProbaModel ;
class patProbaPanelModel ;
class patObservationData ;
class patAggregateObservationData ;
class patIndividualData ;
class patSecondDerivatives ;

class patDiscreteParameterProba {

 public:
  patDiscreteParameterProba(patProbaModel* aModel,
			    patProbaPanelModel* aPanelModel) ;
  patReal evalProbaLog
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

  patReal evalPanelProba(patIndividualData* observation,
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

  void generateCppCode(ostream& str,
		       patBoolean derivatives, 
		       patError*& err) ;


private :
  patProbaModel*  model ;
  patProbaPanelModel* panelModel ;
  vector<vector<patDiscreteTerm>::iterator > beginIterators ;
  vector<vector<patDiscreteTerm>::iterator > endIterators ;
  vector<unsigned long> discreteParamId ;
  patIterator<patDiscreteParameter*>* discreteParameters ;
  patDiscreteParameter* theDiscreteParameter ;
  vector<vector<patDiscreteTerm>::iterator>* oneCombi ;
  patReal weightProbabilityOfTheCombination ;
  patGenerateCombinations<vector<patDiscreteTerm>::iterator,patDiscreteTerm >*  theCombination ;

  patVariables singleBetaDerivatives ;
  patVariables singleParamDerivatives ;
  patReal singleMuDerivative ;
  patReal singleScaleDeriv ;

};


#endif
