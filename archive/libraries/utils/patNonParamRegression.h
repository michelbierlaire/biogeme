//-*-c++-*------------------------------------------------------------
//
// File name : patNonParamRegression.h
// Author :    \URL[Michel Bierlaire]{http://transp-or2.epfl.ch}
// Date :      Fri Dec 21 07:33:41 2007
//
//--------------------------------------------------------------------

#ifndef patNonParamRegression_h
#define patNonParamRegression_h

#include "patType.h"
#include "patError.h"
#include "patVariables.h"

/**
   Performs nonparametric regression and returns table with results
   Regression of ly against lx with bandwidth lh.
   Source: Mogens Fosgerau's Ox code.
*/

class patNonParamRegression {

 public:

  patNonParamRegression(patVariables* ly,
			patVariables* lx,
			patVariables* p,
			patReal bw,
			int nonParamPlotRes, // patParameters::the()->getgevNonParamPlotRes()
			patError*& err) ;

  void compute(patError*& err) ;
  patVariables* getNewX() ;
  patVariables* getMainPlot() ;
  patVariables* getLowerPlot() ;
  patVariables* getUpperPlot() ;
  patVariables* getProbPlot() ;
  patVariables* getDensityPlot() ;
  patString getPstricksPlot() ;
  patString getGnuplotPlot() ;
  void saveOnFile(const patString& theFile) ;
 private:
  patReal bandwidth ;
  const patVariables* y ;
  const patVariables* x ;
  const patVariables* prob ;
  patVariables newX ;
  patVariables mainPlot ;
  patVariables lowerPlot ;
  patVariables upperPlot ;
  patVariables smoothProba ;
  patVariables density ;
  patULong points ;
  patReal pi ;
};

#endif
