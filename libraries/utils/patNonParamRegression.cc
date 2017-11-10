//-*-c++-*------------------------------------------------------------
//
// File name : patNonParamRegression.cc
// Author :    Michel Bierlaire
// Date :      Fri Dec 21 07:51:22 2007
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patNonParamRegression.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"
#include "patNormalPdf.h"
#include "patConst.h"

patNonParamRegression::patNonParamRegression(patVariables* ly,
					     patVariables* lx,
					     patVariables* p,
					     patReal bw,
					     int nonParamPlotRes,
					     patError*& err) :

  bandwidth(bw),
  y(ly),
  x(lx),
  prob(p),
  newX(nonParamPlotRes),
  mainPlot(nonParamPlotRes),
  lowerPlot(nonParamPlotRes),
  upperPlot(nonParamPlotRes),
  smoothProba(nonParamPlotRes),
  density(nonParamPlotRes),
  points(nonParamPlotRes) {

  
  pi = 4.0 * atan(1.0) ;
  if (y == NULL || x == NULL || prob == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return ;
  }
  if (x->size() != y->size() || x->size() != prob->size()) {
    stringstream str ;
    str << "Incompatible sizes: x(" << x->size() << "), y(" << y->size() << "), prob(" << prob->size() << ")" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }

} ;

void patNonParamRegression::compute(patError*& err) {

  patReal step = 1.0 / patReal(points - 1) ;
  patReal xi = 0.0 ;

  for (patULong i = 0 ; i < points ; ++i) {
    patReal xFiltered = 0.0 ;
    patReal yFiltered = 0.0 ;
    patReal pFiltered = 0.0 ;
    for (patULong j = 0 ; j < x->size() ; ++j) {
      patReal xj = (*x)[j] ;
      patReal norm = patNormalPdf()((xi - xj)/bandwidth) ;
      xFiltered += norm ;
      yFiltered += (*y)[j] * norm ;
      pFiltered += (*prob)[j] * norm ;
    }
    newX[i] = xi ;
    mainPlot[i] = yFiltered / xFiltered ;
    smoothProba[i] = pFiltered / xFiltered ;
    density[i] = xFiltered / (bandwidth * patReal(x->size())) ;
    patReal sigma = sqrt(smoothProba[i] * (1.0 - smoothProba[i]) / (2.0 * sqrt(pi)) / xFiltered) ;
    lowerPlot[i] = mainPlot[i] - 1.96 * sigma ;
    upperPlot[i] = mainPlot[i] + 1.96 * sigma ;
    xi += step ;
  }
}

patVariables* patNonParamRegression::getNewX() {
  return &newX ;
}

patVariables* patNonParamRegression::getMainPlot() {
  return &mainPlot ;
}


patVariables* patNonParamRegression::getLowerPlot() {
  return &lowerPlot ;
} 

patVariables* patNonParamRegression::getUpperPlot() {
  return &upperPlot ;
}

patVariables* patNonParamRegression::getProbPlot() {
  return &smoothProba ;
}

patVariables* patNonParamRegression::getDensityPlot() {
  return &density ;
}

patString patNonParamRegression::getPstricksPlot() {
  return patString("Not yet implemented") ;
}

patString patNonParamRegression::getGnuplotPlot() {
  return patString("Not yet implemented") ;
}

void patNonParamRegression::saveOnFile(const patString& theFile) {
  ofstream f(theFile.c_str()) ;
  f << "x\tLower\tRegression\tUpper\tDensity" << endl ;
  patReal step = 1.0 / patReal(points - 1) ;
  patReal xi = 0.0 ;

  for (patULong i = 0 ; i < points ; ++i) {
    f << xi << '\t'
      << lowerPlot[i] << '\t'
      << mainPlot[i] << '\t'
      << upperPlot[i] << '\t'
      << density[i] << endl ;
    xi += step ;
  }
  f.close();
}
