//-*-c++-*------------------------------------------------------------
//
// File name : trParameters.cc
// Date :      Fri Oct 29 16:31:56 2010
//
//--------------------------------------------------------------------

#include "trParameters.h"

trParameters::trParameters() :
  eta1(0.01), 
  eta2(0.9), 
  gamma2(0.5), 
  beta(2.0), 
  maxIter(1000), 
  initQuasiNewtonWithTrueHessian(0), 
  initQuasiNewtonWithBHHH(1), 
  significantDigits(7), 
  usePreconditioner(0), 
  maxTrustRegionRadius(1.0e10), 
  typicalF(1.0), 
  tolerance(6.05545e-06), 
  toleranceSchnabelEskow(0.00492157),  
  exactHessian(0), 
  cheapHessian(1), 
  initRadius(1.0), 
  minRadius(1.0e-7), 
  armijoBeta1(0.1),
  armijoBeta2(0.9),
  stopFileName("STOP"),
  startDraws(10), 
  increaseDraws(2), 
  maxGcpIter(10000),
  fractionGradientRequired(0.1), 
  expTheta(0.5),  
  infeasibleCgIter(0), 
  quasiNewtonUpdate(1), 
  kappaUbs(0.1), 
  kappaLbs(0.9), 
  kappaFrd(0.5), 
  kappaEpp(0.25) 
{

}

