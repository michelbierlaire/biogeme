//-*-c++-*------------------------------------------------------------
//
// File name : trParameters.h
// Date :      Fri Oct 22 07:37:06 2010
//
//--------------------------------------------------------------------

#ifndef trParameters_h
#define trParameters_h

#include "patType.h"

class trParameters {
  
 public:
  trParameters() ;
  patReal eta1; //  patParameters::the()->getBTREta1()
  patReal eta2; //  patParameters::the()->getBTREta2() ;
  patReal gamma2; // patParameters::the()->getBTRGamma2() ;
  patReal beta; // patParameters::the()->getBTRIncreaseTRRadius() ;
  patULong maxIter; //  patParameters::the()->getBTRMaxIter() ;
  patBoolean initQuasiNewtonWithTrueHessian; // patParameters::the()->getBTRInitQuasiNewtonWithTrueHessian()
  patBoolean initQuasiNewtonWithBHHH; // patParameters::the()->getBTRInitQuasiNewtonWithBHHH()
  int significantDigits; // patParameters::the()->getBTRSignificantDigits();
  patBoolean usePreconditioner; //patParameters::the()->getBTRUsePreconditioner();
  patReal maxTrustRegionRadius; // patParameters::the()->getBTRMaxTRRadius()
  patReal typicalF; //patParameters::the()->getBTRTypf()
  patReal tolerance; // patParameters::the()->getBTRTolerance()
  patReal toleranceSchnabelEskow;  // patParameters::getTolSchnabelEskow()
  patBoolean exactHessian; // patParameters::the()->getBTRExactHessian()
  patBoolean cheapHessian; // patParameters::the()->getBTRCheapHessian()
  patReal initRadius; // patParameters::the()->getBTRInitRadius()
  patReal minRadius; // patParameters::the()->getBTRMinTRRadius()
  patReal armijoBeta1 ;
  patReal armijoBeta2 ;
  patString stopFileName ;
  patULong startDraws ; // patParameters::the()->getBTRStartDraws() ;
  patULong increaseDraws ; // patParameters::the()->getBTRIncreaseDraws() ;
  patULong maxGcpIter ;
  patReal fractionGradientRequired ; // patParameters::the()->getTSFractionGradientRequired()
  patReal expTheta ;  // patParameters::the()->getTSExpTheta()
  patULong infeasibleCgIter; // patParameters::the()->getBTRUnfeasibleCGIterations()
  int quasiNewtonUpdate ; // patParameters::the()->getBTRQuasiNewtonUpdate()
  patReal kappaUbs ; // patParameters::the()->getBTRKappaUbs()
  patReal kappaLbs ; // patParameters::the()->getBTRKappaLbs()
  patReal kappaFrd ; //  patParameters::the()->getBTRKappaFrd()
  patReal kappaEpp ; // patParameters::the()->getBTRKappaEpp() 

};

#endif 
