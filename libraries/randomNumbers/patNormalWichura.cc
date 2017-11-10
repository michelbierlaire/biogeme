//-*-c++-*------------------------------------------------------------
//
// File name : patNormalWichura.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Wed May 12 14:54:28 2004
//
//--------------------------------------------------------------------

#include "patNormalWichura.h"
#include "patMath.h"
#include "patUniform.h" 
#include "patDisplay.h"
#include "patFileNames.h"
#include "patErrMiscError.h"

patNormalWichura::patNormalWichura(patBoolean dumpDrawsOnFile) : 
  patRandomNumberGenerator(dumpDrawsOnFile),
  uniformNumberGenerator(NULL),
  logFile(NULL),
  zero(0.e+00), 
  one(1.e+00),
  half(0.5e+00),
  split1(0.425e+00),
  split2(5.e+00),
  const1(0.180625e+00), 
  const2(1.6e+00),
  a0(3.3871328727963666080e+00),
  a1(1.3314166789178437745e+02),
  a2(1.9715909503065514427e+03),
  a3(1.3731693765509461125e+04),
  a4(4.5921953931549871457e+04),
  a5(6.7265770927008700853e+04),
  a6(3.3430575583588128105e+04),
  a7(2.5090809287301226727e+03),
  b1(4.2313330701600911252e+01),
  b2(6.8718700749205790830e+02),
  b3(5.3941960214247511077e+03),
  b4(2.1213794301586595867e+04),
  b5(3.9307895800092710610e+04),
  b6(2.8729085735721942674e+04),
  b7(5.2264952788528545610e+03),
  c0(1.42343711074968357734e+00),
  c1(4.63033784615654529590e+00),
  c2(5.76949722146069140550e+00),
  c3(3.64784832476320460504e+00),
  c4(1.27045825245236838258e+00),
  c5(2.41780725177450611770e-01),
  c6(2.27238449892691845833e-02),
  c7(7.74545014278341407640e-04),
  d1(2.05319162663775882187e+00),
  d2(1.67638483018380384940e+00),
  d3(6.89767334985100004550e-01),
  d4(1.48103976427480074590e-01),
  d5(1.51986665636164571966e-02),
  d6(5.47593808499534494600e-04),
  d7(1.05075007164441684324e-09),
  e0(6.65790464350110377720e+00),
  e1(5.46378491116411436990e+00),
  e2(1.78482653991729133580e+00),
  e3(2.96560571828504891230e-01),
  e4(2.65321895265761230930e-02),
  e5(1.24266094738807843860e-03),
  e6(2.71155556874348757815e-05),
  e7(2.01033439929228813265e-07),
  f1(5.99832206555887937690e-01),
  f2(1.36929880922735805310e-01),
  f3(1.48753612908506148525e-02),
  f4(7.86869131145613259100e-04),
  f5(1.84631831751005468180e-05),
  f6(1.42151175831644588870e-07),
  f7(2.04426310338993978564e-15) {
  patError* err(NULL) ;
  if (dumpDrawsOnFile) {
    logFile = 
      new ofstream(patFileNames::the()->getNormalDrawLogFile(err).c_str()) ;
  }
 }

patNormalWichura::~patNormalWichura() {
  if (logFile != NULL) {
    logFile->close() ;
    DELETE_PTR(logFile) ;
  }
}

void patNormalWichura::setUniform(patUniform* rng) {
  uniformNumberGenerator = rng ;
}

pair<patReal,patReal> patNormalWichura::getNextValue(patError*& err) {
  if (uniformNumberGenerator == NULL) {
    err = new patErrMiscError("No pseudo-random generator specified") ;
    WARNING(err->describe());
    return pair<patReal,patReal> () ;
  }
  pair<patReal,patReal> theDraws ;
  patReal p = uniformNumberGenerator->getUniform(err) ;
  theDraws.second = p ;
  patReal q = p - half;
  
  patReal r, result;		
  if (patAbs(q) <= split1) {
    r = const1 - q * q;

    result =  q
      * (((((((a7 * r + a6) * r + a5) * r + a4) * r + a3) * r + a2) * r + a1) * r + a0)
      / (((((((b7 * r + b6) * r + b5) * r + b4) * r + b3) * r + b2) * r + b1) * r + one);
    if (logFile != NULL) {
      *logFile << result << endl ;
    }
    theDraws.first = result ;
    return theDraws ;
  } 
  else {
    if (q < zero)
      r = p;
    else
      r = one - p;			
    if (r <= zero) {

      theDraws.first = zero ;
      return theDraws;
    }
    r = sqrt(-log(r));
    if (r <= split2) {
      r = r - const2;
      result =
	(((((((c7 * r + c6) * r + c5) * r + c4) * r + c3) * r + c2) * r + c1) * r + c0)
	/ (((((((d7 * r + d6) * r + d5) * r + d4) * r + d3) * r + d2) * r + d1) * r + one);
    } else {
      r = r - split2;
      result =
	(((((((e7 * r + e6) * r + e5) * r + e4) * r + e3) * r + e2) * r + e1) * r + e0)
	/ (((((((f7 * r + f6) * r + f5) * r + f4) * r + f3) * r + f2) * r + f1) * r + one);
    }
    if (q < zero) {
      result = -result;
    }
    if (logFile != NULL) {
      *logFile << theDraws.first << '\t' << theDraws.second << endl ;
    }
    theDraws.first = result ;
    return theDraws;
  }
}



patBoolean patNormalWichura::isSymmetric() const {
  return patTRUE ;
}

patBoolean patNormalWichura::isNormal() const {
  return patTRUE ;
}
