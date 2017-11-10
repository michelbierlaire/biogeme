//-*-c++-*------------------------------------------------------------
//
// File name : patNormalCdf.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sat Jun  4 18:16:18 2005
//
//--------------------------------------------------------------------
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <cassert>
#include "patMath.h"
#include "patNormalCdf.h"
#include "patDisplay.h"
#include "patErrMiscError.h"
#include "patSingletonFactory.h"

patNormalCdf::patNormalCdf() : itmax(100), eps(patEPSILON), fpmin(patMinReal/patEPSILON), cof(6) , stp(2.5066282746310005)  {
  pi = 4.0 * atan(1.0) ;
  oneDivSqrtTwoPi =  1.0/sqrt(2*pi) ;
  cof[0] = 76.18009172947146 ;
  cof[1] = -86.50532032941677 ;
  cof[2] = 24.01409824083091;
  cof[3] = -1.231739572450155 ;
  cof[4] = .1208650973866179e-2 ;
  cof[5] = -.5395239384953e-5 ;
}

patNormalCdf* patNormalCdf::the() {
  return patSingletonFactory::the()->patNormalCdf_the() ;
}

patReal patNormalCdf::compute(patReal gdx, patError*& err) const {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  patReal tgdx = gdx ;
  if (gdx > 40.0) {
    tgdx=40.0 ;
  }
  if (gdx < -40.0) {
    tgdx=-40.0 ;
  }
  if (gdx == 0) {
    tgdx= 1.0 ;
  }
  patReal tx2 = tgdx * tgdx ;

  patReal tz = (1 - (1/(tx2 + 2)) +       
		(1  /((tx2+2)*(tx2+4))) -   
		(5  /((tx2+2)*(tx2+4)*(tx2+6))) +  
		(9  /((tx2+2)*(tx2+4)*(tx2+6)*(tx2+8))) -  
		(129/((tx2+2)*(tx2+4)*(tx2+6)*(tx2+8)*(tx2+10)))) ;
  
  patReal tqx = oneDivSqrtTwoPi * exp(-tx2/2) ;
  patReal tqa = - (tqx/tgdx)*tz ;
  
  if (gdx < -6.0) {
    return(tqa) ;
  }
  if(gdx < 0.0) {
    patReal result  = .5 - .5 * gammp(.5,.5*(gdx*gdx),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    return result ;

  }
  if(gdx < 6.0) {
    patReal result = .5 + .5 * gammp(.5,.5*(gdx*gdx),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    return result ;
  }
  return( 1.0-tqa) ;
 
}


patReal patNormalCdf::gammp(patReal a, patReal x, patError*& err) const {
  if(x < 0.0 || a <= 0.0) {
    err = new patErrMiscError("Bad arguments") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  if(x < a+1.0) {
    patReal result = gser(a,x,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    return (result) ;
  }
  patReal result = 1.0 - gcf(a,x,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return (result) ;
    

}


patReal patNormalCdf::gser(patReal a, patReal x, patError*& err) const {


  patReal gln=gammln(a) ;
  patReal ap=a ;
  patReal sum=1.0/a ;
  patReal del=sum ;
  for (unsigned short n = 0 ; n < itmax ; ++n) {
    ++ap ;
    del *= x/ap ;
    sum += del ;
    if (patAbs(del) < patAbs(sum)*eps) {

      return(sum*exp(-x+a*log(x)-gln)) ;      
    }
  }
  err = new patErrMiscError("Not able to conmpute gser") ;
  WARNING(err->describe());
  return patReal();

}

patReal patNormalCdf::gcf(patReal a, patReal x, patError*& err) const {
  patReal gln = gammln(a) ;
  patReal b = x + 1.0 - a ;
  patReal c = 1.0 / fpmin ;
  patReal d = 1.0 / b ;
  patReal h = d ;
  patReal an, del ;
  for (unsigned short i = 1 ; i <= itmax ; ++i) {
    an = -patReal(i) * (patReal(i)-a) ;
    b += 2.0 ;
    d = an * d + b ;
    if(patAbs(d) < fpmin) {
      d = fpmin ;
    }
    c = b + an / c ;
    if(patAbs(c) < fpmin) {
      c = fpmin ;
    }
    d = 1.0 / d ;
    del = d * c ;
    h *= del ;
    if (patAbs(del-1.0) <= eps) {
      return (exp(-x+a*log(x)-gln)*h) ;
    }
  } 

  err = new patErrMiscError("a too large in gcf") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

patReal patNormalCdf::gammln(patReal xx) const {
  patReal x=xx ;
  patReal y=x ;
  patReal tmp = x + 5.5 ;
  patReal ser(1.000000000190015) ;
  tmp -= (x + 0.5) * log(tmp) ;
  for (unsigned short j = 0 ; j < 6 ; ++j) {
    y = y + 1.0 ;
    ser +=  cof[j] / y ;
  }
  return(-tmp + log(stp * ser / x)) ;

}

patReal patNormalCdf::derivative(patReal x, patError*& err) const {
  return(oneDivSqrtTwoPi * exp(- 0.5 * x * x )) ;
}
