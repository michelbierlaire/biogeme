//-*-c++-*------------------------------------------------------------
//
// File name : bioNormalCdf.cc
// @date   Wed May 30 15:54:36 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------


#include <cmath>
#include "bioNormalCdf.h"
#include "bioExceptions.h"
#include "bioConst.h"

bioNormalCdf::bioNormalCdf() : itmax(100), eps(bioEpsilon), fpmin(bioMinReal/bioEpsilon), cof(6) , stp(2.5066282746310005)  {
  oneDivSqrtTwoPi =  1.0/sqrt(2*bioPi) ;
  cof[0] = 76.18009172947146 ;
  cof[1] = -86.50532032941677 ;
  cof[2] = 24.01409824083091;
  cof[3] = -1.231739572450155 ;
  cof[4] = .1208650973866179e-2 ;
  cof[5] = -.5395239384953e-5 ;
}

bioReal bioNormalCdf::compute(bioReal gdx) const {
  bioReal tgdx = gdx ;
  if (gdx > 40.0) {
    tgdx=40.0 ;
  }
  if (gdx < -40.0) {
    tgdx=-40.0 ;
  }
  if (gdx == 0) {
    tgdx= 1.0 ;
  }
  bioReal tx2 = tgdx * tgdx ;

  bioReal tz = (1 - (1/(tx2 + 2)) +       
		(1  /((tx2+2)*(tx2+4))) -   
		(5  /((tx2+2)*(tx2+4)*(tx2+6))) +  
		(9  /((tx2+2)*(tx2+4)*(tx2+6)*(tx2+8))) -  
		(129/((tx2+2)*(tx2+4)*(tx2+6)*(tx2+8)*(tx2+10)))) ;
  
  bioReal tqx = oneDivSqrtTwoPi * exp(-tx2/2) ;
  bioReal tqa = - (tqx/tgdx)*tz ;
  
  if (gdx < -6.0) {
    return(tqa) ;
  }
  if(gdx < 0.0) {
    bioReal result  = .5 - .5 * gammp(.5,.5*(gdx*gdx)) ;
    return result ;

  }
  if(gdx < 6.0) {
    bioReal result = .5 + .5 * gammp(.5,.5*(gdx*gdx)) ;
    return result ;
  }
  return( 1.0-tqa) ;
 
}


bioReal bioNormalCdf::gammp(bioReal a, bioReal x) const {
  if(x < 0.0 || a <= 0.0) {
      std::stringstream str ;
      str << "Bad arguments: " << a << " and " << x ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  }
  if(x < a+1.0) {
    bioReal result = gser(a,x) ;
    return (result) ;
  }
  bioReal result = 1.0 - gcf(a,x) ;
  return (result) ;
    

}


bioReal bioNormalCdf::gser(bioReal a, bioReal x) const {


  bioReal gln=gammln(a) ;
  bioReal ap=a ;
  bioReal sum=1.0/a ;
  bioReal del=sum ;
  for (bioUInt n = 0 ; n < itmax ; ++n) {
    ++ap ;
    del *= x/ap ;
    sum += del ;
    if (std::abs(del) < std::abs(sum)*eps) {

      return(sum*exp(-x+a*log(x)-gln)) ;      
    }
  }
  std::stringstream str ;
  str << "Not able to conmpute gser" ;
  throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  return bioReal();

}

bioReal bioNormalCdf::gcf(bioReal a, bioReal x) const {
  bioReal gln = gammln(a) ;
  bioReal b = x + 1.0 - a ;
  bioReal c = 1.0 / fpmin ;
  bioReal d = 1.0 / b ;
  bioReal h = d ;
  bioReal an, del ;
  for (bioUInt i = 1 ; i <= itmax ; ++i) {
    an = -bioReal(i) * (bioReal(i)-a) ;
    b += 2.0 ;
    d = an * d + b ;
    if(std::abs(d) < fpmin) {
      d = fpmin ;
    }
    c = b + an / c ;
    if(std::abs(c) < fpmin) {
      c = fpmin ;
    }
    d = 1.0 / d ;
    del = d * c ;
    h *= del ;
    if (std::abs(del-1.0) <= eps) {
      return (exp(-x+a*log(x)-gln)*h) ;
    }
  } 
  std::stringstream str ;
  str << "a too large in gcf" ;
  throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  return bioReal() ;
}

bioReal bioNormalCdf::gammln(bioReal xx) const {
  bioReal x=xx ;
  bioReal y=x ;
  bioReal tmp = x + 5.5 ;
  bioReal ser(1.000000000190015) ;
  tmp -= (x + 0.5) * log(tmp) ;
  for (bioUInt j = 0 ; j < 6 ; ++j) {
    y = y + 1.0 ;
    ser +=  cof[j] / y ;
  }
  return(-tmp + log(stp * ser / x)) ;

}

bioReal bioNormalCdf::derivative(bioReal x) const {
  return(oneDivSqrtTwoPi * exp(- 0.5 * x * x )) ;
}
