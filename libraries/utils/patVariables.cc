//-*-c++-*------------------------------------------------------------
//
// File name : patVariables.cc
// Author :   Michel Bierlaire
// Date :     Mon Dec 21 14:29:50 1998
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <stdlib.h>
#include <algorithm>
#include <functional>
#include <cassert>
#include "patMath.h"
#include "patVariables.h"
#include "patDisplay.h"
#include "patConst.h"

/// Overloads the unary minus operator
patVariables operator-(const patVariables& x) {
  patVariables result(x.size()) ;

  transform(x.begin(),
	    x.end(),
	    result.begin(),
	    negate<patReal>()) ;
  return(result) ;

}

/// Overloads the binary minus operator
patVariables operator-(const patVariables& x,
		       const patVariables& y) {

  
  patVariables result(x.size()) ;
  
  if (x.size() != y.size()) {

    WARNING("Inconsistent sizes. Unpredictible result.") ;
    DEBUG_MESSAGE("x= " << x) ;
    DEBUG_MESSAGE("y= " << y) ;

    DEBUG_MESSAGE("Size x = " << x.size() << " Size y = " << y.size()) ; 
    return(x) ;
  }
  
  transform(x.begin(),
	    x.end(),
	    y.begin(),
	    result.begin(),
	    minus<patReal>()) ;
  return(result) ;
}

/// Overloads the binary divide operator
patVariables operator/(const patVariables& x,
		       const patReal& y) {

  
  
  patVariables result(x.size())  ;
  for (patVariables::size_type i = 0 ; i < result.size() ; ++i) {
    result[i] = x[i] / y ;
  }
  
  return(result) ;
}


patVariables &operator+=(patVariables& x, const patVariables& y) {
  
  if (x.size() != y.size()) {
    DEBUG_MESSAGE("Size x = " << x.size() << " Size y = " << y.size()) ; 
    FATAL("Inconsistent sizes. Unpredictible result.") ;
    return(x) ;
  }
  
  transform(x.begin(),
	    x.end(),
	    y.begin(),
	    x.begin(),
	    plus<patReal>()) ;
  return(x) ;
}

patVariables &operator-=(patVariables& x, const patVariables& y) {
  
  if (x.size() != y.size()) {
    WARNING("Inconsistent sizes. Unpredictible result.") ;
    DEBUG_MESSAGE("Size x = " << x.size() << " Size y = " << y.size()) ; 
    return(x) ;
  }
  
  transform(x.begin(),
	    x.end(),
	    y.begin(),
	    x.begin(),
	    minus<patReal>()) ;
  return(x) ;
}


/// Overloads the plus operator
patVariables operator+(const patVariables& x,
		       const patVariables& y) {

  
  patVariables result(x.size()) ;
  
  if (x.size() != y.size()) {
    WARNING("Inconsistent sizes. Unpredictible result.") ;
    DEBUG_MESSAGE("Size x = " << x.size() << " Size y = " << y.size()) ; 
    return(result) ;
  }
  
  transform(x.begin(),
	    x.end(),
	    y.begin(),
	    result.begin(),
	    plus<patReal>()) ;
  return(result) ;
}

/// Overloads the multiplies operator
patVariables operator*(patReal alpha,
		       const patVariables& y) {

  patVariables result(y.size()) ;

  transform(y.begin(),
	    y.end(),
	    result.begin(),
	    bind2nd(multiplies<patReal>(),alpha)) ;
  return(result) ;
}

patVariables &operator*=(patVariables& x, const float alpha) {
  transform(x.begin(),
	    x.end(),
	    x.begin(),
	    bind2nd(multiplies<float>(),alpha)) ;
  return(x) ;

}

patVariables &operator/=(patVariables& x, const float alpha) {
  transform(x.begin(),
	    x.end(),
	    x.begin(),
	    bind2nd(multiplies<float>(),1.0/alpha)) ;
  return(x) ;

}



/// Decide if two patVariables are symmetric
//  patBoolean isSymmetric::operator()(const patVariables& x) {
//    if (x.size() != ref.size()) {
//      WARNING("Inconsistent sizes.") ;
//      DEBUG_MESSAGE("Size x = " << ref.size() << " ref size = " << ref.size()) ; 
//      return(patFALSE) ;
//    }
//    patVariables::const_iterator itx, ity ;

//    for (itx = x.begin(), ity = ref.begin() ;
//         ity != ref.end() ;
//         ++itx, ++ity) {
//      if ( (*itx) != -(*ity)) return(patFALSE) ; 
//    }
//    return(patTRUE) ;
//  }

///Compare two patVariables using the absolute value of their components.

patBoolean compSymmetric::operator()(const patVariables& x, 
				     const patVariables& y) {
  if (x.size() < y.size()) {
    WARNING("Incompatible sizes") ;
    return patTRUE ;
  }
  if (x.size() > y.size()) {
    WARNING("Incompatible sizes") ;
    return patFALSE ;
  }
  patVariables::const_iterator iterx, itery ;
  for (iterx = x.begin() , itery = y.begin() ;
       iterx != x.end() ;
       ++iterx , ++itery) {
    if (patAbs(*iterx) < patAbs(*itery)) {
      return patTRUE ;
    }
    if (patAbs(*iterx) > patAbs(*itery)) {
      return patFALSE ;
    }
  }
  return patFALSE ;
}

ostream& operator<<(ostream &str, const patVariables& x) {
  if (x.size() == 0) return(str) ;
  str << "(" ;
  for (patVariables::const_iterator i = x.begin() ;
       i != x.end() ;
       ++i) {
    str << *i << '\t' ;
  }
  str << ")" ;
  return(str); 

}


void x_plus_ay(patVariables& x, float a, const patVariables& y) {
  for (unsigned long i = 0 ; i < x.size() ; ++i) {
    x[i] += a * y[i] ;
  }
}

patReal norm2(patVariables& x) {
  patReal result(0.0) ;
  for (unsigned long i = 0 ; i < x.size() ; ++i) {
    result += x[i] * x[i] ;
  }
  return sqrt(result) ;
}
