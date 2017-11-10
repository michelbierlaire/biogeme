#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <numeric>
#include <algorithm>
#include "patMath.h"
#include "patCompare.h"
#include "patErrOutOfRange.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "patHouseholder.h"


patHouseholder::patHouseholder(): v(), norm(0.0), start(0), stop(0) {} ;
patHouseholder::patHouseholder(const patVariables& _v): 
  v(_v), 
  start(0),
  stop(_v.size()) {
  
  norm = inner_product(_v.begin(),_v.end(),_v.begin(),0.0) ;
}

patVariables patHouseholder::operator*(const patVariables& x) const {
  if (x.size() != v.size()) {
    WARNING("Incompatible sizes: v is " << v.size() << " x is " << x.size()) ;
    patVariables dummy ;
    return (dummy) ;
  }
  patReal alpha = inner_product(v.begin(),v.end(),x.begin(),0.0) ;
  alpha *= 2.0 / norm ;
  patVariables res = x - alpha * v ;
  return res ;
}

patVariables::size_type patHouseholder::getDimension() const {
  return v.size() ;
}

void patHouseholder::setToNullify(const patVariables& x, 
				  patVariables::size_type k,
				  patVariables::size_type j,
				  patBoolean checkAssumptions,
				  patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  patVariables::size_type n = x.size() ;
  if (checkAssumptions) {
    if (j <= k) {
      stringstream str ;
      str << "j (" << j << ") must be greater than " << k  ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
    if (j >= n) {
      stringstream str ;
      str << "j (" <<j<< ") must not be greater than " << n-1 << '\0' ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
//     for (patVariables::size_type i = k ;
// 	 i <= j ;
// 	 ++i) {
//       if (x[i] == 0) {
// 	stringstream str ;
// 	str << "x[" << i << "]=0" << '\0' ;
// 	err = new patErrMiscError(str.str()) ;
// 	WARNING(err->describe()) ;
// 	return ;
//       }
//       if (patAbs(x[i]) <= patEPSILON) {
// 	WARNING( "Very small x[i] = " << x[i]);
// 	// We do not return here because it is not necessarily fatal. 
//       }
//     }
  }

  /// Init v to zero
  v.resize(x.size()) ;
  fill(v.begin(),v.end(),0.0);

  patReal m = patAbs(*max_element(x.begin(),
				  x.end(),
				  compAbsValue())) ;

  patReal alpha = 0.0 ;
  
  for (patVariables::size_type i = k ;
       i <= j ;
       ++i) {
    v[i] = x[i] / m ;
    alpha += v[i] * v[i] ;
  }
  alpha = sqrt(alpha) ;
  norm = 2 * alpha * (alpha + patAbs(v[k])) ;
  v[k] += patSgn(v[k])*alpha ;
  start = k ;
  stop = j+1 ;
}

patMyMatrix* patHouseholder::multiply(patMyMatrix* A,
				      patError*& err) const {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return A ;
  }
  if (A == NULL) {
    err = new patErrNullPointer("patMyMatrix") ;
    WARNING(err->describe()) ;
    return A ;
  }
  unsigned long n = A->nRows() ;
  if (n == 0) return A ;
  unsigned long q = A->nCols() ;
  if (n != getDimension()) {
    stringstream str ;
    str << "Incompatible sizes: A is " 
	<< n
	<< "x" 
	<< q 
	<< " and P is " 
	<<getDimension() 
	<< "x" 
	<< getDimension() << '\0';
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return A ;
  }
  
  if (patAbs(norm) < patEPSILON) {
    WARNING("Small norm: " << norm << " considered as zero") ;
    // If the norm is null the Householder matrix is the identity
    return A ;
  }
  for (unsigned long p = 0 ;
       p < q ;
       ++p) {
    patReal s = 0.0 ;
    for (patVariables::size_type ll = start ; ll < stop ; ++ll) {
      s += v[ll] * (*A)[ll][p] ;
    }
    s *= 2.0 / norm ;
    for (patVariables::size_type ll = start ; ll < stop ; ++ll) {
      (*A)[ll][p] -= s * v[ll] ;
    }
  }
  return A ;
}

patVariables* patHouseholder::multiply(patVariables* A,
				    patError*& err) const {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return A ;
  }
  if (A == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return A ;
  }
  unsigned long n = A->size() ;
  if (n == 0) return A ;
  if (n != getDimension()) {
    stringstream str ;
    str << "Incompatible sizes: x is " 
	<< n
	<< "x" 
	<< 1 
	<< " and P is " 
	<<getDimension() 
	<< "x" 
	<< getDimension() << '\0';
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return A ;
  }
  
  if (patAbs(norm) < patEPSILON) {
    WARNING("Small norm: " << norm << " considered as zero") ;
    // If the norm is null the Householder matrix is the identity
    return A ;
  }
  //DEBUG_MESSAGE("start=" <<start<<" stop=" << stop) ;
  patReal s = 0.0 ;
  for (patVariables::size_type ll = start ; ll < stop ; ++ll) {
    s += v[ll] * (*A)[ll] ;
  }
  s *= 2.0 / norm ;
  for (patVariables::size_type ll = start ; ll < stop ; ++ll) {
    (*A)[ll] -= s * v[ll] ;
  }
  return A ;

}

patBoolean patHouseholder::isIdentity() { 
  return(patAbs(norm)<patEPSILON) ;
}
