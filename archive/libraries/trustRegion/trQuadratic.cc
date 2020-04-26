//-*-c++-*------------------------------------------------------------
//
// File name : trQuadratic.cc
// Author :    Michel Bierlaire
// Date :      Mon Jan 24 10:11:10 2000
//
//--------------------------------------------------------------------

#include "trQuadratic.h"

trQuadratic::trQuadratic(const trVector& x) : scaling(x) {
}


patReal trQuadratic::getFunction(const trVector& x,
				  patError*& err) const {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  if (x.size() != getDimension()) {
    err = new patErrMiscError("Incompatible dimensions") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }

  patReal res = 0.0 ;

  for (trVector::size_type i = 0 ;
       i < getDimension() ;
       ++i) {
    res += scaling[i] * x[i] * x[i] ;
  }
  return res ;
  
};

trVector trQuadratic::getGradient(const trVector& x,
				  patError*& err) const {
  
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trVector() ;
  }
  if (x.size() != getDimension()) {
    err = new patErrMiscError("Incompatible dimensions") ;
    WARNING(err->describe()) ;
    return trVector() ;
  }

  trVector res(x.size()) ;
  
  for (trVector::size_type i = 0 ;
       i < getDimension() ;
       ++i) {
    res[i] = 2.0 * scaling[i] * x[i] ;
  }
  return(res) ;
}

trHessian* trQuadratic::computeHessian(const patVariables& x,
				       trHessian& hessianObj,
				       patError*& err) const {
  
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  if (x.size() != getDimension()) {
    err = new patErrMiscError("Incompatible dimensions") ;
    WARNING(err->describe()) ;
    return NULL ;
  }

  if (hessianObj.getDimension() != getDimension()) {
    stringstream str ;
    str << "Function designed for " << getDimension()
	<< " variables. Hessian designed to store " << hessianObj.getDimension()
	<<" values"  << '\0' ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL ;
  }

  
  for (trVector::size_type i = 0 ;
       i < getDimension() ;
       ++i) {
    hessianObj.setElement(i,i,2.0 * scaling[i],err) ;
    if (err != NULL) {
      WARNING("Warning in loop at " << i) ;
      WARNING(err->describe()) ;
      return NULL ;
    }
  }
  return &hessianObj ;
}


unsigned long trQuadratic::getDimension() const {
  return scaling.size() ;
}

trVector trQuadratic::getHessianTimesVector(const trVector& x,
					    const trVector& v,
					    patError*& err) const {
  
  WARNING("Not implemented") ;
  return trVector() ;
}
