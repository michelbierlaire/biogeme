//-*-c++-*------------------------------------------------------------
//
// File name : trBounds.cc
// Date :      Sun Nov 26 15:31:21 2000
//
//--------------------------------------------------------------------

#include <algorithm>
#include <functional>
#include "patMath.h"
#include "trBounds.h"
#include "patErrOutOfRange.h"
#include "patErrMiscError.h"

trBounds::trBounds(unsigned long size)  :
  lower(size,-patMaxReal),
  upper(size,patMaxReal) {

}

trBounds::trBounds(const patVariables& l, const patVariables& u) :
  lower(l),
  upper(u) {

}

trBounds::trBounds(const trBounds& b, 
		   const trVector& x,
		   const vector<patActivityStatus>& activity,
		   patReal trustRegionRadius,
		   patError*& err) {

  if (!b.isFeasible(x,err)) {
    err = new patErrMiscError("Center of trust region must be feasible") ;
    WARNING(err->describe()) ;
    return ;
  }

  if (trustRegionRadius <= 0.0) {
    err = new patErrOutOfRange<patReal>(trustRegionRadius,0,patMaxReal) ;
    WARNING(err->describe()) ;
    return ;
  }



  for (unsigned long i = 0 ;
       i < x.size() ;
       ++i) {
    if (activity[i] == patFree) {
      lower.push_back(patMax(b.lower[i]-x[i],-trustRegionRadius)) ;
      upper.push_back(patMin(b.upper[i]-x[i],trustRegionRadius)) ;
    }
  }
}

void trBounds::setBounds(unsigned long variable,
			 patReal _lower,
			 patReal _upper,
			 patError*& err) {

  if (variable >= getDimension()) {
    err = new patErrOutOfRange<unsigned long>(variable,0,getDimension()-1) ;
    WARNING(err->describe()) ;
    return ;
  }

  if (_lower > _upper) {
    stringstream str ;
    str << "Lower bound " << _lower << " is larger than upper bound "
	<< _upper  ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }

  lower[variable] = _lower ;
  upper[variable] = _upper ;
}

patReal trBounds::getMaxStep(const trVector& x,
			     const trVector& d,
			     trVector* result,
			     patError*& err) const {

  if (x.size() != getDimension()) {
    stringstream str ;
    str << "x is of size " << x.size() << " and should be " 
	<< getDimension() ;
    err = new patErrMiscError(str.str()) ;
    return patReal() ;
  }
  if (d.size() != getDimension()) {
    stringstream str ;
    str << "d is of size " << d.size() << " and should be " 
	<< getDimension() ;
    err = new patErrMiscError(str.str()) ;
    return patReal() ;
  }
  if (result != NULL) {
    if (result->size() != getDimension()) {
      stringstream str ;
      str << "result is of size " << result->size() << " and should be " 
	  << getDimension() ;
      err = new patErrMiscError(str.str()) ;
      return patReal() ;
    }
  }
  
  if (!isFeasible(x,err)) {
    err = new patErrMiscError("Unfeasible point") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }

  patReal alpha = patMaxReal ;
  unsigned long index = getDimension() ;

  for (unsigned long i = 0 ;
       i < getDimension() ;
       ++i) {
    if (d[i] > 0.0) {
      patReal tmpAlpha = (upper[i] - x[i]) / d[i] ;
      if (tmpAlpha < alpha) {
	alpha = tmpAlpha ;
	index = i ;
      }
    }
    else if (d[i] < 0.0) {
      patReal tmpAlpha = (lower[i] - x[i]) / d[i] ;
      if (tmpAlpha < alpha) {
	alpha = tmpAlpha ;
	index = i ;
      }
    }
  }
  if (result != NULL) {
    *result = x + alpha * d ;
    for (unsigned long i = 0 ;
	 i < result->size() ;
	 ++i) {
      if (index == i) {
	if (d[i] < 0) {
	  (*result)[i] = lower[i] ;
	}
	else if (d[i] > 0) {
	  (*result)[i] = upper[i] ;
	}
      }
    }
  }
  return alpha ;
  
}

patBoolean trBounds::isFeasible(const trVector& x,
				patError*& err) const {
  
  if (x.size() != getDimension()) {
    stringstream str ;
    str << "x is of size " << x.size() << " and should be " 
	<< getDimension() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patBoolean() ;
  }
  
  for (unsigned long i = 0 ; 
       i < getDimension() ;
       ++i) {
    if (x[i] < lower[i]) {
      return patFALSE ;
    }
    if (x[i] > upper[i]) {
      return patFALSE ;
    }
  } 
    
  return patTRUE ;
}

patReal trBounds::getLower(unsigned long index,
			   patError*& err) const {

  if (index >= getDimension()) {
    err = new patErrOutOfRange<unsigned long>(index,0,getDimension()-1) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return lower[index] ;
}

patReal trBounds::getUpper(unsigned long index,
			   patError*& err) const {
  
  if (index >= getDimension()) {
    err = new patErrOutOfRange<unsigned long>(index,0,getDimension()-1) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return upper[index] ;
}


unsigned long trBounds::getDimension() const {
  return lower.size() ;
}

trVector trBounds::getProjection(const trVector& x,
				 patError*& err) const {
  
  if (x.size() != getDimension()) {
    stringstream str ;
    str << "x is of size " << x.size() << " and should be " 
	<< getDimension() ;
    err = new patErrMiscError(str.str()) ;
    return trVector() ;
  }
  
  trVector result(x.size()) ;

  for (unsigned long i = 0 ; i < getDimension() ; ++i) {
    if (x[i] < lower[i]) {
      result[i] = lower[i] ;
    }
    else if (x[i] > upper[i]) {
      result[i] = upper[i] ;
    }
    else {
      result[i] = x[i] ;
    }
  }
  return result ;
}


trBounds::patBreakPointsContainer trBounds::getBreakPoints(const trVector& x,
							  const trVector& d,
							  patReal radius,
							  patError*& err) const {

  if (radius <= 0) {
    err = new patErrOutOfRange<patReal>(radius,0,patMaxReal) ;
    WARNING(err->describe()) ;
    return trBounds::patBreakPointsContainer() ;
  }
  if (x.size() != getDimension()) {
    stringstream str ;
    str << "x is of size " << x.size() << " and should be " 
	<< getDimension() ;
    err = new patErrMiscError(str.str()) ;
    return trBounds::patBreakPointsContainer();
  }
  if (d.size() != getDimension()) {
    stringstream str ;
    str << "d is of size " << d.size() << " and should be " 
	<< getDimension() ;
    err = new patErrMiscError(str.str()) ;
    return trBounds::patBreakPointsContainer();
  }
  if (!isFeasible(x,err)) {
    err = new patErrMiscError("Unfeasible point") ;
    WARNING(err->describe()) ;
    cout << *this << endl;
    return trBounds::patBreakPointsContainer() ;
  }

  patBreakPointsContainer breakpoints ;
  for (unsigned long i = 0 ;
       i < getDimension() ;
       ++i) {
    patReal bp = 0.0 ;
    if (d[i] > patEPSILON) {
      bp = patMin(upper[i]-x[i],radius)/d[i] ;
    }
    else if (d[i] < -patEPSILON) {
      bp = patMax(lower[i]-x[i],-radius)/d[i] ;
    }

    patBreakPoint tmp(bp,i) ;
    breakpoints.push(tmp) ;
    
  }

  return breakpoints ;
}


vector<trBounds::patActivityStatus> trBounds::getActivity(trVector& x,
							  patError*& err) const {
  
  if (x.size() != getDimension()) {
    stringstream str ;
    str << "x is of size " << x.size() << " and should be " 
	<< getDimension() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return vector<trBounds::patActivityStatus>();
  }

  if (!isFeasible(x,err)) {
    err = new patErrMiscError("x must be feasible") ;
    WARNING(err->describe()) ;
    return vector<trBounds::patActivityStatus>();
  }
  
  vector<trBounds::patActivityStatus> res ;
  
  for (unsigned long i = 0 ; i < getDimension() ; ++i) {
    patActivityStatus st ;
    if (patAbs(x[i] - lower[i]) <= patEPSILON) {
      st = patLower ;
      x[i] = lower[i] ;
    }
    else if (patAbs(x[i] - upper[i]) <= patEPSILON) {
      st = patUpper ;
      x[i] = upper[i] ;
    }
    else {
      st = patFree ;
    }
    res.push_back(st) ; 
  }
  return res ;
}

ostream& operator<<(ostream &str, const trBounds& x) {
  for (unsigned long i = 0 ; i < x.getDimension() ; ++i) {
    str << x.lower[i] << " <= x[" << i <<"] <= " << x.upper[i] << endl ;
  }
  return str ;
}

vector<patReal> trBounds::getLowerVector() {
  return lower ;
}

vector<patReal> trBounds::getUpperVector() {
  return upper ;
}
