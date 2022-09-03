//-*-c++-*------------------------------------------------------------
//
// File name : bioDerivatives.h
// @date   Fri Apr 13 10:31:21 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include <iostream>
#include <cmath>
#include "bioDebug.h"
#include "bioDerivatives.h"
#include "bioExceptions.h"

// Dealing with exceptions across threads
static std::exception_ptr theExceptionPtr = nullptr ;

/**
 Constructor .cc
 @param n number of variables
*/


bioDerivatives::bioDerivatives(): with_g(true), with_h(true), with_bhhh(true), bhhh_weight(1.0) {

}

void bioDerivatives::resize(bioUInt n) {
  if (n == 0) {
    clear() ;
    return ;
  }
  if (with_g) {
    if (g.size() != n) {
      try {
	g.resize(n) ;
      }
      catch (std::exception& e) {
	throw bioExceptions(__FILE__, __LINE__, e.what()) ;
      }
      catch (const std::bad_alloc&) {
	std::stringstream str ;
	str << "Impossible to allocate memory for vector of size " << n ;  
	throw bioExceptions(__FILE__, __LINE__, str.str()) ;
      }
    }
  }
  if (with_h) {
    if (h.size() != n) {
      try {
	// **** WARNING ****
	// Mon Oct  5 11:51:41 2020
	// Bug with STL.
	
	// When n is large, the following statement kills the process (with
	// a message "bad_alloc") without trigerring an exception.
	// The only way to deal with it would be to abandon the use of STL vectors.
	// This would require a significant re-engineering of the code.
	// This may be considered in the future.
	
	h.resize(n, std::vector<bioReal>(n, 0.0)) ;
      }
      catch (std::exception& e) {
	throw bioExceptions(__FILE__, __LINE__, e.what()) ;
      }
      catch (const std::bad_alloc&) {
	std::stringstream str ;
	str << "Impossible to allocate memory for matrix of size " << n << "x" << n;  
	throw bioExceptions(__FILE__, __LINE__, str.str()) ;
      }
      catch (...) {
	std::stringstream str ;
	str << "Impossible to allocate memory for matrix of size " << n << "x" << n;  
	throw bioExceptions(__FILE__, __LINE__, str.str()) ;
      }
    }
  }

  if (with_bhhh) {
    if (bhhh.size()  != n) {
      try {
	// **** WARNING ****
	// Mon Oct  5 11:51:41 2020
	// Bug with STL.
	
	// When n is large, the following statement kills the process (with
	// a message "bad_alloc") without trigerring an exception.
	// The only way to deal with it would be to abandon the use of STL vectors.
	// This would require a significant re-engineering of the code.
	// This may be considered in the future.
	
	bhhh.resize(n, std::vector<bioReal>(n, 0.0)) ;
      }
      catch (std::exception& e) {
	throw bioExceptions(__FILE__, __LINE__, e.what()) ;
      }
      catch (const std::bad_alloc&) {
	std::stringstream str ;
	str << "Impossible to allocate memory for matrix of size " << n << "x" << n;  
	throw bioExceptions(__FILE__, __LINE__, str.str()) ;
      }
      catch (...) {
	std::stringstream str ;
	str << "Impossible to allocate memory for matrix of size " << n << "x" << n;  
	throw bioExceptions(__FILE__, __LINE__, str.str()) ;
      }
    }
  }
}

void bioDerivatives::clear() {
  g.clear() ;
  h.clear() ;
  bhhh.clear() ;
}

bioUInt bioDerivatives::getSize() const {
  if (with_g) {
    return g.size();
  }
  return 0 ;
}

void bioDerivatives::setEverythingToZero() {
  f = 0 ;
  setDerivativesToZero() ;
}

void bioDerivatives::setDerivativesToZero() {
  if (with_g) {
    std::fill(g.begin(), g.end(), 0.0) ;
  }
  if (with_h) {
    std::fill(h.begin(), h.end(), g) ;
  }
  if (with_bhhh) {
    std::fill(bhhh.begin(), bhhh.end(), g) ;
  }
}

std::ostream& operator<<(std::ostream &str, const bioDerivatives& x) {
  str << "f = " << x.f << std::endl ;
  if (x.with_g) {
    str << "g = [" ; 
    for (std::vector<bioReal>::const_iterator i = x.g.begin() ; i != x.g.end() ; ++i) {
      if (i != x.g.begin()) {
	str << ", " ;
      }
      str << *i ;
    }
    str << "]" << std::endl ;
  }
  if (x.with_h) {
    str << "h = [ " ;
    for (std::vector<std::vector<bioReal> >::const_iterator row = x.h.begin() ; row != x.h.end() ; ++row) {
      if (row != x.h.begin()) {
	str << std::endl ;
      }
      str << " [ " ;
      for (std::vector<bioReal>::const_iterator col = row->begin() ; col != row->end() ; ++col) {
	if (col != row->begin()) {
	  str << ", " ;
	}
	str << *col ;
      }
      str << " ] " << std::endl ;
    }
  }
  if (x.with_bhhh) {
    str << "BHHH = [ " ;
    for (std::vector<std::vector<bioReal> >::const_iterator row = x.bhhh.begin() ; row != x.bhhh.end() ; ++row) {
      if (row != x.bhhh.begin()) {
	str << std::endl ;
      }
      str << " [ " ;
      for (std::vector<bioReal>::const_iterator col = row->begin() ; col != row->end() ; ++col) {
	if (col != row->begin()) {
	  str << ", " ;
	}
	str << *col ;
      }
      str << " ] " << std::endl ;
    }
  }
  return str ;
}

bioDerivatives& bioDerivatives::operator+=(const bioDerivatives& rhs) {
  bioUInt n = getSize() ;
  if (n != rhs.getSize()) {
    std::stringstream str ;
    str << "Incompatible sizes: " << n << " and " << rhs.getSize();  
    throw bioExceptions(__FILE__, __LINE__, str.str()) ;
  }
  f += rhs.f ;
  if (with_g) {
    if (n != g.size()) {
      std::stringstream str ;
      str << "Incorrect allocation of memory for the gradient: " << g.size() << " instead of " << n ;  
      throw bioExceptions(__FILE__, __LINE__, str.str()) ;
    }
    for (bioUInt i = 0 ; i < n ; ++i) {
      g[i] += rhs.g[i] ;
      if (with_h) {
	if (n != h.size()) {
	  std::stringstream str ;
	  str << "Incorrect allocation of memory for the hessian: " << h.size() << " instead of " << n ;  
	  throw bioExceptions(__FILE__, __LINE__, str.str()) ;
	}
	for (bioUInt j = 0 ; j < n ; ++j) {
	  h[i][j] += rhs.h[i][j] ;
	}
      }
      if (with_bhhh) {
	if (n != bhhh.size()) {
	  std::stringstream str ;
	  str << "Incorrect allocation of memory for BHHH: " << bhhh.size() << " instead of " << n ;  
	  throw bioExceptions(__FILE__, __LINE__, str.str()) ;
	}
	for (bioUInt j = 0 ; j < n ; ++j) {
	  bhhh[i][j] += rhs.bhhh[i][j] ;
	}
      }
    }
  }
  return *this ;
}

void bioDerivatives::dealWithNumericalIssues() {
  bioUInt n = getSize() ;
  if (!std::isfinite(f)) {
    f = -std::numeric_limits<bioReal>::max() ;
  }
  if (with_g) {
    for (bioUInt i = 0 ; i < n ; ++i) {
      if (!std::isfinite(g[i])) {
	g[i] = -std::numeric_limits<bioReal>::max() ;
      }
      if (with_h) {
	for (bioUInt j = i ; j < n ; ++j) {
	  if (!std::isfinite(h[i][j])) {
	    h[i][j] = -std::numeric_limits<bioReal>::max() ;
	  }
	}
      }
      if (with_bhhh) {
	if (n != bhhh.size()) {
	  std::stringstream str ;
	  str << "Incorrect allocation of memory for BHHH: " << bhhh.size() << " instead of " << n ;  
	  throw bioExceptions(__FILE__, __LINE__, str.str()) ;
	}
	for (bioUInt j = i ; j < n ; ++j) {
	  if (!std::isfinite(bhhh[i][j])) {
	    bhhh[i][j] = -std::numeric_limits<bioReal>::max() ;
	  }
	}
      }
    }
  }
}

void bioDerivatives::computeBhhh() {
  bioUInt n = getSize() ;
  if (with_bhhh) {
    if (n != bhhh.size()) {
      std::stringstream str ;
      str << "Incorrect allocation of memory for BHHH: " << bhhh.size() << " instead of " << n ;  
      throw bioExceptions(__FILE__, __LINE__, str.str()) ;
    }
    for (bioUInt i = 0 ; i < n ; ++i) {
      for (bioUInt j = i ; j < n ; ++j) {
	if (bhhh_weight == 1.0) {
	  bhhh[i][j] = g[i] * g[j] ;
	}
	else {
	  bhhh[i][j] = bhhh_weight * g[i] * g[j] ;
	}
      }
    }
  }
}
