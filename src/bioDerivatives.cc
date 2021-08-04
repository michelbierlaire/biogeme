//-*-c++-*------------------------------------------------------------
//
// File name : bioDerivatives.h
// @date   Fri Apr 13 10:31:21 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include <iostream>
#include "bioDebug.h"
#include "bioDerivatives.h"
#include "bioExceptions.h"

// Dealing with exceptions across threads
static std::exception_ptr theExceptionPtr = nullptr ;

/**
 Constructor .cc
 @param n number of variables
*/


bioDerivatives::bioDerivatives() {

}

void bioDerivatives::resize(bioUInt n) {
  if (n == getSize()) {
    return ;
  }
  clear() ;
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

void bioDerivatives::clear() {
  g.clear() ;
  h.clear() ;
}

bioUInt bioDerivatives::getSize() const {
  return g.size();
}

void bioDerivatives::setDerivativesToZero() {
  std::fill(g.begin(),g.end(),0.0) ;
  std::fill(h.begin(),h.end(),g) ;
}

void bioDerivatives::setGradientToZero() {
  std::fill(g.begin(),g.end(),0.0) ;
}

std::ostream& operator<<(std::ostream &str, const bioDerivatives& x) {
  str << "f = " << x.f << std::endl ;
  str << "g = [" ; 
  for (std::vector<bioReal>::const_iterator i = x.g.begin() ; i != x.g.end() ; ++i) {
    if (i != x.g.begin()) {
      str << ", " ;
    }
    str << *i ;
  }
  str << "]" << std::endl ;
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
  return str ;
}
