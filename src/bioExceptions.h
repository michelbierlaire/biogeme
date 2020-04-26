//-*-c++-*------------------------------------------------------------
//
// File name : bioExceptions.h
// @date   Fri Apr 13 09:03:41 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExceptions_h
#define bioExceptions_h

#include <stdexcept>
#include <sstream>
#include <exception>
#include "bioString.h"

// Prepares the messages for the exceptions
class bioExceptions: public std::runtime_error {
public:
  bioExceptions(bioString f, int l, bioString m) throw() : std::runtime_error(m) {
    std::stringstream str ;
    str << f << ":" << l << ": Biogeme exception: " << m ;
    msg = str.str() ;
  }
  virtual ~bioExceptions() throw() {}
  const char *what() const throw() {
    return msg.c_str();
  }

private:
  bioString msg ;
};

class bioExceptNullPointer: public bioExceptions {
public:
  bioExceptNullPointer(bioString f, int l, bioString ptr) throw() : bioExceptions(f,l,"Null pointer: "+ptr) {
  }
  virtual ~bioExceptNullPointer() throw() {}
};


template <class T> class bioExceptOutOfRange: public bioExceptions {
  bioString msg ;
 public:
  bioExceptOutOfRange(bioString f, int l, T v, T lb, T ub) throw() : bioExceptions(f,l,"Value "+ std::to_string(v) + " out of range [" + std::to_string(lb) + "," + std::to_string(ub) + "]") {
    
  }
  virtual ~bioExceptOutOfRange() throw() {}
};

#endif
