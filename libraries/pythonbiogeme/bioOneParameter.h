//-*-c++-*------------------------------------------------------------
//
// File name : bioOneParameter.h
// Author :    Michel Bierlaire
// Date :      Mon Apr 25 19:48:30 2016
//
//--------------------------------------------------------------------

#ifndef bioOneParameter_h
#define bioOneParameter_h

#include "patType.h"
#include "patString.h"

template <class T> class bioOneParameter {

  friend ostream& operator<<(ostream& stream, const bioOneParameter<T>& p) {
    stream << p.theName << "=" << p.theValue << " [" << p.theDescription << "]" ;
    return stream ;
  }
  
  friend patBoolean operator<(const bioOneParameter<T>& p1,
			      const bioOneParameter<T>& p2) {
    return (p1.theName < p2.theName) ;
  }
  
  
 public:
  bioOneParameter<T>(): theName("__uninitializedParameter__") {
    
  }
  bioOneParameter<T>(patString n, 
		     T value, 
		     patString description)  :
  theName(n),
    theDescription(description),
    theValue(value) {
    }
  
  T getValue() const {
    return theValue ;
  } 
  patString getDescription() const  {
    return theDescription ;
  }
  patString getName() const {
    return theName ;
  }
 private:
  patString theName ;
  patString theDescription ;
  T theValue ;
};
#endif
