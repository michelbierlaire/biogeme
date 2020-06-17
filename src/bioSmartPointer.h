//-*-c++-*------------------------------------------------------------
//
// File name : bioSmartPointer.h
// @date   Wed Jun 17 12:08:56 2020
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#ifndef bioSmartPointer_h
#define bioSmartPointer_h

#include "bioTypes.h"
#include "bioReferenceCounting.h"

template <typename T> class bioSmartPointer {
 private:
  T*  thePointer ;
  bioReferenceCounting* refs ;
public:
  bioSmartPointer() : thePointer(NULL), refs(NULL) {
    refs = new bioReferenceCounting() ;
    refs->add() ;
  }
  
  bioSmartPointer(T* aPointer) : thePointer(aPointer), refs(NULL) {
    refs = new bioReferenceCounting() ;
    refs->add() ;
  }

  bioSmartPointer(const bioSmartPointer<T>& aSp): thePointer(aSp.thePointer), refs(aSp.refs) {
    refs->add() ;
  }
  
  ~bioSmartPointer() {
    if (refs->release() == 0) {
      delete thePointer ;
      delete refs ;
    }
  }

  bioBoolean operator==(T &p) const {
    return thePointer == p.thePointer ;
  }

  bioBoolean operator==(T *p) const {
    return thePointer == p ;
  }

  bioBoolean operator!=(T &p) const {
    return thePointer != p.thePointer ;
  }

  bioBoolean operator!=(T *p) const {
    return thePointer != p ;
  }

  // bioBoolean isNull() const {
  //   return thePointer == NULL ;
  // }
  
  T& operator* () {
    return *thePointer;
  }
  
  T* operator-> () {
    return thePointer;
  }

  const T* operator-> () const {
    return thePointer;
  }
  
  bioSmartPointer<T>& operator=(const bioSmartPointer<T>& aSp) {
    if (this != &aSp) {
      if (refs->release() == 0) {
	delete thePointer ;
	delete refs ;
      }
      thePointer = aSp.thePointer ;
      refs = aSp.refs ;
      refs->add() ;
    }
    return *this ;
  }
  
};

#endif
