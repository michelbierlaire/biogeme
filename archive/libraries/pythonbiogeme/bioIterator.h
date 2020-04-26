//-*-c++-*------------------------------------------------------------
//
// File name : bioIterator.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Wed Jun 17 11:15:26  2009
//
//--------------------------------------------------------------------

#ifndef bioIterator_h
#define bioIterator_h

/*!
 This virtual class represents an interface for an iterator
*/

#include "patType.h"
#include "patConst.h"

template <class T> class bioIterator {
public:
  virtual void first() = PURE_VIRTUAL ;
  virtual void next() = PURE_VIRTUAL ;
  virtual patBoolean isDone() = PURE_VIRTUAL ;
  virtual T currentItem() = PURE_VIRTUAL ;
  virtual patULong nbrOfItems() const = PURE_VIRTUAL ;
protected:
  bioIterator() {}
  virtual ~bioIterator() {}
};

#endif
