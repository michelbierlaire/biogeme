//-*-c++-*------------------------------------------------------------
//
// File name : patIterator.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed May 16 15:59:49 2001
//
//--------------------------------------------------------------------

#ifndef patIterator_h
#define patIterator_h

/**
   @doc Generic interface for an iterator
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed May 16 15:59:49 2001)
   @see Gamma et. al (1995) Design Patterns, Addison-Wesley
 */

#include "patType.h"
#include "patConst.h"

template <class T> class patIterator {
public:
  virtual void first() = PURE_VIRTUAL ;
  virtual void next() = PURE_VIRTUAL ;
  virtual patBoolean isDone() = PURE_VIRTUAL ;
  virtual T currentItem() = PURE_VIRTUAL ;
  virtual ~patIterator() {}
protected:
  patIterator() {}
};

#endif
