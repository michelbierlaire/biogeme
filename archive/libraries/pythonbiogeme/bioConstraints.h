//-*-c++-*------------------------------------------------------------
//
// File name : bioConstraints.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Sun Dec 20 11:03:55 2009
//
//--------------------------------------------------------------------

#ifndef bioConstraints_h
#define bioConstraints_h

#include "patError.h"
#include "bioConstraintWrapper.h"

class bioConstraints {
 public:
  virtual ~bioConstraints() ;
  virtual void addConstraints() = PURE_VIRTUAL ;
  vector<bioConstraintWrapper*>* getConstraints() ;
 protected:
  vector<bioConstraintWrapper*> constraints ;
} ;

#endif
