//-*-c++-*------------------------------------------------------------
//
// File name : bioCountOperations.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Sun Sep 13 17:51:54 2009
//
//--------------------------------------------------------------------

#ifndef bioCountOperations_h
#define bioCountOperations_h

#include "patType.h"

class bioCountOperations {

 public:
  patULong fctBeforeSimplification ;
  patULong fctAfterSimplification ;
  patULong gradBeforeSimplification ;
  patULong gradAfterSimplification ;
};

ostream& operator<<(ostream &str, const bioCountOperations& x) ;

#endif
