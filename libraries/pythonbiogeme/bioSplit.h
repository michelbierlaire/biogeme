//-*-c++-*------------------------------------------------------------
//
// File name : bioSplit.h
// Author :    Michel Bierlaire
// Date :      Sun Apr  4 17:45:07 2010
//
//--------------------------------------------------------------------

#ifndef bioSplit_h
#define bioSplit_h

// Split a list of n items into p sublists. This has been designed
// to split the data set on several processors.
// The list is supposed to be numbered from 0 to n-1.
// Each interval is associated to two numbers (b and e), such that the
// interval contains all integers between b and e-1.

#include "patType.h"
#include "patError.h"
#include "bioIteratorSpan.h"

class bioSplit {
  friend ostream& operator<<(ostream &str, const bioSplit& x) ;

 public:
  bioSplit(patULong n, patULong p) ;
  bioIteratorSpan getSublist(patULong index, patError*& err) ;
  
 private:
  patULong length ;
  vector<bioIteratorSpan> sublists ;
};
#endif
