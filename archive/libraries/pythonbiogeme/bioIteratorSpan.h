//-*-c++-*------------------------------------------------------------
//
// File name : bioIteratorSpan.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Thu Jul 16 22:20:53 2009
//
//--------------------------------------------------------------------

#ifndef bioIteratorSpan_h
#define bioIteratorSpan_h

#include "patString.h"
#include "patType.h"
#include "patConst.h"

// The iterators run from firstRow to lastRow-1

class bioIteratorSpan {
public:
  bioIteratorSpan() ;
  bioIteratorSpan(patString name, patULong fr, patULong lr = patBadId) ;
  bioIteratorSpan intersection(bioIteratorSpan another) ;
  patString name ;
  patULong firstRow ;
  patULong lastRow ;
  } ;

ostream& operator<<(ostream &str, const bioIteratorSpan& x) ;

#endif
