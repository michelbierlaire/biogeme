//-*-c++-*------------------------------------------------------------
//
// File name : bioSplit.cc
// Author :    Michel Bierlaire
// Date :      Sun Apr  4 17:50:26 2010
//
//--------------------------------------------------------------------

#include "bioSplit.h"
#include "patErrOutOfRange.h"
#include "patMath.h"

ostream& operator<<(ostream &str, const bioSplit& x) {
  str << "(0," << x.length-1 << "):" ;
  for (vector<bioIteratorSpan >::const_iterator i = x.sublists.begin();
       i != x.sublists.end() ;
       ++i) {
    str << *i ;
  }
  return str ;
}

bioSplit::bioSplit(patULong n, patULong p):length(n),sublists(p) {
  patULong s = ceil(patReal(n)/patReal(p)) ;
  patULong b = 0 ; 
  for (patULong i = 0 ; i < p ; ++i) {
    patULong e = patMin(b + s , n) ;
    stringstream str ;
    str << "Thread " << i ;
    bioIteratorSpan sl(str.str(),b,e) ;
    sublists[i] = sl ;
    b = e ;
  }
}

bioIteratorSpan bioSplit::getSublist(patULong index, patError*& err) {
  if (index >= sublists.size()) {
    err = new patErrOutOfRange<patULong>(index,0,sublists.size()-1) ;
    WARNING(err->describe()) ;
    return bioIteratorSpan() ;
  }
  return sublists[index] ;
}
  

