//-*-c++-*------------------------------------------------------------
//
// File name : bioIteratorSpan.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Thu Jul 16 22:32:08 2009
//
//--------------------------------------------------------------------

#include "bioIteratorSpan.h"
#include <iostream>
#include "patMath.h"

bioIteratorSpan::bioIteratorSpan() :
  firstRow(0) ,
  lastRow(patBadId) {

}

bioIteratorSpan::bioIteratorSpan(patString n, patULong fr, patULong lr) :
  name(n),
  firstRow(fr),
  lastRow(lr) {


}

ostream& operator<<(ostream &str, const bioIteratorSpan& x) {
  str << x.name <<"[" << x.firstRow << "->" << x.lastRow << "]" ;
  return str ;
}

bioIteratorSpan bioIteratorSpan::intersection(bioIteratorSpan another) {
  stringstream str ;
  str << "Intersection of " << name << " and " << another.name ;
  patULong fr = patMax(firstRow,another.firstRow) ;
  patULong lr = patMin(lastRow,another.lastRow) ;
  return (bioIteratorSpan(str.str(),fr,lr)) ;
}

