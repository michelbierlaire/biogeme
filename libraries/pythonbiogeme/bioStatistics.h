//-*-c++-*------------------------------------------------------------
//
// File name : bioStatistics.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Sat Dec 19 19:12:12 2009
//
//--------------------------------------------------------------------

#ifndef bioStatistics_h
#define bioStatistics_h

#include "patError.h"
class bioSample ;
class bioStatistics {

 public:
  virtual ~bioStatistics() {} ;
  virtual void bio__generateStatistics(bioSample* sample, patError*& err) = PURE_VIRTUAL ;
  vector<pair<patString,patReal> >* getStatistics() ;
 protected:
  vector<pair<patString,patReal> > statistics ;
} ;

#endif
