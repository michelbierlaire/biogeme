//-*-c++-*------------------------------------------------------------
//
// File name : bioBayesianResults.h
// Author :    Michel Bierlaire
// Date :      Thu Aug  2 08:29:49 2012
//
//--------------------------------------------------------------------

#ifndef bioBayesianResults_h
#define bioBayesianResults_h

#include "patType.h"
#include "patHybridMatrix.h"
#include "patError.h"

class bioBayesianResults {

 public:
  bioBayesianResults() ;
  bioBayesianResults(const bioBayesianResults& b) ;
  const bioBayesianResults& operator=( const bioBayesianResults& rhs );
  bioBayesianResults(vector<vector<patReal> >* d,vector<patString> b) ;
  ~bioBayesianResults() ;
  patULong nDraws() const ;
  patULong nBetas() const ;
  void computeStatistics(patError*& err) ;

 private:
  vector<vector<patReal> >* theBayesianDraws ;
 public:
  vector<patString> paramNames ;
  patHybridMatrix* varCovar ;
  vector<patReal> mean ;

};

#endif
