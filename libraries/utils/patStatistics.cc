//-*-c++-*------------------------------------------------------------
//
// File name : patStatistics.cc
// Author :    \URL[Michel Bierlaire]{http://people.epfl.ch/michel.bierlaire}
// Date :      Tue Jul 14 15:04:07 2015
//
//--------------------------------------------------------------------

#include "patStatistics.h"


patStatistics::patStatistics() :
  n(0), mean(0.0), sumOfSquares(0.0), variance(0.0), standardDeviation(0.0) {

}
patULong patStatistics::getSize() const {
  return n ;
}

patReal patStatistics::getMean() const {
  return mean ;
}

patReal patStatistics::getVariance() const {
  return variance ;
}

patReal patStatistics::getStandardDeviation() const {
  return standardDeviation ;
}


void patStatistics::addData(patReal d) {
  vector<patReal> vd(1,d) ;
  addData(vd) ;
}

void patStatistics::addData(const vector<patReal>& d) {

  patReal oldMean = mean ;
  patULong m = d.size() ;
  
  patReal sumData(0.0) ;
  for (vector<patReal>::const_iterator i = d.begin() ;
       i != d.end() ;
       ++i) {
    sumData += *i ;
  }
  mean = (n * oldMean + sumData) / patReal(n+m) ;

  patReal sumSquaresData(0.0) ;
  for (vector<patReal>::const_iterator i = d.begin() ;
       i != d.end() ;
       ++i) {
    sumSquaresData += (*i-mean)*(*i-mean) ;
  }
  sumOfSquares += n * (oldMean - mean) * (oldMean - mean) + sumSquaresData ;

  variance = sumOfSquares / patReal(n+m-1) ;
  standardDeviation = variance / patReal(n+m) ;
  n+= m ;

}

