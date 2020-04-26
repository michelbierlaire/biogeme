//-*-c++-*------------------------------------------------------------
//
// File name : patStatistics.h
// Author :    \URL[Michel Bierlaire]{http://people.epfl.ch/michel.bierlaire}
// Date :      Tue Jul 14 15:00:47 2015
//
//--------------------------------------------------------------------

#ifndef patStatistics_h
#define patStatistics_h

#include "patType.h"
#include <vector>

/**
   @doc Calculate and updates statistics on a large set of data with storing the
   @author \URL[Michel Bierlaire]{http://people.epfl.ch/michel.bierlaire}
 */
class patStatistics {

public:
  patStatistics() ;
  void addData(const vector<patReal>& d) ;
  void addData(patReal d) ;
  patULong getSize() const ;
  patReal getMean() const ;
  patReal getVariance() const ;
  patReal getStandardDeviation() const ;
private:
  patULong n ;
  patReal mean ;
  patReal sumOfSquares ;
  patReal variance ;
  patReal standardDeviation ;
  
};

#endif
