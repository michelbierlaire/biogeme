//-*-c++-*------------------------------------------------------------
//
// File name : bioSimulatedValues.h
// Author :    Michel Bierlaire
// Date :      Sat Nov 12 16:36:19 2016
//
//--------------------------------------------------------------------


#ifndef bioSimulatedValues_h
#define bioSimulatedValues_h

#include <map>
#include "patError.h"
#include "patInterval.h"


class bioSimulatedValues {

 public: 
  bioSimulatedValues(patULong dimension=0) ;
  void resize(patULong dim,patULong draws) ;
  void setNumberOfDraws(patULong draws) ;
  patULong getNumberOfValues() const ;
  patULong getNumberOfDraws() const ;

  patBoolean hasSimulatedValues() const ;

  patReal getTotal(patError*& err)  ;
  patInterval getTotalConfidenceInterval(patError*& err)  ;
  patReal getWeightedTotal(patError*& err)  ;
  patInterval getWeightedTotalConfidenceInterval(patError*& err)  ;
  patReal getAverage(patError*& err)  ;
  patInterval getAverageConfidenceInterval(patError*& err)  ;
  patReal getWeightedAverage(patError*& err)  ;
  patInterval getWeightedAverageConfidenceInterval(patError*& err)  ;
  patReal getNonZeroAverage(patError*& err)  ;
  patInterval getNonZeroAverageConfidenceInterval(patError*& err)  ;
  patReal getWeightedNonZeroAverage(patError*& err)  ;
  patInterval getWeightedNonZeroAverageConfidenceInterval(patError*& err)  ;
  patReal getMinimum(patError*& err) ;
  patInterval getMinimumConfidenceInterval(patError*& err) ;
  patReal getMaximum(patError*& err) ;
  patInterval getMaximumConfidenceInterval(patError*& err) ;

  patULong getNonZeros(patError*& err)  ;
  patReal getWeightedNonZeros(patError*& err)  ;
  patReal getTotalWeight(patError*& err)  ;
  void setWeights(vector<patReal>* w) ;
  void calculateConfidenceIntervals(patError*& err) ;
  void setNominalValue(patULong i, patReal v) ;
  patReal getNominalValue(patULong i) const ;
  void setSimulatedValue(patULong i, patULong r, patReal v) ;
  patInterval getConfidenceInterval(patULong i) const ;
private:
  void calculateAggregateValues(patError*& err) ;

 private:
  patInterval generateConfidenceInterval(vector<patReal>* v, patError*& err) ;
  vector<patReal> nominalValues ;
  vector<vector<patReal> > simulatedValues ;
  vector<patInterval> confidenceIntervals ;

  patReal alpha ;
  vector<patReal>* weights ;
  patBoolean confidenceIntervalsCalculated ;
  patBoolean aggregateValuesCalculated ;
  patULong nbrDraws ;
  map<patString,vector<patReal> > aggregateValues ;

  patReal nominalTotal ;
  patReal nominalWeightedTotal ;
  patULong nominalNonZeros ;
  patULong nominalWeightedNonZeros ;
  patReal nominalMinimum ;
  patReal nominalMaximum ;

  vector<patReal> total ;
  vector<patReal> weightedTotal ;
  vector<patReal> minimum ;
  vector<patReal> maximum ;

};

#endif
