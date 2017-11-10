//-*-c++-*------------------------------------------------------------
//
// File name : bioSimulatedValues.cc
// Author :    Michel Bierlaire
// Date :      Sat Nov 12 16:42:38 2016
//
//--------------------------------------------------------------------

#include "bioSimulatedValues.h"
#include "patQuantiles.h"
#include "patErrMiscError.h"
#include "patErrOutOfRange.h"
#include "bioParameters.h"

bioSimulatedValues::bioSimulatedValues(patULong dimension) :
  nominalValues(dimension),simulatedValues(dimension), alpha(0.05),weights(NULL), confidenceIntervalsCalculated(patFALSE), aggregateValuesCalculated(patFALSE),nbrDraws(0) {
}

void bioSimulatedValues::resize(patULong dim,patULong draws) {
  nbrDraws = draws ;
  nominalValues.resize(dim) ;
  simulatedValues.resize(dim,vector<patReal>(nbrDraws)) ;
  confidenceIntervals.resize(dim) ;
  total.resize(nbrDraws) ;
  weightedTotal.resize(nbrDraws) ;
  minimum.resize(nbrDraws) ;
  maximum.resize(nbrDraws) ;
  confidenceIntervalsCalculated = patFALSE ;
  aggregateValuesCalculated = patFALSE ;
}
void bioSimulatedValues::setNumberOfDraws(patULong draws) {
  for (patULong i = 0 ; i < simulatedValues.size() ; ++i) {
    simulatedValues[i].resize(draws) ;
  }
  total.resize(nbrDraws) ;
  weightedTotal.resize(nbrDraws) ;
  minimum.resize(nbrDraws) ;
  maximum.resize(nbrDraws) ;
  nbrDraws = draws ;
  confidenceIntervalsCalculated = patFALSE ;
  aggregateValuesCalculated = patFALSE ;
}

patULong bioSimulatedValues::getNumberOfValues() const {
  return nominalValues.size() ;
}
patULong bioSimulatedValues::getNumberOfDraws() const {
  return nbrDraws ;
}

void bioSimulatedValues::calculateConfidenceIntervals(patError*& err) {
  if (confidenceIntervalsCalculated) {
    return ;
  }
  if (!hasSimulatedValues()) {
    return ;
  }
  patReal alpha = bioParameters::the()->getValueReal("sensitivityAnalysisAlpha",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  if (alpha < 0 || alpha > 1) {
    err = new patErrOutOfRange<patReal>(alpha,0,1) ;
    WARNING(err->describe()) ;
    return ;
  }
  if (alpha < 1.0-alpha) {
    alpha = 1.0 - alpha ;
  }
  
  
  patULong dim = getNumberOfValues() ;

  for (patULong i = 0 ; i < dim ; ++i) {
    patQuantiles theQuantiles(&simulatedValues[i]) ;
    patReal q1 = theQuantiles.getQuantile(alpha,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    patReal q2 = theQuantiles.getQuantile(1.0-alpha,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    if (q1 < q2) {
      confidenceIntervals[i].set(q1,q2) ;
    }
    else {
      confidenceIntervals[i].set(q2,q1) ;
    }
  }
  confidenceIntervalsCalculated = patTRUE ;
}

patBoolean bioSimulatedValues::hasSimulatedValues() const {
  return (nbrDraws) != 0 ;
}

void bioSimulatedValues::calculateAggregateValues(patError*& err) {
  if (aggregateValuesCalculated) {
    return ;
  }
  calculateConfidenceIntervals(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
  }


  if (weights != NULL) {
    if (weights->size() != getNumberOfValues()) {
      stringstream str ;
      str << "Incompatible number of weights: " << weights->size() << " instead of " << getNumberOfValues() ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
  }

  nominalTotal = 0.0 ;
  nominalWeightedTotal = 0.0  ;
  nominalNonZeros = 0 ;
  nominalWeightedNonZeros = 0.0 ;
  nominalMinimum = patMaxReal ;
  nominalMaximum = -patMaxReal ;
  
  for (patULong i = 0 ; i < getNumberOfValues() ; ++i) {
    patReal value = nominalValues[i] ;
    nominalTotal += value ;
    if (weights != NULL) {
      nominalWeightedTotal += (*weights)[i] * value ;
    }
    if (value != 0.0) {
      ++nominalNonZeros ;
      if (weights != NULL) {
	nominalWeightedNonZeros += (*weights)[i] ;
      }
    }
    if (value < nominalMinimum) {
      nominalMinimum = value ;
    }
    if (value > nominalMaximum) {
      nominalMaximum = value ;
    }
  }

  std::fill(total.begin(), total.end(), 0.0);
  std::fill(weightedTotal.begin(), weightedTotal.end(), 0.0);
  std::fill(minimum.begin(), minimum.end(), patMaxReal);
  std::fill(maximum.begin(), maximum.end(), -patMaxReal);

  for (patULong r = 0 ; r < nbrDraws ; ++r) {
    for (patULong i = 0 ; i < getNumberOfValues() ; ++i) {

      patReal value = simulatedValues[i][r] ;
      total[r] += value ;
      if (weights != NULL) {
	weightedTotal[r] += (*weights)[i] * value ;
      }
      if (value < minimum[r]) {
	minimum[r] = value ;
      }
      if (value > maximum[r]) {
	maximum[r] = value ;
      }
    }
  }

  aggregateValuesCalculated = patTRUE ;
}

patReal bioSimulatedValues::getTotal(patError*& err) {
  if (!aggregateValuesCalculated) {
    calculateAggregateValues(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
  }
  return nominalTotal ;
}

patInterval bioSimulatedValues::getTotalConfidenceInterval(patError*& err)  {
  patInterval res = generateConfidenceInterval(&total,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patInterval() ;
  }
  return res ;
}

patReal bioSimulatedValues::getWeightedTotal(patError*& err)  {
  if (!aggregateValuesCalculated) {
    calculateAggregateValues(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
  }
  return nominalWeightedTotal ;
}

patInterval bioSimulatedValues::getWeightedTotalConfidenceInterval(patError*& err)  {
  patInterval res = generateConfidenceInterval(&weightedTotal,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patInterval() ;
  }
  return res ;
}

patReal bioSimulatedValues::getAverage(patError*& err)  {
  if (!aggregateValuesCalculated) {
    calculateAggregateValues(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
  }
  return nominalTotal / patReal(getNumberOfValues()) ;
}

patInterval bioSimulatedValues::getAverageConfidenceInterval(patError*& err)  {
  if (!aggregateValuesCalculated) {
    calculateAggregateValues(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patInterval() ;
    }
  }
  patInterval result = getTotalConfidenceInterval(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patInterval() ;
  }
  return result / patReal(getNumberOfValues())  ;
}
patReal bioSimulatedValues::getWeightedAverage(patError*& err)  {
  if (!aggregateValuesCalculated) {
    calculateAggregateValues(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
  }
  return nominalWeightedTotal / patReal(getNumberOfValues()) ;
}

patInterval bioSimulatedValues::getWeightedAverageConfidenceInterval(patError*& err)  {
  if (!aggregateValuesCalculated) {
    calculateAggregateValues(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patInterval() ;
    }
  }
  patInterval result = getWeightedTotalConfidenceInterval(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patInterval() ;
  }
  return result / patReal(getNumberOfValues())  ;
}

patReal bioSimulatedValues::getNonZeroAverage(patError*& err)  {
  if (!aggregateValuesCalculated) {
    calculateAggregateValues(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
  }
  return nominalTotal / patReal(nominalNonZeros) ;

}
patInterval bioSimulatedValues::getNonZeroAverageConfidenceInterval(patError*& err)  {
  if (!aggregateValuesCalculated) {
    calculateAggregateValues(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patInterval() ;
    }
  }
  patInterval result = getTotalConfidenceInterval(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patInterval() ;
  }
  return result / patReal(nominalNonZeros)  ;

}
patReal bioSimulatedValues::getWeightedNonZeroAverage(patError*& err)  {
  if (!aggregateValuesCalculated) {
    calculateAggregateValues(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
  }
  return nominalWeightedTotal / patReal(nominalNonZeros) ;

}

patInterval bioSimulatedValues::getWeightedNonZeroAverageConfidenceInterval(patError*& err)  {
  if (!aggregateValuesCalculated) {
    calculateAggregateValues(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patInterval() ;
    }
  }
  patInterval result = getWeightedTotalConfidenceInterval(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patInterval() ;
  }
  return result / patReal(nominalNonZeros)  ;

}

patULong bioSimulatedValues::getNonZeros(patError*& err)  {
  if (!aggregateValuesCalculated) {
    calculateAggregateValues(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patULong() ;
    }
  }
  return nominalNonZeros ;
}

patReal bioSimulatedValues::getWeightedNonZeros(patError*& err) {
  if (!aggregateValuesCalculated) {
    calculateAggregateValues(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
  }
  return nominalWeightedNonZeros ;
}


void bioSimulatedValues::setWeights(vector<patReal>* w) {
  weights = w ;
  confidenceIntervalsCalculated = patFALSE ;
  aggregateValuesCalculated = patFALSE ;
}

void bioSimulatedValues::setNominalValue(patULong i, patReal v) {
  nominalValues[i] = v ;
  confidenceIntervalsCalculated = patFALSE ;
  aggregateValuesCalculated = patFALSE ;
}

void  bioSimulatedValues::setSimulatedValue(patULong i, patULong r, patReal v) {
  simulatedValues[i][r] = v ;
  confidenceIntervalsCalculated = patFALSE ;
  aggregateValuesCalculated = patFALSE ;
}

patReal bioSimulatedValues::getNominalValue(patULong i) const {
  return nominalValues[i] ;
}

patInterval bioSimulatedValues::getConfidenceInterval(patULong i) const {
  return confidenceIntervals[i] ;
}

patReal bioSimulatedValues::getMinimum(patError*& err) {
  if (!aggregateValuesCalculated) {
    calculateAggregateValues(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
  }
  return nominalMinimum ;
}
patInterval bioSimulatedValues::getMinimumConfidenceInterval(patError*& err) {
  patInterval res =  generateConfidenceInterval(&minimum,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patInterval() ;
  }
  return res ;
}
patReal bioSimulatedValues::getMaximum(patError*& err) {
  if (!aggregateValuesCalculated) {
    calculateAggregateValues(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
  }
  return nominalMaximum ;

}
patInterval bioSimulatedValues::getMaximumConfidenceInterval(patError*& err) {
  patInterval res =  generateConfidenceInterval(&maximum,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patInterval() ;
  }
  return res ;
}

patInterval bioSimulatedValues::generateConfidenceInterval(vector<patReal>* v, patError*& err) {
  if (!aggregateValuesCalculated) {
    calculateAggregateValues(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patInterval() ;
    }
  }

  if (alpha < 0 || alpha > 1) {
    err = new patErrOutOfRange<patReal>(alpha,0,1) ;
    WARNING(err->describe()) ;
    return patInterval() ;
  }
  patInterval result ;
  patQuantiles theQuantiles(v) ;
  patReal q1 = theQuantiles.getQuantile(alpha,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patInterval() ;
  }
  patReal q2 = theQuantiles.getQuantile(1.0-alpha,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patInterval() ;
  }
  if (q1 < q2) {
    result.set(q1,q2) ;
  }
  else {
    result.set(q2,q1) ;
  }
  return result ;

}
