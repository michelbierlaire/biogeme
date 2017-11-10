//-*-c++-*------------------------------------------------------------
//
// File name : patIndividualData.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Tue Mar 30 08:41:47 2004
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patDisplay.h"
#include "patIndividualData.h"
#include "patObservationData.h"

void patIndividualData::writeBinary(ostream& aStream) {
  unsigned long nObs = theObservations.size() ;
  aStream.write((char*)&nObs,sizeof(nObs)) ;
  for (vector<patObservationData>::iterator i = theObservations.begin() ;
       i != theObservations.end() ;
       ++i) {
    i->writeBinary(aStream) ;
  }
  unsigned long nAggObs = theAggregateObservations.size() ;
  aStream.write((char*)&nAggObs,sizeof(nAggObs)) ;
  for (vector<patAggregateObservationData>::iterator i = theAggregateObservations.begin() ;
       i != theAggregateObservations.end() ;
       ++i) {
    i->writeBinary(aStream) ;
  }

}
  
void patIndividualData::readBinary(istream& aStream) {
  unsigned long nObs ;
  theObservations.erase(theObservations.begin(),theObservations.end()) ;
  aStream.read((char *)&nObs,sizeof(nObs)) ;
  if (aStream.eof()) {
    return ;
  }
  patObservationData theObs ;
  for (unsigned long i = 0 ;
       i < nObs ;
       ++i) {
    theObs.readBinary(aStream) ;
    theObservations.push_back(theObs);
  }

  unsigned long nAggObs ;
  theAggregateObservations.erase(theAggregateObservations.begin(),theAggregateObservations.end()) ;
  aStream.read((char *)&nAggObs,sizeof(nAggObs)) ;
  if (aStream.eof()) {
    return ;
  }
  patAggregateObservationData theAggObs ;
  for (unsigned long i = 0 ;
       i < nAggObs ;
       ++i) {
    theAggObs.readBinary(aStream) ;
    theAggregateObservations.push_back(theAggObs);
  }


  

}

ostream& operator<<(ostream &str, const patIndividualData& x) {
  str << "Individual #" << x.panelId << endl ;
  for (vector<patObservationData>::const_iterator i = x.theObservations.begin() ;
       i != x.theObservations.end() ;
       ++i) {
    str << *i << endl ;
  }
  for (vector<patAggregateObservationData>::const_iterator i = x.theAggregateObservations.begin() ;
       i != x.theAggregateObservations.end() ;
       ++i) {
    str << *i << endl ;
  }
  return str ;
}

patReal patIndividualData::getWeight() {
  if (theObservations.empty()) {
    if (theAggregateObservations.empty()) {
      return 0.0 ;
    }
    else {
      return theAggregateObservations[0].getWeight() ;
    }
  }
  return theObservations[0].weight ;
}

patReal patIndividualData::getGroup() {
  if (theObservations.empty()) {
    if (theAggregateObservations.empty()) {
      return 0.0 ;
    }
    else {
      return theAggregateObservations[0].getGroup() ;
    }
  }
  return theObservations[0].group ;
}

patReal patIndividualData::getUnifDrawsForSnpPolynomial(unsigned long drawNumber, unsigned short term) {
  if (!theObservations.empty()) {
    return theObservations[0].unifDrawsForSnpPolynomial[drawNumber-1][term] ;
  }
  if (!theAggregateObservations.empty()) {
    if (!theAggregateObservations[0].theObservations.empty()) {
      return theAggregateObservations[0].theObservations[0].unifDrawsForSnpPolynomial[drawNumber-1][term] ;
    }
  }
  WARNING("ERROR---") ;
  return patReal() ;
}
