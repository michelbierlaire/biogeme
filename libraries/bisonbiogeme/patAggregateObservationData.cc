//-*-c++-*------------------------------------------------------------
//
// File name : patAggregateObservationData.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sun Jan 15 10:06:50 2006
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patAggregateObservationData.h"


void patAggregateObservationData::writeBinary(ostream& aStream) {
  unsigned long nObs = theObservations.size() ;
  aStream.write((char*)&nObs,sizeof(nObs)) ;
  for (vector<patObservationData>::iterator i =
	 theObservations.begin() ;
       i != theObservations.end() ;
       ++i) {
    i->writeBinary(aStream) ;
  }
}
  
void patAggregateObservationData::readBinary(istream& aStream) {
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
    theObservations.push_back(theObs) ;
  }
}

patReal patAggregateObservationData::getWeight() {
  if (theObservations.empty()) {
    return 0.0 ;
  }
  return theObservations[0].weight ;
}

patReal patAggregateObservationData::getGroup() {
  if (theObservations.empty()) {
    return 0.0 ;
  }
  return theObservations[0].group ;
}

ostream& operator<<(ostream &str, const patAggregateObservationData& x) {

  str << "--------------------------------------------" << endl ;
  str << "New aggregate observation: " << endl ;
  for (vector<patObservationData>::const_iterator i = x.theObservations.begin() ;
       i != x.theObservations.end() ;
       ++i) {
    str << *i << endl ;
  }
  return str ;
}

unsigned long patAggregateObservationData::getSize() const {
  return theObservations.size();
}
