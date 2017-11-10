//-*-c++-*------------------------------------------------------------
//
// File name : patAggregateObservationData.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sun Jan 15 09:49:49 2006
//
//--------------------------------------------------------------------

#ifndef patAggregateObservationData_h
#define patAggregateObservationData_h

#include "patObservationData.h"

/**
@doc Class for data related to latent choice. An aggregate observation contains several concrete potential observations, associated with weights.
  @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL ( Sun Jan 15 09:49:49 2006)
*/

class patAggregateObservationData {

 public:
  /**
   */
  void writeBinary(ostream& aStream) ;
  
  /**
   */
  void readBinary(istream& aStream) ;

  /**
     It is assumed that the weight is exactly the same for all observations
   */
  patReal getWeight() ;
  /**
     The group must be exactly the same for all observations
   */
  patReal getGroup() ;

  /**
   */
  unsigned long getSize() const ;

public:
  /**
   */
  vector<patObservationData> theObservations ;
  
};

ostream& operator<<(ostream &str, const patAggregateObservationData& x) ;

#endif
