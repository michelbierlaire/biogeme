//-*-c++-*------------------------------------------------------------
//
// File name : patIndividualData.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Mon Mar 29 18:08:44 2004
//
//--------------------------------------------------------------------

#ifndef patIndividualData_h
#define patIndividualData_h

#include "patObservationData.h"
#include "patAggregateObservationData.h"

/**
@doc Class for data related to one individual. In a panel data context, it may contain more than one observation
  @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Mon Mar 29 18:08:44 2004)
*/

class patIndividualData {

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
  patReal getUnifDrawsForSnpPolynomial(unsigned long drawNumber, unsigned short term) ;

  /**
   */
  vector<patObservationData> theObservations ;

  /**
   */
  vector<patAggregateObservationData> theAggregateObservations ;

  /**
   */
  unsigned long panelId ;
  /**
   */
  unsigned long nAlt ;
  /**
   */
  unsigned long nAttr ;
  /**
   */
  unsigned long nRandomTerms ; 
  /**
   */
  unsigned long nDraws ;

  
};

ostream& operator<<(ostream &str, const patIndividualData& x) ;

#endif
