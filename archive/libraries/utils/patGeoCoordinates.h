//----------------------------------------------------------------
// File: patGeoCoordinates.h
// Author: Michel Bierlaire
// Creation: Fri Oct 31 08:25:08 2008
//----------------------------------------------------------------

#ifndef patGeoCoordinates_h
#define patGeoCoordinates_h

#include <iostream>
#include "patType.h"

class patGeoCoordinates {

  friend ostream& operator<<(ostream& str, const patGeoCoordinates& x) ;

 public:
  /**
    Constructor
    @param lat latitude (in degrees)
    @param lon longitude (in degrees)
  */
  patGeoCoordinates(patReal lat, patReal lon) ;

  /**
     Compute the distance (in meters) between the current point and another point
   */
  patReal distanceTo(const patGeoCoordinates& anotherPoint) ;

  /**
   */
  patString getKML() const ;

 public:
  patReal latitudeInDegrees ;
  patReal longitudeInDegrees ;

 private:
  patReal pi ;
  patReal latitudeInRadians ;
  patReal longitudeInRadians ;
  short latDegree ;
  short latMinutes ;
  patReal latSeconds ;
  char latOrientation ;
  short lonDegree ;
  short lonMinutes ;
  patReal lonSeconds ;
  char lonOrientation ;
};


#endif
