//----------------------------------------------------------------
// File: patGeoCoordinates.cc
// Author: Michel Bierlaire
// Creation: Fri Oct 31 08:35:23 2008
//----------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include "patGeoCoordinates.h"
#include "patConst.h"

patGeoCoordinates::patGeoCoordinates(patReal lat, patReal lon) :
  latitudeInDegrees(lat), longitudeInDegrees(lon) {
    pi = 4.0 * atan(1.0) ;
  latitudeInRadians  = latitudeInDegrees * pi / 180.0 ;
  longitudeInRadians = longitudeInDegrees * pi / 180.0 ;
  
  latOrientation = (latitudeInDegrees  >= 0) ? 'N' : 'S' ;
  patReal tmp = (latitudeInDegrees  >= 0) ? latitudeInDegrees : -latitudeInDegrees ;
  latDegree = floor(tmp) ;
  tmp -= patReal(latDegree) ;
  tmp *= 60 ;
  latMinutes = floor(tmp) ;
  tmp -= patReal(latMinutes) ;
  tmp *= 60 ;
  latSeconds = tmp ;

  lonOrientation = (longitudeInDegrees >= 0) ? 'E' : 'W' ;
  tmp = (longitudeInDegrees  >= 0) ? longitudeInDegrees : -longitudeInDegrees ;
  lonDegree = floor(tmp) ;
  tmp -= patReal(lonDegree) ;
  tmp *= 60 ;
  lonMinutes = floor(tmp) ;
  tmp -= patReal(lonMinutes) ;
  tmp *= 60 ;
  lonSeconds = tmp ;
  
}

patReal patGeoCoordinates::distanceTo(const patGeoCoordinates& anotherPoint) {
  // Formula from http://mathforum.org/library/drmath/view/51711.html

  patReal earthRadius = 6372000.7976 ;

  patReal A = latitudeInRadians ;
  patReal B = longitudeInRadians ;
  patReal C = anotherPoint.latitudeInRadians ;
  patReal D = anotherPoint.longitudeInRadians ;

  if ((A == C) && (B == D)) {
    return 0 ;
  }
  patReal tmp = sin(A) * sin(C) + cos(A) * cos(C) * cos(B-D) ;
  if (tmp > 1.0) {
    return earthRadius * acos(1);
  }
  return earthRadius * acos(tmp) ;
}

patString patGeoCoordinates::getKML() const {
  stringstream str ;
  str << longitudeInDegrees << "," << latitudeInDegrees ;
  return patString(str.str()) ;
}

ostream& operator<<(ostream &str, const patGeoCoordinates& x) {
  str << x.latDegree << "d" << x.latMinutes << "'" << x.latSeconds << "\"" << x.latOrientation << "; " << x.lonDegree << "d" << x.lonMinutes << "'" << x.lonSeconds << "\"" << x.lonOrientation ;
  return str ;
}

