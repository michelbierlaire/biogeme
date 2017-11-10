//----------------------------------------------------------------
// File: patAngle.cc
// Author: Michel Bierlaire
// Creation: Mon Jun  8 22:37:25 2009
//----------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <iostream>
#include "patAngle.h"
#include "patConst.h"
#include "patMath.h"

patAngle::patAngle(patReal a) : angleInRadians(a) {
    pi = 4.0 * atan(1.0) ;

}

patReal patAngle::getAngleInRadians() const {
  return angleInRadians ;
}

patReal patAngle::getAngleInDegrees() const {
  return angleInRadians * 180 / pi ;
}

void patAngle::setAngleInDegree(patReal degree) {
  angleInRadians = degree * pi / 180.0 ;
}

ostream& operator<<(ostream &str, const patAngle& x) {
  str << x.getAngleInDegrees() 
      << " deg. (" 
      << x.getAngleInRadians() 
      << " rad.)" ;
  return str ;
}

patBoolean operator<(const patAngle& a1, const patAngle& a2) {
  return (a1.angleInRadians < a2.angleInRadians) ;
}

patAngle patAngle::operator-() {
  patAngle theResult(-angleInRadians) ;
  return theResult ;
}
