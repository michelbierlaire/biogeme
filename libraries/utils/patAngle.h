//----------------------------------------------------------------
// File: patAngle.h
// Author: Michel Bierlaire
// Creation: Mon Jun  8 22:33:42 2009
//----------------------------------------------------------------

#ifndef patAngle_h
#define patAngle_h

#include "patType.h" 

class patAngle {
  friend ostream& operator<<(ostream &str, const patAngle& x) ;
  friend patBoolean operator<(const patAngle& a1, const patAngle& a2) ;
 public:
  patAngle operator-() ;
  patAngle(patReal angleInRadians = 0) ;
  patReal getAngleInRadians() const ;
  patReal getAngleInDegrees() const ;
  void setAngleInDegree(patReal degree) ;
 private:
  patReal pi ;
  patReal angleInRadians ;
};


#endif
