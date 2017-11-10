//-*-c++-*------------------------------------------------------------
//
// File name : patInterval.h
// Author :    Michel Bierlaire
// Date :      Sun Nov 13 15:26:36 2016
//
//--------------------------------------------------------------------

#ifndef patInterval_h
#define patInterval_h

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "patType.h"

class patInterval {
  friend ostream& operator<<(ostream& stream, const patInterval& interval) ;
  friend patInterval operator*(patInterval i, patReal scale) ;
  friend patInterval operator/(patInterval i, patReal scale) ;
  friend patInterval operator*(patReal scale, patInterval i) ;
public:
  patInterval() ;
  patInterval(patReal l, patReal u) ;
  void set(patReal l, patReal u) ;
  patReal getLowerBound() const ;
  patReal getUpperBound() const ;
  patReal getLength() const;
  private:
  patReal lowerBound ;
  patReal upperBound ;
  
};

#endif 

