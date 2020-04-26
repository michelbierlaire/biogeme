//-*-c++-*------------------------------------------------------------
//
// File name : patInterval.cc
// Author :    Michel Bierlaire
// Date :      Sun Nov 13 15:32:04 2016
//
//--------------------------------------------------------------------


#include "patInterval.h"
#include "patConst.h"

patInterval::patInterval():lowerBound(-patMaxReal),upperBound(patMaxReal) {

}

patInterval::patInterval(patReal l, patReal u) : lowerBound(l), upperBound(u) {

}

void patInterval::set(patReal l, patReal u) {
  lowerBound = l ;
  upperBound = u ;
}

patReal patInterval::getLowerBound() const {
  return lowerBound ;
}

patReal patInterval::getUpperBound() const {
  return upperBound ;
}

patReal patInterval::getLength() const {
  return upperBound-lowerBound ;
}

ostream& operator<<(ostream& stream, const patInterval& interval) {
  stream << "[" << interval.lowerBound << "," << interval.upperBound << "]" ;
  return stream ;
}


patInterval operator*(patInterval i, patReal scale) {
  i.lowerBound *= scale ;
  i.upperBound *= scale ;
  return i ;
}

patInterval operator/(patInterval i, patReal scale) {
  i.lowerBound /= scale ;
  i.upperBound /= scale ;
  return i ;
}

patInterval operator*(patReal scale, patInterval i) {
  i.lowerBound *= scale ;
  i.upperBound *= scale ;
  return i ;
}
