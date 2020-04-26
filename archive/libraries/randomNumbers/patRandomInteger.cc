//-*-c++-*------------------------------------------------------------
//
// File name : patRandomInteger.cc
// Author :    Michel Bierlaire
// Date :      Thu Aug 12 11:44:30 2010
//
//--------------------------------------------------------------------


#include "patRandomInteger.h"

patRandomInteger::patRandomInteger(patULong seed) {
  srand(seed) ;
}

patULong patRandomInteger::drawNumber(patULong minNumber, patULong maxNumber) {
  return (minNumber + rand() % (maxNumber - minNumber)) ;
}

