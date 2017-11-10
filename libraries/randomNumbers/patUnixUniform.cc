//-*-c++-*------------------------------------------------------------
//
// File name : patUnixUniform.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Thu Mar  6 16:21:25 2003
//
//--------------------------------------------------------------------

#include <cstdlib>
#include <ctime> 
#include "patAbsTime.h"
#include "patUnixUniform.h"

patUnixUniform::patUnixUniform() {
  seed = (unsigned)time(0) ;
  srand(seed) ;
}


patUnixUniform::patUnixUniform(unsigned long s){
  if (s == 0) {
    patAbsTime now ;
    now.setTimeOfDay() ;
    seed = now.getUnixFormat() ;
  } else {
    seed = s ;
  }
  srand(seed) ;
}
void patUnixUniform::setSeed(unsigned long s) {
  seed = s ;
  srand(seed) ;
}
unsigned long patUnixUniform::getSeed() {
  return seed ;
}
patReal patUnixUniform::getUniform(patError*& err) {
  return rand() / (patReal(RAND_MAX+1.0)) ;
}

patString patUnixUniform::getType() const {
  return patString("Unix rand()") ;
}
