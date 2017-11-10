//-*-c++-*------------------------------------------------------------
//
// File name : patLu.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Thu Jun 16 09:13:36 2005
//
//--------------------------------------------------------------------

#ifndef patLu_h
#define patLu_h

#include "patError.h"
#include "patVariables.h"
class patMyMatrix ;

class patLu {

 public:
  patLu(patMyMatrix* aMat) ;
  void computeLu(patError*& err) ;
  const vector<unsigned long>* getPermutation() const ;
  const patVariables* solve(const patVariables& b,patError*& err) ;
  patBoolean isSuccessful() const ; 
 private:
  patMyMatrix* theMatrix ;
  vector<unsigned long> index ;
  short interchanges ;
  patBoolean success ;
  patBoolean luPerformed ;
  unsigned long n ;
  patVariables solution ;
};

#endif
