//-*-c++-*------------------------------------------------------------
//
// File name : bioSimulation.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Sat Feb  6 15:08:58 2010
//
//--------------------------------------------------------------------

#ifndef bioSimulation_h
#define bioSimulation_h

#include "patError.h"
class bioSample ;
class bioSimulation {

 public:
  virtual void bio__generateSimulatedValued(bioSample* sample, patError*& err) = PURE_VIRTUAL ;
  vector<pair<patString,patReal> >* getSimulation() ;
 protected:
  vector<pair<patString,patReal> > simulation ;
} ;


#endif
