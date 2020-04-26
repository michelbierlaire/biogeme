//-*-c++-*------------------------------------------------------------
//
// File name : patKalman.h
// Author :   Michel Bierlaire
// Date :     Fri Jul 10 18:44:21 2015
//
//--------------------------------------------------------------------

#ifndef patKalman_h
#define patKalman_h

#include "patVariables.h"
#include "patError.h"
#include "patHybridMatrix.h"

class patKalman {

 public:
  patKalman() ;
  patKalman(patVariables dep, patVariables indep, patError*& err) ;
  patReal getCoefficient() const ;
  patReal getIntercept() const ;
  patReal evaluate(patReal x) const ;
  void addData(patVariables dep, patVariables indep, patError*& err) ;
  void addOneData(patReal dep, patReal indep,patError*& err) ;
 private:
  // This function is called after the filter has been updated
  void updateParameters(patVariables dep, patVariables indep, patError*& err) ;
  patReal coefficient ;
  patReal intercept ;
  patVariables dependent ;
  patVariables independent ;
  patHybridMatrix H ;
};

#endif
