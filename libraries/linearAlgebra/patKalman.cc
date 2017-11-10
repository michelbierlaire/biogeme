//-*-c++-*------------------------------------------------------------
//
// File name : patKalman.cc
// Author :   Michel Bierlaire
// Date :     Sun Jul 12 15:50:07 2015
//
//--------------------------------------------------------------------


#include "patErrMiscError.h"
#include "patKalman.h"
#include "patMyMatrix.h"
#include "patLu.h"

patKalman::patKalman() :
  coefficient(0.0),
  intercept(0.0),
  H(2,0.0) {
}
patKalman::patKalman(patVariables dep, 
		     patVariables indep, 
		     patError*& err) :
  coefficient(0.0),
  intercept(0.0),
  dependent(dep),
  independent(indep),
  H(2,0.0) {
  updateParameters(dep,indep,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}
patReal patKalman::getCoefficient() const {
  return coefficient ;
}
patReal patKalman::getIntercept() const {
  return intercept ;
}
patReal patKalman::evaluate(patReal x) const {
  return(coefficient * x + intercept) ;
}

void patKalman::updateParameters(patVariables dep, patVariables indep, patError*& err) {
  patULong m = dep.size() ;
  if (indep.size() != m) {
    stringstream str ; 
    str << "Incompatible sizes of data: " << indep.size() << " and " << m ;
    err = new patErrMiscError(str.str());
    WARNING(err->describe()) ;
    return ;
  }
  patReal Atb0(0.0) ;
  patReal Atb1(0.0) ;
  patReal AtA00(0.0) ;
  patReal AtA01(0.0) ;
  patReal AtA11(0.0) ;
  for (patULong i = 0 ; i < m ; ++i) {
    AtA00 += indep[i] * indep[i] ;
    AtA01 += indep[i] ;
    AtA11 += 1.0 ;
    Atb0 += indep[i] * dep[i] ;
    Atb1 += dep[i] ;
  }
  H.addElement(0,0,AtA00,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  H.addElement(0,1,AtA01,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  H.addElement(1,1,AtA11,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  patVariables rhs(2) ;
  rhs[0] = Atb0 - (AtA00 * coefficient + AtA01 * intercept) ;
  rhs[1] = Atb1 - (AtA01 * coefficient + AtA11 * intercept) ;
  patMyMatrix theH(H,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  patLu theLu(&theH) ;
  theLu.computeLu(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  const patVariables* result = theLu.solve(rhs,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  coefficient += (*result)[0] ;
  intercept += (*result)[1] ;
}

void patKalman::addData(patVariables dep, patVariables indep,patError*& err) {
  updateParameters(dep,indep,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}

void patKalman::addOneData(patReal dep, patReal indep, patError*& err) {
  patVariables vecdep(1,dep) ;
  patVariables vecindep(1,indep) ;
  addData(vecdep,vecindep,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
}

