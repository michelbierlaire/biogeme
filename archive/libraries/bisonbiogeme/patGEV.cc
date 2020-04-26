//-*-c++-*------------------------------------------------------------
//
// File name : patGEV.cc
// Author :    Michel Bierlaire
// Date :      Tue Jan 28 17:06:19 2003
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include "patDisplay.h"
#include "patMath.h"
#include "patErrMiscError.h"
#include "patGEV.h"

patGEV::~patGEV() {

}
patReal patGEV::getDerivative_xi_finDiff(unsigned long index,
					 const patVariables* x,
					 const patVariables* param,
					 const patReal* mu,
					 const vector<patBoolean>& available,
					 patError*& err) {

  if (!available[index]) {
    stringstream str ;
    str << "Alt. " << index << " is not available" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }

  patReal G = evaluate(x,param,mu,available,err) ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }

  patVariables xplus = *x ;
  patReal sqrteta = pow(patEPSILON, 0.5);
  patReal stepsize = sqrteta * patMax(patAbs(xplus[index]),patReal(1.0)) * patSgn(xplus[index]) ;
  xplus[index] += stepsize ;
  patReal Gplus = evaluate(&xplus,param,mu,available,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return ((Gplus - G)/stepsize) ;

}

patReal patGEV::getDerivative_mu_finDiff(const patVariables* x,
					 const patVariables* param,
					 const patReal* mu,
					 const vector<patBoolean>& available,
					 patError*& err) {

  patReal G = evaluate(x,param,mu,available,err) ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }

  patReal muplus = *mu ;
  patReal sqrteta = pow(patEPSILON, 0.5);
  patReal stepsize = sqrteta * patMax(patAbs(muplus),patReal(1.0)) * patSgn(muplus) ;
  muplus += stepsize ;
  patReal Gplus = evaluate(x,param,&muplus,available,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return ((Gplus - G)/stepsize) ;

}

patReal patGEV::getDerivative_param_finDiff(unsigned long index,
					    const patVariables* x,
					    const patVariables* param,
					    const patReal* mu, 
					    const vector<patBoolean>& available,
					    patError*& err) {
  
  patReal G = evaluate(x,param,mu,available,err) ;
  
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  
  patVariables xplus = *param ;
  patReal sqrteta = pow(patEPSILON, 0.5);
  patReal stepsize = sqrteta * patMax(patAbs(xplus[index]),patReal(1.0)) * patSgn(xplus[index]) ;
  xplus[index] += stepsize ;
  patReal Gplus = evaluate(x,&xplus,mu,available,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return ((Gplus - G)/stepsize) ;   
}


patReal patGEV::getSecondDerivative_xi_xj_finDiff(unsigned long index1,
						  unsigned long index2,
						  const patVariables* x,
						  const patVariables* param,
						  const patReal* mu,
						  const vector<patBoolean>& available,
						  patError*& err) {

  if (!available[index1]) {
    stringstream str ;
    str << "Alt. " << index1 << " is not available" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  if (!available[index2]) {
    stringstream str ;
    str << "Alt. " << index2 << " is not available" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }

  patReal G1 = getDerivative_xi(index1,x,param,mu,available,err) ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }

  patVariables xplus = *x ;
  patReal sqrteta = pow(patEPSILON, 0.5);
  patReal stepsize = sqrteta * patMax(patAbs(xplus[index2]),patReal(1.0)) * patSgn(xplus[index2]) ;
  xplus[index2] += stepsize ;
  patReal Gplus = getDerivative_xi(index1,&xplus,param,mu,available,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return ((Gplus - G1)/stepsize) ;

}


patReal patGEV::getSecondDerivative_xi_mu_finDiff(unsigned long index,
						  const patVariables* x,
						  const patVariables* param,
						  const patReal* mu,
						  const vector<patBoolean>& available,
						  patError*& err) {

  patReal G = getDerivative_xi(index,x,param,mu,available,err) ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }

  patReal muplus = *mu ;
  patReal sqrteta = pow(patEPSILON, 0.5);
  patReal stepsize = sqrteta * patMax(patAbs(muplus),patReal(1.0)) * patSgn(muplus) ;
  muplus += stepsize ;
  patReal Gplus = getDerivative_xi(index,x,param,&muplus,available,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return ((Gplus - G)/stepsize) ;



}

patReal patGEV::getSecondDerivative_param_finDiff(unsigned long indexVar,
						  unsigned long indexParam,
						  const patVariables* x,
						  const patVariables* param,
						  const patReal* mu, 
						  const vector<patBoolean>& available,
						  patError*& err) {

  patReal G = getDerivative_param(indexParam,x,param,mu,available,err) ;
  
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  
  patVariables xplus = *x ;
  patReal sqrteta = pow(patEPSILON, 0.5);
  patReal stepsize = sqrteta * patMax(patAbs(xplus[indexVar]),patReal(1.0)) * patSgn(xplus[indexVar]) ;
  xplus[indexVar] += stepsize ;
  patReal Gplus = getDerivative_param(indexParam,&xplus,param,mu,available,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return ((Gplus - G)/stepsize) ;   
}


