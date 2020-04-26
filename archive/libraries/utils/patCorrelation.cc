//-*-c++-*------------------------------------------------------------
//
// File name : patCorrelation.cc
// Author :    Michel Bierlaire
// Date :      Sat Apr 30 10:53:42 2011
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patCorrelation.h"

ostream& operator<<(ostream &str, const patCorrelation& x) {
  str << "[" << x.firstCoef << "," << x.secondCoef << "]:" ;
  str << "Covar=" << x.covariance << "," ;
  str << "Corr=" << x.correlation << "," ;
  str << "t-test=" << x.ttest ;
  str << "Rob. Covar=" << x.robust_covariance << "," ;
  str << "Rob. Corr=" << x.robust_correlation << "," ;
  str << "Rob. t-test=" << x.robust_ttest ;
  return str ;
}
