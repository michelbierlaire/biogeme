//-*-c++-*------------------------------------------------------------
//
// File name : patCorrelation.h
// Author :    Michel Bierlaire
// Date :      Wed Jul  4 15:48:39 2001
//
//--------------------------------------------------------------------

#ifndef patCorrelation_h
#define patCorrelation_h

#include "patString.h"
#include "patType.h"

/**
   @doc Structure designed to store the covariance and correlation information
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Jul  4 15:48:39 2001)
*/

struct patCorrelation {
  /**
   */
  patString firstCoef ;

  /**
   */
  patString secondCoef ;

  /**
   */
  patReal covariance ;

  /**
   */
  patReal correlation ;

  /**
   */
  patReal ttest ;

  /**
   */
  patReal robust_covariance ;

  /**
   */
  patReal robust_correlation ;

  /**
   */
  patReal robust_ttest ;



};

ostream& operator<<(ostream &str, const patCorrelation& x) ;


#endif


