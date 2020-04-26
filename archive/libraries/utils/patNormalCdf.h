//-*-c++-*------------------------------------------------------------
//
// File name : patNormalCdf.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sat Jun  4 18:16:18 2005
//
//--------------------------------------------------------------------

#ifndef patNormalCdf_h
#define patNormalCdf_h

/**
   @doc This singleton is designed to compute the Cumulative Distribution Function of a normalized normal variable 
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL
   (Sat Jun  4 18:16:18 2005)
*/

#include "patType.h"
#include "patError.h"
#include "patVariables.h"

class patNormalCdf {

  friend class patSingletonFactory ;
 public: 
  static patNormalCdf* the() ;
  patReal compute(patReal gdx, patError*& err) const ;
  patReal derivative(patReal x, patError*& err) const ;
 private:
  patNormalCdf() ;
  patReal gammp(patReal a, patReal x, patError*& err) const ;
  patReal gser(patReal a, patReal x, patError*& err) const ;
  patReal gcf(patReal a, patReal x, patError*& err) const ;
  patReal gammln(patReal xx) const ; 
  patReal pi ;
  patReal oneDivSqrtTwoPi ;
  unsigned short itmax ;
  patReal eps ;
  patReal fpmin ;
  patVariables cof ;
  patReal stp ;
};

#endif
