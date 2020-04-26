//-*-c++-*------------------------------------------------------------
//
// File name : patNormalPdf.cc
// Author :    Michel Bierlaire
// Date :      Sun Dec 16 12:31:01 2007
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <cmath>
#include "patNormalPdf.h"

patNormalPdf::patNormalPdf(){
  pi = 4.0 * atan(1.0) ;
  oneDivSqrtTwoPi =  1.0/sqrt(2*pi) ;
}

patReal patNormalPdf::operator()(patReal gdx) {

  if (gdx > 40 || gdx < -40) return 0.0 ;
  return patReal(oneDivSqrtTwoPi * exp(- gdx * gdx / 2.0)) ;

}
