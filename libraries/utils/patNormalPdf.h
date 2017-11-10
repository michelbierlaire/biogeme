//-*-c++-*------------------------------------------------------------
//
// File name : patNormalPdf.h
// Author :    \URL[Michel Bierlaire]{http://transp-or2.epfl.ch}
// Date :      Sun Dec 16 12:29:33 2007
//
//--------------------------------------------------------------------

#ifndef patNormalPdf_h
#define patNormalPdf_h

/**
   @doc This singleton is designed to compute the probability density function of a normalized normal variable 
   @author \URL[Michel Bierlaire]{http://transp-or2.epfl.ch}
   (Sun Dec 16 12:29:33 2007)
*/

#include "patType.h"

class patNormalPdf {
  
 public: 
  patNormalPdf() ;
  patReal operator()(patReal gdx) ;
 private:
  patReal pi ;
  patReal oneDivSqrtTwoPi ;
};

#endif
