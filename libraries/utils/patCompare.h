//-*-c++-*------------------------------------------------------------
//
// File name : patCompare.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Tue Jun 15 17:45:58 1999
//
//--------------------------------------------------------------------

#ifndef patCompare_h
#define patCompare_h

#include "patType.h"
#include "patMath.h"

/**
   @doc This class has one operator, comparing the absolute value of two real.
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Tue Jun 15 17:45:58 1999)
 */
class compAbsValue {
public :	
  /**
   */
  patBoolean operator()(patReal x, patReal y) {
    return(patAbs(x) < patAbs(y)) ;
  }
};
#endif
