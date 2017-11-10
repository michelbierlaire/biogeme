//-*-c++-*------------------------------------------------------------
//
// File name : patCnlAlphaParameter.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed May 16 16:28:51 2001
//
//--------------------------------------------------------------------

#ifndef patCnlAlphaParameter_h
#define patCnlAlphaParameter_h

#include "patBetaLikeParameter.h"

/**
 */
struct patCnlAlphaParameter {
    
  /**
   */
  patBetaLikeParameter alpha ;
  /**
   */
  patString nestName ;
  /**
   */
  patString altName ;
};



#endif
