//-*-c++-*------------------------------------------------------------
//
// File name : patAlternative.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Sun May 20 20:46:11 2001
//
//--------------------------------------------------------------------

#ifndef patAlternative_h
#define patAlternative_h

#include <list>
#include "patString.h"
#include "patType.h"

#include "patUtilFunction.h"

/**
   @doc Contains info related to alternatives
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed May 16 16:26:57 2001)
*/

struct patAlternative {
  /**
   */
  patString name ;
  /**
   */
  patUtilFunction utilityFunction ;

  /**
     ids must be consecutive between 0 and nAlt-1
   */
  unsigned long id ;
  /**
     User defined id. Must be non negative.
   */
  unsigned long userId ;

};


#endif
