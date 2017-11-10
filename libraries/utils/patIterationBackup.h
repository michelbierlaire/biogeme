//-*-c++-*------------------------------------------------------------
//
// File name : patIterationBackup.h
// Author:     Michel Bierlaire, EPFL
// Date  :     Tue Jun 29 17:16:22 2010
//
//--------------------------------------------------------------------

#ifndef patIterationBackup_h
#define patIterationBackup_h

#include "patConst.h"

/* Interface to backup the current iterate of the optimization process. Abstract object.
 */

class patIterationBackup {
 public:
  virtual void saveCurrentIteration() = PURE_VIRTUAL ;

};

#endif
