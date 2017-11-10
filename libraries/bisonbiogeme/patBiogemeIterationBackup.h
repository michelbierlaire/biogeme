//-*-c++-*------------------------------------------------------------
//
// File name : patBiogemeIterationBackup.h
// Author:     Michel Bierlaire, EPFL
// Date  :     Tue Jun 29 17:37:23 2010
//
//--------------------------------------------------------------------

#ifndef patBiogemeIterationBackup_h
#define patBiogemeIterationBackup_h

#include "patIterationBackup.h"

/* Interface to backup the current iterate of the optimization process. Abstract object.
 */

class patBiogemeIterationBackup: public patIterationBackup {
 public:
  void saveCurrentIteration() ;

};

#endif
