//-*-c++-*------------------------------------------------------------
//
// File name : bioIterationBackup.h
// Author:     Michel Bierlaire, EPFL
// Date  :     Thu Oct 28 11:48:03 2010
//
//--------------------------------------------------------------------

#ifndef bioIterationBackup_h
#define bioIterationBackup_h

#include "patIterationBackup.h"

/* Interface to backup the current iterate of the optimization process. Abstract object.
 */

class bioIterationBackup: public patIterationBackup {
 public:
  void saveCurrentIteration() ;

};

#endif
