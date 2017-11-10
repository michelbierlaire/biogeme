//-*-c++-*------------------------------------------------------------
//
// File name : patFileExists.h
// Author :   \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :     Mon Dec 21 16:27:10 1998
//
//--------------------------------------------------------------------
//

#ifndef patFileExists_h
#define patFileExists_h

#include <string>
#include "patType.h"

/**
   @doc Encapsulates the function enabling to check if a file exists. This is designed to isolate operating system specific functions.
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Mon Dec 21 16:27:10 1998)
 */
class patFileExists {

public:
  /**
   */
  patBoolean operator()(const patString& fileName) ;
};

#endif
