//-*-c++-*------------------------------------------------------------
//
// File name : patErrNullPointer.h
// Author :   \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :     Mon Dec 21 14:39:16 1998
//
//--------------------------------------------------------------------
//

#ifndef patErrNullPointer_h
#define patErrNullPointer_h

#include "patError.h"

/**
   @doc Implements an error when a NULL pointer is detected
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi} (Mon Dec 21 14:39:16 1998)
 */
class patErrNullPointer: public patError {
public:
  /**
     @param type type of the pointer
   */
  patErrNullPointer(const string& type) ;
  /**
   */
  virtual ~patErrNullPointer() {} ;
  /**
   */
  string describe() ;
  /**
   */
  patBoolean tryToRepair() ;
private:
  string type ;
};

#endif /* patErrNullPointer_h */
