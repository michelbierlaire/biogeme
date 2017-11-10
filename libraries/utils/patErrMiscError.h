//-*-c++-*------------------------------------------------------------
//
// File name : patErrMiscError.h
// Author :   \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :     Mon Dec 21 14:39:40 1998
//
//--------------------------------------------------------------------
//

#ifndef patErrMiscError_h
#define patErrMiscError_h

#include "patError.h"

/**
   @doc Implements a miscellaneous error
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
 */

class patErrMiscError: public patError {
public:
  /**
     @param comment error description
   */
  patErrMiscError(const string& comment) ;

  /**
   */
  virtual ~patErrMiscError() {} ;
  /**
   */
  string describe() ;
  /**
   */
  patBoolean tryToRepair() ;
};

#endif /* patErrMiscError_h */
