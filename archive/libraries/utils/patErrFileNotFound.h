//-*-c++-*------------------------------------------------------------
//
// File name : patErrFileNotFound.h
// Author :   \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :     Mon Dec 21 14:40:06 1998
//
//--------------------------------------------------------------------
//

#ifndef patErrFileNotFound_h
#define patErrFileNotFound_h


#include "patError.h"

/** 
    @doc Concrete error class for 'File Not Found'
    @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi} (Mon Dec 21 14:40:06 1998)
*/
class patErrFileNotFound: public patError {
  public:
  /**
     @param fileName name of the file that is not found
   */
    patErrFileNotFound(const string& fileName) ;
  /**
   */
    string describe() ;
  /**
   */
    patBoolean tryToRepair() ;
  private:
    string file ;
};

#endif /* patErrFileNotFound_h */
