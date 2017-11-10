//-*-c++-*------------------------------------------------------------
//
// File name : patErrOutOfRange.h
// Author :   \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :     Mon Dec 21 14:38:55 1998
//
//--------------------------------------------------------------------
//

#ifndef patErrOutOfRange_h
#define patErrOutOfRange_h

#include "patError.h"
#include "patDisplay.h"
#include <sstream>

/** 
  @doc Concrete error class for 'Out of Range'
  @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi} (Mon Dec 21 14:38:55 1998)
*/
template <class T>
class patErrOutOfRange: public patError {
  public:
                                    // This error is produced when the value
                                    // is out of interval [lower,upper]
  /**
     @param value value producing the error
     @param lower lower bound of the interval
     @param upper upper bound of the interval
   */
    patErrOutOfRange(T value, 
		     T lower, 
		     T upper) {

      stringstream str ;
      str << "Value " 
	  << value 
	  << " out of range [" 
	  << lower 
	  << "," 
	  << upper 
	  << "]" << '\0' ;
      range = str.str() ;
    }
  /**
   */
  string describe() {
    string out = "Pattern: " ;
    out += range ;
    out += " " ;
    out += comment_ ;
    return out ;
  }
  /**
   */
  patBoolean tryToRepair() {
    WARNING("Sorry. I don't know how to repair that error.") ;
    return patFALSE ;
    
  }
  private:
    string range;
};

#endif /* patErrOutOfRange_h */
