//-*-c++-*------------------------------------------------------------
//
// File name : patError.h
// Author :   \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :     Mon Dec 21 14:38:39 1998
//
//--------------------------------------------------------------------
//
#ifndef patError_h
#define patError_h

#include <string>
#include "patConst.h"
#include "patAbsTime.h"

/**
@doc Defines an abstract interface for error handling.
@author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
 */

class patError {
public:
  /**
   */
  friend ostream& operator<<(ostream& stream, patError& error) ;
  /**
   */
  patError() ;
  /**
   */
  virtual ~patError() {} ;
  /**
     Purely virtual. Provides a description of the error
   */
  virtual string describe() = PURE_VIRTUAL ;
  /** Purely virtual. Tries to repair the error. If it fails to do so, it
      will return patFALSE. Included in the first design, but never implemented.
  */
  virtual patBoolean tryToRepair() = PURE_VIRTUAL ;

  /** Provides the time when the object has been
      created. Especially useful in a log file.*/

  patAbsTime time() const ;
  /** A specific comment can be defined. For example, a description of the
      file name and the line number where the error occurred.*/
  void setComment(const string& c) ;

  /** The comment can be updated */
  void addComment(const string& c) ;

protected:
  string comment_ ;
private:
  patAbsTime timeStamp ;

};

#endif /* patError_h */
