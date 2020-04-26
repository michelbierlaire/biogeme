//-*-c++-*------------------------------------------------------------
//
// File name : patError.cc
// Author :   Michel Bierlaire
// Date :     Mon Dec 21 14:29:50 1998
//
// Modification history:
//
// Date                     Author            Description
// ======                   ======            ============
//
//--------------------------------------------------------------------
//

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patError.h"

/// Def ctor: current time is set
patError::patError() {
  timeStamp.setTimeOfDay() ;
}

patAbsTime patError::time() const {
  return timeStamp ;
}

void patError::setComment(const string& c) {
  comment_ = c ;
}

void patError::addComment(const string& c) {
  comment_ += " " + c ;
}


ostream& operator<<(ostream& stream, patError& error) {
  stream << "Error detected at " << error.time() << endl
	 << error.describe() << endl ;
  return(stream) ;
}
