//-*-c++-*------------------------------------------------------------
//
// File name : patMessage.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Mon Jun  2 22:53:15 2003
//
//--------------------------------------------------------------------

#ifndef patMessage_h
#define patMessage_h

#include "patAbsTime.h"
#include "patImportance.h"

class patMessage {
  
public:
  patMessage() ;
  patMessage(const patMessage& msg) ;
  patString fullVersion() ;
  patString shortVersion() ;

  patAbsTime theTime ;
  patImportance theImportance ;
  patString text ;
  patString fileName ;
  patString lineNumber ;

};


#endif
