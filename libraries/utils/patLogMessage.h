//-*-c++-*------------------------------------------------------------
//
// File name : patLogMessage.h
// Author :    Michel Bierlaire
// Date :      Wed Apr  6 10:21:58 2016
//
//--------------------------------------------------------------------

#ifndef patLogMessage_h
#define patLogMessage_h

#include "patConst.h"
#include "patString.h"

class patLogMessage {

 public:
  virtual void addLogMessage(patString m) = PURE_VIRTUAL ;

};
#endif
