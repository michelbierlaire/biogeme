//-*-c++-*------------------------------------------------------------
//
// File name : patExecScript.h
// Author :    Michel Bierlaire
// Date :      Sat Apr  1 17:18:59 2017
//
//--------------------------------------------------------------------

#ifndef patExecScript_h
#define patExecScript_h

#ifdef HAVE_SYS_WAIT_H
#include "pstream.h"
#endif

#include <vector>
#include "patString.h"
#include "patType.h"

class patExecScript {

 public:
  patExecScript(vector<patString> c) ;
  void run() ;
  patString getOutput() ;
  patBoolean killAfterRun() const ;
private:
  // Contains the command and it arguments
  vector<patString> command ;
  stringstream output ;
  

};
#endif

