//-*-c++-*------------------------------------------------------------
//
// File name : patRandomDraws.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Tue Mar  4 20:58:33 2003
//
//--------------------------------------------------------------------

#ifndef patRandomDraws_h
#define patRandomDraws_h

#include "patConst.h"
#include "patString.h"

class patError ;

class patRandomDraws {

 public:
  virtual patReal getDraw(const patString& drawName, 
			  unsigned long drawNumber,
			  unsigned long individual,
			  patError*& err) 
    = PURE_VIRTUAL ;

};

#endif
