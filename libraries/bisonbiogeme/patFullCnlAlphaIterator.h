//-*-c++-*------------------------------------------------------------
//
// File name : patFullCnlAlphaIterator.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Sun May 20 20:35:12 2001
//
//--------------------------------------------------------------------

#ifndef patFullCnlAlphaIterator_h
#define patFullCnlAlphaIterator_h


#include <map>
#include "patIterator.h"
#include "patCnlAlphaParameter.h"

/**
   @doc Iterator on a map<patString,patCnlAlphaParameter>
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed May 16 17:34:39 2001)
 */
class patFullCnlAlphaIterator: public patIterator<patCnlAlphaParameter> {
public:
  patFullCnlAlphaIterator(map<patString,patCnlAlphaParameter>* x) ;
  void first()  ;
  void next()  ;
  patBoolean isDone()  ;
  patCnlAlphaParameter currentItem()  ;
private:
  map<patString,patCnlAlphaParameter>* theMap ;
  map<patString,patCnlAlphaParameter>::const_iterator i ;

};
#endif 
