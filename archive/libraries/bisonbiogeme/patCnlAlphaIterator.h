//-*-c++-*------------------------------------------------------------
//
// File name : patCnlAlphaIterator.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed May 16 17:34:39 2001
//
//--------------------------------------------------------------------

#ifndef patCnlAlphaIterator_h
#define patCnlAlphaIterator_h


#include <map>
#include "patIterator.h"
#include "patCnlAlphaParameter.h"

/**
   @doc Iterator on a map<patString,patCnlAlphaParameter>
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed May 16 17:34:39 2001)
 */
class patCnlAlphaIterator: public patIterator<patBetaLikeParameter> {
public:
  patCnlAlphaIterator(map<patString,patCnlAlphaParameter>* x) ;
  void first()  ;
  void next()  ;
  patBoolean isDone()  ;
  patBetaLikeParameter currentItem()  ;
private:
  map<patString,patCnlAlphaParameter>* theMap ;
  map<patString,patCnlAlphaParameter>::const_iterator i ;

};
#endif 
