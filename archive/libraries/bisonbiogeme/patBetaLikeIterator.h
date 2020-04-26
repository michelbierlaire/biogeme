//-*-c++-*------------------------------------------------------------
//
// File name : patBetaLikeIterator.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed May 16 16:05:40 2001
//
//--------------------------------------------------------------------

#ifndef patBetaLikeIterator_h
#define patBetaLikeIterator_h


#include <map>
#include "patIterator.h"
#include "patBetaLikeParameter.h"
#include "patString.h"


/**
   @doc Iterator on a map<patString,patBetaLikeParameter>
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed May 16 16:05:40 2001)
 */
class patBetaLikeIterator: public patIterator<patBetaLikeParameter> {
public:
  /**
   */
  patBetaLikeIterator(map<patString,patBetaLikeParameter>* x) ;
  /**
   */
  void first()  ;
  /**
   */
  void next()  ;
  /**
   */
  patBoolean isDone()  ;
  /**
   */
  patBetaLikeParameter currentItem()  ;
private:
  map<patString,patBetaLikeParameter>* theMap ;
  map<patString,patBetaLikeParameter>::const_iterator i ;

};
#endif 
