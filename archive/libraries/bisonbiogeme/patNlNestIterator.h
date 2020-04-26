//-*-c++-*------------------------------------------------------------
//
// File name : patNlNestIterator.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed May 16 16:19:49 2001
//
//--------------------------------------------------------------------

#ifndef patNlNestIterator_h
#define patNlNestIterator_h


#include <map>
#include "patIterator.h"
#include "patNlNestDefinition.h"

/**
   @doc Iterator on a map<patString,patNlNestDefinition>
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed May 16 16:19:49 2001)
 */
class patNlNestIterator: public patIterator<patBetaLikeParameter> {
public:
  patNlNestIterator(map<patString,patNlNestDefinition>* x) ;
  void first()  ;
  void next()  ;
  patBoolean isDone()  ;
  patBetaLikeParameter currentItem()  ;
private:
  map<patString,patNlNestDefinition>* theMap ;
  map<patString,patNlNestDefinition>::const_iterator i ;

};
#endif 
