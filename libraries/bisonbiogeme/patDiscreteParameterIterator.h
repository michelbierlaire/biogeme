//-*-c++-*------------------------------------------------------------
//
// File name : patDiscreteParameterIterator.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sun Dec  5 18:13:59 2004
//
//--------------------------------------------------------------------

#ifndef patDiscreteParameterIterator_h
#define patDiscreteParameterIterator_h

#include <vector>
#include "patIterator.h"

class patDiscreteParameter ;

/**
   @doc Iterator on discrete parameters
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Sun Dec  5 18:13:59 2004)
 */

class patDiscreteParameterIterator: public patIterator<patDiscreteParameter*> {

 public:
  /**
   */
  patDiscreteParameterIterator(vector<patDiscreteParameter>* dataPtr) ;

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
  patDiscreteParameter* currentItem()  ;
 private :
  vector<patDiscreteParameter>* theVector ;
  vector<patDiscreteParameter>::iterator i ;
};

#endif
