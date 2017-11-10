//-*-c++-*------------------------------------------------------------
//
// File name : patImportance.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Jun  2 22:54:20 2003
//
//--------------------------------------------------------------------

#ifndef patImportance_h
#define patImportance_h

#include "patType.h"

/**
   @doc Various levels of importance for the messages
 */
class patImportance {
  public:

  enum type {
    /**
     */
    patFATAL ,
    /**
     */
    patERROR ,
    /**
     */
    patWARNING ,
    /**
     */
    patGENERAL ,
    /**
     */
    patDETAILED,
    /**
     */
    patDEBUG
  } ;

  patImportance(type p) ;

  friend patBoolean operator<(const patImportance& i1, const patImportance& i2) ;
  friend patBoolean operator<=(const patImportance& i1, const patImportance& i2) ;

  type operator()() const ;
private:
  type theType ;
};

#endif
