//-*-c++-*------------------------------------------------------------
//
// File name : patUtilFunction.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Sat Aug  2 14:15:08 2003
//
//--------------------------------------------------------------------

#ifndef patUtilFunction_h
#define patUtilFunction_h

class patArithRandom ;

#include <list>

/**
 */
struct patUtilTerm{

  /**
   */
  patUtilTerm() : randomParameter(NULL), massAtZero(patFALSE) {} ;

  /**
     mean of the random parameters, or name of the determ. parameter
   */
  patString beta ;
  /**
   */
  unsigned long betaIndex ;
  /**
     name of the attribute
   */
  patString x ;
  /**     
   */
  unsigned long xIndex ;
  /**
   */
  unsigned long rndIndex ;
  /**
   */
  patBoolean random ;
  /**
   */
  patArithRandom* randomParameter ;
  /**
   */
  patBoolean massAtZero ;

}   ;

/**
 */
typedef list<patUtilTerm> patUtilFunction ;


/**
 */
ostream& operator<<(ostream &str, const patUtilFunction& x) ;


#endif
