//-*-c++-*------------------------------------------------------------
//
// File name : patBiogemeScripting.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Tue Dec 13 15:48:16 2005
//
//--------------------------------------------------------------------

#ifndef patBiogemeScripting_h
#define patBiogemeScripting_h

class patFastBiogeme ;

#ifdef GIANLUCA
#include "patGianlucaBiogeme.h"
#else
#include "patBiogeme.h"
#endif

/**
   @doc This class is designed to be called from a scripting language
   like Python, Perl or Tcl, through a SWIG interface
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Tue Dec 13 15:48:16 2005)
*/

class patBiogemeScripting {
 public:
  patBiogemeScripting() ;

  // Interface called from main()
  void estimate(int argc, char *argv[]) ;
  void simulation(int argc, char *argv[]) ;
  void mod2py(int argc, char *argv[]) ;


 private:
  void invokeBiogeme() ;

 private:

  patBiogeme biogeme ;
};

#endif
