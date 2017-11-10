//-*-c++-*------------------------------------------------------------
//
// File name : patLegendre.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Wed Nov 30 10:26:16 2005
//
//--------------------------------------------------------------------

#ifndef patLegendre_h
#define patLegendre_h


#include "patType.h"
#include "patVariables.h"

/**
@doc Implements the transformed Legendre polynomials as proposed by Bierens (2005) and used by Fosgerau and Bierlaire (2005).
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Wed Nov 30 10:26:16 2005)
*/

class patLegendre {

 public:
  /**
   */
  patLegendre() ;

  /**
   */
  ~patLegendre() ;

  /**
   */
  patReal evaluate(unsigned short n, patReal x) ;

  /**
   */
  patVariables* derivative(unsigned short n, patReal x, patVariables* g) ;

 private:
  patReal sqrt3 ;
  patReal sqrt5 ;
  patReal sqrt7 ;
  patReal sqrt9 ;
  patReal sqrt11 ;
};

#endif
