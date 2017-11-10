//-*-c++-*------------------------------------------------------------
//
// File name : patLegendre.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Wed Nov 30 10:29:50 2005
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patMath.h"
#include "patLegendre.h"

patLegendre::patLegendre() :
  sqrt3(sqrt(3.0)),
  sqrt5(sqrt(5.0)),
  sqrt7(sqrt(7.0)),
  sqrt9(3.0),
  sqrt11(sqrt(11.0))
{
  
}

patLegendre::~patLegendre() {

}

patReal patLegendre::evaluate(unsigned short n, patReal x) {
  switch (n) {
  case 0:
    {
      return 1;
    }
  case 1:
    {
      return sqrt3 * (2.0 * x - 1) ;
    }
  case 2:
    {
      return sqrt5 * ( 6.0 * x * x - 6.0 * x + 1) ;
    }
  case 3:
    {
      return sqrt7 * (20.0 * x * x * x - 30.0 * x * x + 12.0 * x - 1.0) ;
    }
  case 4:
    {
      return sqrt9 * (70.0 * x * x * x * x 
		      - 140.0 * x * x * x 
		      + 90.0 * x * x 
		      - 20.0 * x 
		      + 1.0) ;
    }
  default:
    {
      patReal dn(n) ;
      patReal t1 = sqrt(4.0 * dn * dn - 1.0) / dn ;
      patReal t2 = (dn - 1.0) * sqrt(2.0 * dn + 1.0) / (dn * sqrt(2 * dn - 3.0)) ;
      patReal p1 = evaluate(n-1,x) ;
      patReal p2 = evaluate(n-2,x) ;
      return t1 * (2.0 * x - 1.0) * p1 - t2 * p2 ;
    }
  };
}

  /**
   */
  patVariables* derivative(unsigned short n, patReal x, patVariables* g) ;

