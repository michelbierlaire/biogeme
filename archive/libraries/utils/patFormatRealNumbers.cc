//-*-c++-*------------------------------------------------------------
//
// File name : patFormatRealNumbers.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Wed May 31 09:11:17 2006
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patFormatRealNumbers.h"
#include <iostream>
#include <iomanip>
#include <sstream>

#include "patMath.h"
#include "patPower.h"
#include "patConst.h"

patFormatRealNumbers::patFormatRealNumbers() :
  forceScientificNotation(patFALSE),
  decimalDigitsTTest(2),
  significantDigitsParameters(3),
  decimalDigitsStats(3) {

}
patString patFormatRealNumbers::format(patBoolean scientific,
				       patBoolean significant, 
				       unsigned short digits,
				       patReal number) {
  
  stringstream str ;
  // if (digits < 0) {
  //   str << number ;
  //   return patString(str.str()) ;
  // }
  if (scientific) {
    str << setiosflags(ios::scientific|ios::showpos) 
	<< setprecision(digits) 
	<< number ;
    return (str.str()) ;
   
  }
  if (significant) {
    str << setiosflags( ios::showpoint ) ; 
    str << setprecision(digits) << number ; 
    return patString(str.str()) ;
  }
  
  str << setiosflags( ios::fixed | ios::basefield | ios::showpoint ) ; 
  str << setprecision(digits)
      << number ;
  return (str.str()) ;
}


patString patFormatRealNumbers::formatParameters(patReal number) {
  return format(forceScientificNotation,
		patTRUE,
		significantDigitsParameters,
		number) ;
				    
}

patString patFormatRealNumbers::formatTTests(patReal number) {
  return format(forceScientificNotation,
		patFALSE,
		decimalDigitsTTest,
		number) ; 
}

patString patFormatRealNumbers::formatStats(patReal number) {
  return format (patFALSE,
		 patFALSE,
		 decimalDigitsStats,
		 number) ;
}
