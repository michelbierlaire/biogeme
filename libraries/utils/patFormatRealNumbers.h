//-*-c++-*------------------------------------------------------------
//
// File name : patFormatRealNumbers.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Wed May 31 09:08:18 2006
//
//--------------------------------------------------------------------

#ifndef patFormatRealNumbers_h
#define patFormatRealNumbers_h

#include "patType.h"

class patFormatRealNumbers {

 public:
  patFormatRealNumbers() ;
  void setForceScientificNotation(patBoolean forceScientificNotation) ;
  void setDecimalDigitsTTests(int d) ;
  void setDecimalDigitsStats(int d) ;
  patString format(patBoolean scientific,
		   patBoolean significant, 
		   unsigned short digits,
		   patReal number) ;

  patString formatParameters(patReal number) ;
  patString formatTTests(patReal number) ;
  patString formatStats(patReal number) ;
private:
  patBoolean forceScientificNotation ;
  int decimalDigitsTTest ;
  int significantDigitsParameters ;
  int decimalDigitsStats ;
};


#endif
