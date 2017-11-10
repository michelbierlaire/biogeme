//-*-c++-*------------------------------------------------------------
//
// File name : patCompareCorrelation.h
// Author :    Michel Bierlaire
// Date :      Wed Jul  4 16:14:10 2001
//
//--------------------------------------------------------------------

#ifndef patCompareCorrelation_h
#define patCompareCorrelation_h

#include <functional>
#include "patCorrelation.h"

/**
   @doc Function-object to compare two patCorrelation structures
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Jul  4 16:14:10 2001)
 */

struct patCompareCorrelation : 
  binary_function<patCorrelation,patCorrelation,patBoolean> {
  patBoolean operator()(const patCorrelation& c1,
			const patCorrelation& c2) const ;

};
#endif
