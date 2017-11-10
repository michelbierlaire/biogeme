//-*-c++-*------------------------------------------------------------
//
// File name : patTimer.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Aug 25 22:34:06 2003
//
//--------------------------------------------------------------------

#ifndef patTimer_h
#define patTimer_h

#include <time.h>
#include <map>
#include "patString.h"
#include "patType.h"

/**
   Compute cumulative times for given procedures
 */

class patTimer {

  friend class patSingletonFactory ;
public:
  static patTimer* the() ;
  
  void tic(unsigned long i) ;
  void toc(unsigned long i) ;
  void resetTime(unsigned long i,patString name) ;
  void generateReport() ;
  
private:
  patTimer() ;
  vector<patReal> globalProcedureTime ;
  vector<clock_t> startProcedureTime ;
  vector<clock_t> stopProcedureTime ;
  vector<unsigned long> ncalls ;
  vector<patString> names ;
  unsigned long number ;
  
};

#endif
