//-*-c++-*------------------------------------------------------------
//
// File name : patTimer.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Aug 25 22:37:12 2003
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <cassert>
#include <iomanip>
#include "patTimer.h"
#include "patDisplay.h" 
#include "patSingletonFactory.h"

patTimer::patTimer() :
  globalProcedureTime(100),
  startProcedureTime(100),
  stopProcedureTime(100),
  ncalls(100),
  names(100),
  number(0) {

}

patTimer* patTimer::the() {
  return patSingletonFactory::the()->patTimer_the() ;
}

void patTimer::tic(unsigned long procedure) {
  startProcedureTime[procedure] = clock() ;
  ncalls[procedure] += 1 ;
  
}

void patTimer::toc(unsigned long procedure) {
  stopProcedureTime[procedure] = clock() ;
  globalProcedureTime[procedure] += 
    1000.0 * (stopProcedureTime[procedure] - startProcedureTime[procedure]) / patReal(CLOCKS_PER_SEC) ;
}

void patTimer::resetTime(unsigned long procedure, patString name) {
  if (procedure > number) {
    number = procedure ;
  }
  globalProcedureTime[procedure] = 0.0 ;
  ncalls[procedure] = 0 ;
  names[procedure] = name ;
}

void patTimer::generateReport() {


  DEBUG_MESSAGE("Time\t#calls\tAvg\tProcedure") ;
  DEBUG_MESSAGE("++++\t++++++\t+++\t+++++++++") ;
  for (unsigned long i = 0 ; i <= number ; ++i) {
    DEBUG_MESSAGE(globalProcedureTime[i] << '\t' <<
		  ncalls[i] << '\t' <<
		  globalProcedureTime[i] / ncalls[i] << '\t' <<
		  names[i]) ;
  }
}
