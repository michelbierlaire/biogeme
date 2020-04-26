//-*-c++-*------------------------------------------------------------
//
// File name : patLoopTime.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Sun Sep  4 12:29:30 2005
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patLoopTime.h"
#include "patTimeInterval.h"

patLoopTime::patLoopTime(unsigned long noi) : numberOfIterations(noi) {
  startingTime.setTimeOfDay() ;
}
void patLoopTime::setIteration(unsigned long iter) {
  currentIter = iter ;
  currentTime.setTimeOfDay() ;
  patTimeInterval theInterval(startingTime,currentTime) ;
  patUnitTime diff = theInterval.getLengthInSeconds() ;
  terminationTime = startingTime ;
  if (iter > 0) {
    terminationTime += time_t(numberOfIterations * diff / iter) ;
  }
}

ostream& operator<<(ostream& stream, const patLoopTime& t) {
  //  stream << t.currentTime << "[" << t.currentIter << "/" << t.numberOfIterations << "] Estimated termination time: " << t.terminationTime.getTimeString(patTsfFULL) ;
  
  patTimeInterval ti(t.currentTime,t.terminationTime) ;
  stream << "[" << t.currentIter << "/" << t.numberOfIterations << "] " << "Est. term. time: " << t.terminationTime.getTimeString(patTsfFULL) << " (in " << ti.getLength() << ")" ;
  return stream ;
}

