//-*-c++-*------------------------------------------------------------
//
// File name : trNonLinearAlgo.cc
// Author :    Michel Bierlaire
// Date :      Fri Aug 31 11:50:10 2001
//
//--------------------------------------------------------------------

#include "trNonLinearAlgo.h"
#include "patNonLinearProblem.h"
#include "patDisplay.h"
#include "patIterationBackup.h"

trNonLinearAlgo::trNonLinearAlgo(patNonLinearProblem* aProblem) :
  theProblem(aProblem), theBackup(NULL) {
  if (aProblem != NULL) {
    //    DETAILED_MESSAGE("Create algorithm to solve " << aProblem->getProblemName()) ;
  } 
  else {
    WARNING("No problem specified") ;
  }
}

trNonLinearAlgo::~trNonLinearAlgo() {

}

void trNonLinearAlgo::setBackup(patIterationBackup* aBackup) {
  theBackup = aBackup ;
}

patBoolean trNonLinearAlgo::isAvailable() const {
  return patTRUE ;
}
