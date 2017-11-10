//-*-c++-*------------------------------------------------------------
//
// File name : patDataFileIndIterator.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Tue Mar 30 07:51:48 2004
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patDisplay.h"
#include "patDataFileIndIterator.h"
#include "patString.h"

patDataFileIndIterator::patDataFileIndIterator(patString aFile,
					       patIndividualData aRow) 
  : fileName(aFile),
    inFile(aFile.c_str(),ios::in|ios::binary),
    currentIndividual(aRow),
    error(patFALSE) {
  
}

patDataFileIndIterator::~patDataFileIndIterator() {
  inFile.close() ;
}


void patDataFileIndIterator::first() {

  counter = 0 ;
  error = patFALSE ;
  inFile.close() ;
  inFile.open(fileName.c_str(),ios::in|ios::binary) ;
  inFile.seekg(0,ios::end) ;
  inFile.seekg(0,ios::beg) ;
  read() ;
}

void patDataFileIndIterator::read() {
  ++counter ;
  currentIndividual.readBinary(inFile) ;
  
  if (!inFile.good() || inFile.bad() || inFile.eof()) {
    error = patTRUE ;
    return ;
  }

}

void patDataFileIndIterator::next() {
  read() ;
}

patBoolean patDataFileIndIterator::isDone() {
  if (!inFile.good() || inFile.bad() || inFile.eof()) {
    return patTRUE ;
  }
  if (error) {
    return patTRUE ;
  }
  return patFALSE ;
}

patIndividualData* patDataFileIndIterator::currentItem() {
  return &currentIndividual ;
}
