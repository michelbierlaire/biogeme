//-*-c++-*------------------------------------------------------------
//
// File name : patDataFile.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Fri Dec  5 14:29:44 2003
//
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patDataFile.h"
#include "patOutputFiles.h"
#include "patDataFileObsIterator.h"
#include "patDataFileAggObsIterator.h"
#include "patDataFileIndIterator.h"
#include "patModelSpec.h"

patDataFile::patDataFile()  :
  typicalIndividual(),
  fileName(patParameters::the()->getgevBinaryDataFile()),
  outFile(fileName.c_str(),ios::out|ios::binary),
  size(0),
  obsIter(NULL),
  aggObsIter(NULL),
  indIter(NULL)
 {
  DEBUG_MESSAGE("Create data file") ;
  erase() ;

 }

void patDataFile::push_back(patIndividualData* data){
  if (data == NULL) {
    return ;
  }

  static patBoolean first = patTRUE ;
  if (first) {
    first = patFALSE ;
  }

  data->writeBinary(outFile) ;
  
   ++size ;
}

patDataFile::~patDataFile() {
  outFile.close() ;
}

unsigned long patDataFile::getSize() {
  return size ;
}

void patDataFile::erase() {
  outFile.close() ;
  outFile.open(fileName.c_str(),ios::out|ios::binary) ;
  size = 0 ;
}

patIterator<patObservationData*>* patDataFile::createObsIterator() {
  outFile.seekp(0) ;
  if (obsIter == NULL) {
    obsIter = new patDataFileObsIterator(this) ;
  }
  return obsIter ;
}

void patDataFile::releaseObsIterator() {
  DELETE_PTR(obsIter) ;
}

vector<patIterator<patObservationData*>*> 
patDataFile::createObsIteratorThread(unsigned int nbrThreads, 
				     patError*& err) {
  err = new patErrMiscError("Feature not yet implemented") ;
  WARNING(err->describe()) ;
  return vector<patIterator<patObservationData*>*>() ;
}

void patDataFile::releaseObsIteratorThread() {
  return ;
}

vector<patIterator<patIndividualData*>*> 
patDataFile::createIndIteratorThread(unsigned int nbrThreads, 
				     patError*& err) {
  err = new patErrMiscError("Feature not yet implemented") ;
  WARNING(err->describe()) ;
  return vector<patIterator<patIndividualData*>*>() ;
}

void patDataFile::releaseIndIteratorThread() {
  return ;
}

patIterator<patAggregateObservationData*>* patDataFile::createAggObsIterator() {
  outFile.seekp(0) ;
  if (aggObsIter == NULL) {
    aggObsIter = new patDataFileAggObsIterator(this) ;
  }
  return aggObsIter ;
}

void patDataFile::releaseAggObsIterator() {
  DELETE_PTR(aggObsIter) ;
}


patIterator<patIndividualData*>* patDataFile::createIndIterator() {
  outFile.seekp(0) ;
  if (indIter == NULL) {
    indIter = new patDataFileIndIterator(fileName,typicalIndividual) ;
  }
  return indIter ;
}

void patDataFile::releaseIndIterator() {
  DELETE_PTR(indIter) ;
}
void patDataFile::reserveMemory(unsigned long size, 
				patIndividualData* instance) {
  if (instance == NULL) {
    return ;
  }
  typicalIndividual = *instance ;
}

void patDataFile::finalize() {
  DEBUG_MESSAGE("Close file") ;
  outFile.close() ;
  patOutputFiles::the()->addDebugFile(fileName,"Data file in binary format");
}
