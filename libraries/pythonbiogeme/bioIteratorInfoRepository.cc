//-*-c++-*------------------------------------------------------------
//
// File name : bioIteratorInfoRepository.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Jul 14 13:25:19 2009
//
//--------------------------------------------------------------------

#include <sstream>
#include "patDisplay.h"
#include "bioIteratorInfoRepository.h"
#include "patErrMiscError.h"
#include "patErrOutOfRange.h"
#include "bioPythonSingletonFactory.h"

bioIteratorInfoRepository::bioIteratorInfoRepository(): topIterator("") {
}

bioIteratorInfoRepository* bioIteratorInfoRepository::the() {
  return bioPythonSingletonFactory::the()->bioIteratorInfoRepository_the() ;
}

void bioIteratorInfoRepository::addIteratorInfo(patString name, 
						bioIteratorInfo ii, 
						patError*& err) {
  //  DEBUG_MESSAGE("Add iterator info ["<< name <<"]: " <<  ii) ;
  ii.iteratorName = name ;
  pair<patString,bioIteratorInfo> p(name,ii) ;
  theRepository.insert(p) ;
  if (ii.isTopIterator() && !ii.isDrawIterator()) {
    if (topIterator != "") {
      stringstream str ;
      str << "There are (at least) two top iterators: " << topIterator << " and " << name ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return  ;
    }
    else {
      topIterator = name ;
    }
  }
}

const bioIteratorInfo* bioIteratorInfoRepository::getIteratorInfo(patString name,patError*& err ) const {


  map<patString,bioIteratorInfo>::const_iterator found = theRepository.find(name) ;
  if (found == theRepository.end()) {
    stringstream str ;
    str << "Iterator " << name << " is unknown" ;
    str << "Available iterators: " ;
    for (map<patString,bioIteratorInfo>::const_iterator iter = theRepository.begin() ;
	 iter != theRepository.end() ;
	 ++iter) {
      str << iter->first << " " ;
    }
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  return &(found->second) ;
}


patString bioIteratorInfoRepository::debug(patError*& err) {
  stringstream str ;
  str << "List of iterators" << endl ;
  for (map<patString,bioIteratorInfo>::iterator i = theRepository.begin() ;
       i != theRepository.end() ;
       ++i) {
    str << i->second << endl ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patString() ;
    }
  }
  return patString(str.str()) ;
}


//map<patString,bioIteratorInfo>::const_iterator 
//bioIteratorInfoRepository::getRepositoryIterator() const {
//  return theRepository.begin() ;
//}


void bioIteratorInfoRepository::addNewRowId(patString indexName, patULong rowNumber) {
  for (map<patString,bioIteratorInfo>::iterator i = theRepository.begin() ;
       i != theRepository.end() ;
       ++i) {
    if (i->second.indexName == indexName) {
      i->second.mapIds.push_back(rowNumber) ;
    }
  }
}

void bioIteratorInfoRepository::computeRowPointers(patULong sampleSize, 
					 patError*& err) {
  for (map<patString,bioIteratorInfo>::iterator i = theRepository.begin() ;
       i != theRepository.end() ;
       ++i) {
    i->second.computePointers(sampleSize,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }

  
  
}

patBoolean bioIteratorInfoRepository::isRowIterator(patString name, patError*& err) const {
  const bioIteratorInfo* theIter = getIteratorInfo(name,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patFALSE ;
  }
  return (theIter->type == ROW) ;
  
}

patBoolean bioIteratorInfoRepository::isMetaIterator(patString name, patError*& err) const {
  const bioIteratorInfo* theIter = getIteratorInfo(name,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patFALSE ;
  }
  return (theIter->type == META) ;
  
}

patBoolean bioIteratorInfoRepository::isTopIterator(patString name, patError*& err) const {
  const bioIteratorInfo* theIter = getIteratorInfo(name,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patFALSE ;
  }
  return (theIter->isTopIterator()) ;
  
}



bioIteratorSpan bioIteratorInfoRepository::getSublist(patULong index, patError*& err) {
  if (index >= subListsForThreads.size()) {
    err = new patErrOutOfRange<patULong>(index,0,subListsForThreads.size()-1) ;
    WARNING(err->describe()) ;
    return bioIteratorSpan() ;
  }
  return subListsForThreads[index] ;
}

patULong bioIteratorInfoRepository::generateSublistsForThreads(patULong dataSize, 
							   patULong numberOfThreads,
							   patError*& err) {

  // Rows spanned by the top iterator. The thread spans should
  // correspond with these values.

  bioIteratorInfo theTopIterator = theRepository[topIterator] ;

  if (!theTopIterator.isTopIterator()) {
    return 0;
  }
  vector<patULong> mapIds = theTopIterator.mapIds ;
  mapIds.push_back(dataSize) ;
  DEBUG_MESSAGE("Generate iterators for " << numberOfThreads << " threads") ;
  patULong approximateSizeOfEachBlock = ceil(patReal(dataSize)/patReal(numberOfThreads)) ;
  DEBUG_MESSAGE("Approximate size of each block: " << approximateSizeOfEachBlock) ;
  patULong currentThread = 1 ;
  patULong beginCurrentThread = 0 ;
  patULong sizeCurrentThread = 0 ;
  for (patULong i = 0 ; i < mapIds.size()-1 ; ++i) {
    sizeCurrentThread += mapIds[i+1]-mapIds[i] ;
    if (sizeCurrentThread >= approximateSizeOfEachBlock) {
      stringstream str ;
      str << "Thread " << currentThread ;
      bioIteratorSpan aSpan(str.str(),beginCurrentThread,mapIds[i+1]) ;
      DEBUG_MESSAGE("Create thread: " << aSpan) ;
      subListsForThreads.push_back(aSpan) ;
      ++currentThread ;
      beginCurrentThread = mapIds[i+1] ;
      sizeCurrentThread = 0 ;
    }
  }
  if (currentThread <= numberOfThreads) {
    stringstream str ;
    str << "Thread " << currentThread ;
    bioIteratorSpan aSpan(str.str(),beginCurrentThread,dataSize) ;
    DEBUG_MESSAGE("Create thread: " << aSpan) ;
    subListsForThreads.push_back(aSpan) ;
    ++currentThread ;
  }


  return currentThread-1 ;

}


patString bioIteratorInfoRepository::getInfo(patString name, patError*& err) const {
  const bioIteratorInfo* theInfo = getIteratorInfo(name,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  return theInfo->getInfo() ;
}

bioIteratorType bioIteratorInfoRepository::getType(patString name, patError*& err) const {
  const bioIteratorInfo* theInfo = getIteratorInfo(name,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return bioIteratorType() ;
  }
  return theInfo->getType() ;
}

patULong bioIteratorInfoRepository::getNumberOfIterations(patString name, patError*& err) const {
  const bioIteratorInfo* theInfo = getIteratorInfo(name,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patULong() ;
  }
  return theInfo->getNumberOfIterations() ;

}

vector<patULong> bioIteratorInfoRepository::getMapIds(patString name, patError*& err) const {
  const bioIteratorInfo* theInfo = getIteratorInfo(name,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return vector<patULong>() ;
  }
  return theInfo->getMapIds() ;
}

vector<patULong> bioIteratorInfoRepository::getRowPointers(patString name, patError*& err) const {
  const bioIteratorInfo* theInfo = getIteratorInfo(name,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return vector<patULong>() ;
  }
  return theInfo->getRowPointers() ;

}

vector<patString> bioIteratorInfoRepository::getListOfIterators() const {
  vector<patString> result ;
  for (map<patString,bioIteratorInfo>::const_iterator i = theRepository.begin() ;
       i != theRepository.end() ;
       ++i ) {
    result.push_back(i->first) ;
  }
  return result ;
}

patString bioIteratorInfoRepository::getIndexName(patString name, patError*& err) const {
  const bioIteratorInfo* theInfo = getIteratorInfo(name,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  return theInfo->indexName ;


}

