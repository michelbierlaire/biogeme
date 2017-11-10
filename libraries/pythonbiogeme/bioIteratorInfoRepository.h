//-*-c++-*------------------------------------------------------------
//
// File name : bioIteratorInfoRepository.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Jul 14 13:21:51 2009
//
//--------------------------------------------------------------------

#ifndef bioIteratorInfoRepository_h
#define bioIteratorInfoRepository_h

#include "bioIteratorInfo.h"
#include "bioIteratorSpan.h"
#include <map>

class bioIteratorInfoRepository {
  friend class bioPythonSingletonFactory ;
  friend class bioIteratorInfo ;
public:
  /**!
     @return pointer to the single instance of the class
   */
  static bioIteratorInfoRepository* the() ;


 void addIteratorInfo(patString name, bioIteratorInfo ii, patError*& err) ;

  patString debug(patError*& err) ;

  void addNewRowId(patString indexName, patULong rowNumber) ;
  void computeRowPointers(patULong sampleSize, patError*& err) ;
  bioIteratorSpan getSublist(patULong index, patError*& err) ;
  // Returns the number of threads actually needed
  patULong generateSublistsForThreads(patULong dataSize, patULong numberOfThreads,patError*& err) ;
  
  patBoolean isTopIterator(patString name, patError*& err) const ;
  patBoolean isRowIterator(patString name, patError*& err) const ;
  patBoolean isMetaIterator(patString name, patError*& err) const ;
  patString getInfo(patString name, patError*& err) const ;
  bioIteratorType getType(patString name, patError*& err) const ;
  patULong getNumberOfIterations(patString name, patError*& err) const ;
  vector<patULong> getMapIds(patString name, patError*& err) const ; 
  vector<patULong> getRowPointers(patString name, patError*& err) const ;

  vector<patString> getListOfIterators() const ;
  patString getIndexName(patString name, patError*& err) const ;
private:

  const bioIteratorInfo* getIteratorInfo(patString name, patError*& err) const ;
  bioIteratorInfoRepository() ;
  patString topIterator ;
  map<patString,bioIteratorInfo> theRepository ;
  vector<bioIteratorSpan> subListsForThreads ;
};

#endif
