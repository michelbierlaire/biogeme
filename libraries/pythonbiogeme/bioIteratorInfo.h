//-*-c++-*------------------------------------------------------------
//
// File name : bioIteratorInfo.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Mon Jul 13 15:01:06  2009
//
//--------------------------------------------------------------------

#ifndef bioIteratorInfo_h
#define bioIteratorInfo_h

#include "patError.h"
#include "patString.h"

/*!

*/
typedef enum {
  META,
  ROW,
  DRAW,
  UNDEFINED
} bioIteratorType ;

class bioIteratorInfo {

  friend class bioIteratorInfoRepository ;
  friend ostream& operator<<(ostream &str, const bioIteratorInfo& x) ;
public:
  bioIteratorInfo() ;
  patBoolean operator==(const bioIteratorInfo &other) const ;
  patBoolean operator!=(const bioIteratorInfo &other) const ;


  // Ctor for iterators at the file level (aFile), or on another iterator (par) 
  bioIteratorInfo(patString aFile, 
		  patString par,
                  patString childIteratorName,
                  patString anIndex, 
                  bioIteratorType aType) ;
  
  // Ctor for draws iterator
  bioIteratorInfo(patString aFile) ;

  patString getInfo() const ;

  void computePointers(patULong sampleSize,patError*& err) ;

  patBoolean isTopIterator() const;
  patBoolean isRowIterator() const ;
  patBoolean isMetaIterator() const ;
  patBoolean isDrawIterator() const ;

  bioIteratorType getType() const ;

  patULong getNumberOfIterations() const ;

  vector<patULong> getMapIds() const ; 
  vector<patULong> getRowPointers() const ;

 protected:
  // Contains the IDs of the first row of each block
  vector<patULong> mapIds ; 

  // Contains for each row a pointer to the first row of the next
  // block, or patBadId if the row does not correspond to the
  // beginning of a block.
  vector<patULong> rowPointers ;

  patULong numberOfIterations ;

  patString childIteratorName ;
  patString datafile ;
  patString indexName ;
  bioIteratorType type ;
  patString theParent ;
  patString iteratorName ;

};


#endif
