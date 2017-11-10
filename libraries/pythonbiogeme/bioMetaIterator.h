//-*-c++-*------------------------------------------------------------
//
// File name : bioMetaIterator.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Fri Jul 10 12:15:08  2009
//
//--------------------------------------------------------------------

#ifndef bioMetaIterator_h
#define bioMetaIterator_h

#include "patError.h"
#include "bioIterator.h"
#include "bioIteratorSpan.h"
#include "patVariables.h"

class bioIteratorInfo ;

/*!
  This class represents an iterator on another iterator
  The range of the iterator is defined by "aSpan".
  The "threadSpan" defines the range designed for a given thread.
  Rule: assume aSpan = [l,u] and threadSpan[b,e].
  Then every item i in the iterations such that i is strictly before b
  are ignored. Note that the upper bound e is completely ignored. This is on purpose, in order to avoid possible misalignent with the meta iterators.
*/


class bioMetaIterator: public bioIterator<bioIteratorSpan> {

public:
  bioMetaIterator(const vector<patVariables>* db, 
		  bioIteratorSpan aSpan, 
		  bioIteratorSpan threadSpan,
		  patError*& err) ;
    
  void first() ;
  void next() ; 
  patBoolean isDone() ;
  bioIteratorSpan currentItem() ;
  // Returns the data of the first row of the block
  const patVariables* getFirstRow() ;
  patULong nbrOfItems() const ;

private:
  const vector<patVariables>* dataBase ;
  vector<patULong> rowPointers ;
  patString theIteratorName ;
  patULong firstRow ;
  patULong lastRow ;
  patULong currentRow ;

};

#endif
