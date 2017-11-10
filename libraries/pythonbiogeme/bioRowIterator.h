//-*-c++-*------------------------------------------------------------
//
// File name : bioRowIterator.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Fri Jul 10 12:04:05  2009
//
//--------------------------------------------------------------------

#ifndef bioRowIterator_h
#define bioRowIterator_h

#include "patError.h"
#include "bioIterator.h"
#include "patVariables.h"
#include "bioIteratorSpan.h"
#include "bioIteratorInfo.h"
#include "patLoopTime.h"
/*!
  This class represents an iterator through a dataset.
  The range of the iterator is defined by "aSpan".
  The "threadSpan" defines the range designed for a given thread.
  Rule: assume aSpan = [l,u] and threadSpan[b,e].
  Then every item i in the iterations such that i is strictly before b
  are ignored. Note that the upper bound e is completely ignored. This is on purpose, in order to avoid possible misalignent with the meta iterators.

*/


class bioRowIterator: public bioIterator<const patVariables*> {

public:

  bioRowIterator(const vector<patVariables>* db, 
		 bioIteratorSpan aSpan,
		 bioIteratorSpan threadSpan,
		 patError*& err) ;

  bioRowIterator(const vector<patVariables>* db, 
		 bioIteratorSpan aSpan,
		 bioIteratorSpan threadSpan,
		 patBoolean printLoopTime,
		 patError*& err) ;
  void first() ;
  void next() ; 
  patBoolean isDone() ;
  const patVariables* currentItem() ;
  patULong getCurrentRow() const ;
  patULong nbrOfItems() const ;
protected:
  vector<patULong> rowPointers ;
  patString theIteratorName ;
  const vector<patVariables>* dataBase ;
  patULong firstRow ;
  patULong lastRow ;
  patULong currentRow ;
  patBoolean printLoopTime ;
  patLoopTime theLoopTime ;
  patULong displayInterval ;

};

#endif
