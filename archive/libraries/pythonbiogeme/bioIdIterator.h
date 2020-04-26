//-*-c++-*------------------------------------------------------------
//
// File name : bioIdIterator.h
// Author :    Michel Bierlaire
// Date :      Tue May 19 18:21:48 2015
//
//--------------------------------------------------------------------

#ifndef bioIdIterator_h
#define bioIdIterator_h

#include "patError.h"
#include "bioRowIterator.h"
#include "patVariables.h"
#include "bioIteratorSpan.h"
#include "bioIteratorInfo.h"
#include "patLoopTime.h"
/*!  
  This class represents an iterator extracting
  the first row of each block of row. A block is defined a
  list of consecutive rows with the same value of an ID in a specified
  column.

  The range of the iterator is defined by "aSpan".  The "threadSpan"
  defines the range designed for a given thread.  Rule: assume aSpan =
  [l,u] and threadSpan[b,e].  Then every item i in the iterations such
  that i is strictly before b are ignored. Note that the upper bound e
  is completely ignored. This is on purpose, in order to avoid
  possible misalignent with the meta iterators.

*/


class bioIdIterator: public bioRowIterator {

public:

  bioIdIterator(const vector<patVariables>* db, 
		bioIteratorSpan aSpan,
		bioIteratorSpan threadSpan,
		patULong col,
		patError*& err) ;
  
  void first() ;
  void next() ; 
  patBoolean isDone() ;
  const patVariables* currentItem() ;

private:
  patULong column ;
};

#endif
