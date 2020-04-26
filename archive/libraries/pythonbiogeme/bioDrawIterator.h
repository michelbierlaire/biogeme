//-*-c++-*------------------------------------------------------------
//
// File name : bioDrawIterator.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Fri Jul 31 13:23:48 2009
//
//--------------------------------------------------------------------

#ifndef bioDrawIterator_h
#define bioDrawIterator_h

#include <map>
#include "bioIterator.h"
#include "patVariables.h"

class bioIteratorInfo ;

/*!
  This class represents an iterator through the draws from random variables
*/


class bioDrawIterator: public bioIterator<pair<patReal **,patReal**> > {
public:

  bioDrawIterator(patReal ***db, patReal ***unif,patULong R) ;
  void first() ;
  void next() ; 
  patBoolean isDone() ;
  pair<patReal**,patReal**> currentItem() ;
  patULong getNumberOfDraws() const ;
  patULong nbrOfItems() const ;

private:
  patReal ***dataBase ;
  patReal ***uniform ;
  patULong currentRow ;
  patULong nDraws ;
  
};

#endif
