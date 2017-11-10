//-*-c++-*------------------------------------------------------------
//
// File name : patArrayIterator.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Mon Jul 10 17:10:47 2006
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <vector>
#include "patArrayIterator.h"
#include "patDisplay.h"

patArrayIterator::patArrayIterator(patPythonReal** a, 
				   unsigned long row,
				   unsigned long column,
				   vector<patString>* header) :
  theArray(a), 
  nRows(row), 
  nColumns(column), 
  currentRow(column),
  currentHeader(header) {

}
void patArrayIterator::first() {
  currentRowIndex = 0 ;
}
void patArrayIterator::next() {
  ++currentRowIndex ;
}
patBoolean patArrayIterator::isDone() {
  return (currentRowIndex >= nRows) ;
}

pair<vector<patString>*,vector<patReal>*> patArrayIterator::currentItem() {
  for (unsigned long i = 0 ; i < nColumns ; ++i) {
    currentRow[i] = theArray[currentRowIndex][i] ;
  }
  return pair<vector<patString>*,vector<patReal>*>(currentHeader,&currentRow) ;

}
