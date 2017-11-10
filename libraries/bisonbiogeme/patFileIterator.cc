//-*-c++-*------------------------------------------------------------
//
// File name : patFileIterator.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sun Jul  9 09:51:42 2006
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patDisplay.h"
#include "patFileIterator.h"
#include "patParameters.h"

patFileIterator::patFileIterator(patString fileName,
				 unsigned long rowSize,
				 vector<patString>* header) :
  currentRow(rowSize),currentHeader(header),f(fileName.c_str())  {

  if (!f) {
    WARNING("Unable to open file " << fileName) ;
  }
}

void patFileIterator::first() {
  // Rewind ot the beginning of the file
  f.clear();              
  f.seekg(0, ios::beg);   
  // Skip the first row
  buffer = new char[patParameters::the()->getgevBufferSize()] ;
  f.getline(buffer,patParameters::the()->getgevBufferSize()) ;
  delete[] buffer ; 
  
  readOneLine() ;
}
void patFileIterator::next() {
  readOneLine() ;
}
patBoolean patFileIterator::isDone() {
  if (!f) {
    return patTRUE ;
  }
  return f.eof() ;
}
pair<vector<patString>*,vector<patReal>*> patFileIterator::currentItem() {
  return pair<vector<patString>*,vector<patReal>*>(currentHeader,&currentRow) ;
}

void patFileIterator::readOneLine() {
  if (!f.eof()) {
    f >> currentRow[0] ;
    if (f.eof()) {
      DEBUG_MESSAGE("End of file detected") ;
    }
    else {
      for (unsigned long i = 1 ;  i < currentRow.size() ; ++i) {
	if (!f.eof()) {
	  f >> currentRow[i] ;
	}
      }
    }
  } 
}
