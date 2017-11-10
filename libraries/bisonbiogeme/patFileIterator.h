//-*-c++-*------------------------------------------------------------
//
// File name : patFileIterator.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sun Jul  9 09:45:41 2006
//
//--------------------------------------------------------------------

#ifndef patFileIterator_h
#define patFileIterator_h

#include <fstream>
#include <vector>
#include "patIterator.h"

/**
   Iterator used when the data are made available by the user in one
   or several files. This is the default for biogeme.
 */

class patFileIterator : public patIterator<pair<vector<patString>*,
                                                vector<patReal>  * > > {
 public:
  /**
   */
  patFileIterator(patString fileName, 
		  unsigned long rowSize,
		  vector<patString>* header) ;
    /**
   */
  void first() ;
  /**
   */
  void next() ;

  /**
   */
  patBoolean isDone() ;

  /**
   */
  pair<vector<patString>*,vector<patReal>*> currentItem() ;

 private:
  void readOneLine() ;
  
 private:
  char* buffer ;
  vector<patReal> currentRow ;
  vector<patString>* currentHeader ;
  patString theFileName ;
  ifstream f ;
};

#endif
