//-*-c++-*------------------------------------------------------------
//
// File name : patArrayIterator.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Mon Jul 10 17:06:53 2006
//
//--------------------------------------------------------------------

#ifndef patArrayIterator_h
#define patArrayIterator_h

#include <fstream>
#include <vector>
#include "patIterator.h"


/**
   Iterator used when the data are made available by the user in an
   array. ypically, this is used when biogeme is called from python.
 */

class patArrayIterator : public patIterator<pair<vector<patString>*,
                                                vector<patReal>  * > > {
 public:
  /**
   */
  patArrayIterator(patPythonReal** a, 
		   unsigned long row,
		   unsigned long column,
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
  patPythonReal** theArray ;
  unsigned long nRows ;
  unsigned long nColumns ;
  vector<patReal> currentRow ;
  vector<patString>* currentHeader ;
  unsigned long currentRowIndex ;
};

#endif
