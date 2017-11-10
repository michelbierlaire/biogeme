//-*-c++-*------------------------------------------------------------
//
// File name : patOutputTable.h
// Author :    \URL[Michel Bierlaire]{http://transp-or2.epfl.ch}
// Date :      Thu Jun 28 07:14:46 2007
//
//--------------------------------------------------------------------

#ifndef patOutputTable_h
#define patOutputTable_h

#include <vector>
#include "patString.h"
#include "patError.h"
#include "patType.h"

class patOutputTable {

  /**
   */
  friend ostream& operator<<(ostream &str, const patOutputTable& x) ;
  
 public:
  /**
   */
  patOutputTable(unsigned short c, patBoolean jl) ;
  
  /**
   */
  void appendRow(vector<patString> row, patError*& err) ;

  /**
   */
  unsigned short nCols() const ;
  /**
   */
  void computeColWidth() ;

 protected:
  patBoolean justifyLeft ;
  vector<vector<patString> > theTable ;
  vector<unsigned short> colWidth ;
};


#endif
