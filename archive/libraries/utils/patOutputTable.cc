//-*-c++-*------------------------------------------------------------
//
// File name : patOutputTable.h
// Author :    Michel Bierlaire
// Date :      Thu Jun 28 07:14:46 2007
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include "patOutputTable.h"
#include "patErrMiscError.h"
#include "patDisplay.h"

patOutputTable::patOutputTable(unsigned short c, patBoolean jl) : 
  justifyLeft(jl),
  colWidth(c,0) {

}
  
void patOutputTable::appendRow(vector<patString> row, patError*& err) {
  if (row.size() != nCols()) {
    stringstream str ;
    str << "Appended row has " << row.size() << " columns instead of " << nCols() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  theTable.push_back(row) ;
}

void patOutputTable::computeColWidth() {
  for (vector<vector<patString> >::iterator rowIter = theTable.begin() ;
       rowIter != theTable.end() ;
       ++rowIter) {
    for (unsigned short c = 0 ; c < nCols() ; ++c) {
      if ((*rowIter)[c].size() > colWidth[c]) {
	colWidth[c] = (*rowIter)[c].size() ;
      }
    }
  }
}

ostream& operator<<(ostream &str, const patOutputTable& x) {
  for (vector<vector<patString> >::const_iterator rowIter = x.theTable.begin() ;
       rowIter != x.theTable.end() ;
       ++rowIter) {
    for (unsigned short c = 0 ; c < x.nCols() ; ++c) {
      if (x.colWidth[c] > 0) {
	str << fillWithBlanks((*rowIter)[c],x.colWidth[c]+1,x.justifyLeft) ;
      }
    }
    str << endl ;
  }
  return str ;
}

unsigned short patOutputTable::nCols() const {
  return colWidth.size() ;
}
