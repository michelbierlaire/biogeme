//-*-c++-*------------------------------------------------------------
//
// File name : bioUiFile.h
// Author :    Michel Bierlaire
// Date :      Tue Mar 14 09:10:09 2017
//
//--------------------------------------------------------------------

#ifndef bioUiFile_h
#define bioUiFile_h

#include <vector>
#include "patString.h"
#include "patType.h"
#include "patError.h"

class bioUiFile {

 public :
  bioUiFile(patString name="gtkbiogeme.ui") ;
  patBoolean fileFound() ;
  patString getUiFile(patError*& err) ;
 private:
  patString fileName ;
  patString fullPath ;
  vector<patString> listOfDirectories ;
  
};
#endif
