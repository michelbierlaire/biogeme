//-*-c++-*------------------------------------------------------------
//
// File name : patQuantiles.h
// Author :   Michel Bierlaire
// Date :     Sun May  6 08:05:15 2012
//
//--------------------------------------------------------------------
//

#ifndef patQuantiles_h
#define patQuantiles_h

#include "patError.h"
#include "patVariables.h"

class patQuantiles {
public:
  patQuantiles(patVariables* x) ;
  patReal getQuantile(patReal p, patError*& err) ;
 private:
  patVariables* data ;
  
};

#endif
