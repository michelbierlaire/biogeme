//-*-c++-*------------------------------------------------------------
//
// File name : patEigenVectors.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Fri Jan 20 14:53:17 2006
//
//--------------------------------------------------------------------

#ifndef patEigenVectors_h
#define patEigenVectors_h

#include <map>
#include "patError.h"
#include "patVariables.h"
class patSvd ;
class patMyMatrix ;
class patHybridMatrix ;

class patEigenVectors {
 public:
  patEigenVectors(patHybridMatrix* theMatrix, 
		  int svdMaxIter, 
patError*& err) ;
  ~patEigenVectors() ;
  map<patReal,patVariables> getEigenVector(patReal threshold) ; // patParameters::the()->getgevSingularValueThreshold()
  unsigned int getRows();
  unsigned int getCols() ;

 private:
  
  patMyMatrix* theMatrix ;
  patSvd* theSvd ;
};

#endif
