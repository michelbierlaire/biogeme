//-*-c++-*------------------------------------------------------------
//
// File name : patGenerateNormalDraw.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Thu Mar  6 17:18:06 2003
//
//--------------------------------------------------------------------

#include <fstream>
#include <iomanip>
#include "patErrFileNotFound.h"
#include "patErrNullPointer.h"
#include "patDisplay.h"
#include "patNormal.h"
#include "patGenerateNormalDraws.h"

patGenerateNormalDraws::patGenerateNormalDraws(const patString& f,
					       unsigned long nd,
					       unsigned long n) :
  fileName(f),
  nDraws(nd),
  nIndividuals(n) {

}

void patGenerateNormalDraws::addVariable(const patString& v) {
  variables.push_back(v) ;
}

void patGenerateNormalDraws::generate(unsigned int index,
				      patNormal* generator,
				      patError*& err) {

  if (generator == NULL) {
    err = new patErrNullPointer("patNormal") ;
    return ;
  }

  DEBUG_MESSAGE("Generate " << nDraws << " draws for " << nIndividuals << " individual") ;

  ofstream theFile(fileName.c_str()) ;
  if (!theFile) {
    err = new patErrFileNotFound(fileName) ;
    WARNING(err->describe()) ;
    return ;
  }
  theFile << setprecision(7) << setiosflags(ios::scientific|ios::showpos) ;
  unsigned int nVar = variables.size() ;

  for (vector<patString>::iterator i = variables.begin() ;
       i != variables.end() ;
       ++i) {
    if (i != variables.begin()) {
      theFile << '\t' ;
    }
    theFile << *i ;
  }
  theFile << endl ;

  for (unsigned long row = 0 ; row < nDraws*nIndividuals ; ++row) {
    for (unsigned int col = 0 ; col < nVar ; ++ col) {
      if (col != 0) {
	theFile << '\t' ;
      }
      pair<patReal,patReal> draws = generator->getNextValue(err) ;
      theFile << draws.first ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
    theFile << endl ;
  }   theFile.close() ;
  return ;
}

