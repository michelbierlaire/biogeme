//-*-c++-*------------------------------------------------------------
//
// File name : patDrawsFromFile.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sun Aug 24 16:52:54 2003
//
//--------------------------------------------------------------------

#ifndef patDrawsFromFile_h
#define patDrawsFromFile_h

#include <fstream>
#include "patRandomNumberGenerator.h"

class patError ;

/**
   @doc Generates random numbers just by reading them from a file. No
   distribution is assumed. The file is a sequence of pairs of
   numbers. The first number is a draw from the distribution, and the
   second is the draw fron the uniform distribution that was used to
   generate the first one.  
   @author \URL[Michel
   Bierlaire]{http://people.epfl.ch/michel.bierlaire}, EPFL (Sun Aug 24 16:52:54
   2003)
 */

class patDrawsFromFile : public patRandomNumberGenerator {

public:
  patDrawsFromFile(const patString& f, patError*& err) ;
  pair<patReal,patReal> getNextValue(patError*& err) ;
  virtual patBoolean isSymmetric() const ;
  
 private:
  patString fileName ;
  ifstream theFile ;
} ;

#endif

