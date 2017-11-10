//-*-c++-*------------------------------------------------------------
//
// File name : patGenerateNormalDraws.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Thu Mar  6 17:15:05 2003
//
//--------------------------------------------------------------------

#ifndef patGenerateNormalDraws_h
#define patGenerateNormalDraws_h

#include<vector>
#include "patError.h"
#include "patString.h"

class patNormal ;

class patGenerateNormalDraws {
  
public :
  patGenerateNormalDraws(const patString& f,
			 unsigned long nd,
			 unsigned long ni) ;
  void addVariable(const patString& v);
  void generate(unsigned int index,
		patNormal* generator,
		patError*& err) ;

private:
  vector<patString> variables ;
  patString fileName ;
  unsigned long nDraws ;
  unsigned long nIndividuals ;
};

#endif
