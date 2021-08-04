//-*-c++-*------------------------------------------------------------
//
// File name : bioThreadMemorySimul.h
// @date   Mon Mar  8 14:31:41 2021
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioThreadMemorySimul_h
#define bioThreadMemorySimul_h

#include <pthread.h> 
#include <vector>
#include <map>
#include "bioTypes.h"
#include "bioString.h"
#include "bioSeveralFormulas.h"

class bioSeveralExpressions ;

typedef struct{
  bioUInt threadId ;
  std::vector< std::vector<bioReal> > results;
  std::vector< std::vector<bioReal> >* data ;
  std::vector< std::vector<bioUInt> >* dataMap ;
  bioReal missingData ;
  bioUInt startData ;
  bioUInt endData ;
  bioSeveralFormulas theFormulas ;
  bioBoolean panel ;
} bioThreadArgSimul ;


class bioThreadMemorySimul {

 public:
  bioThreadMemorySimul() ;
  ~bioThreadMemorySimul() ;
  void resize(bioUInt nThreads) ;
  bioThreadArgSimul* getInput(bioUInt t) ;
  void setFormulas(std::vector<std::vector<bioString> > vectOfExpressionsStrings) ;
  bioUInt numberOfThreads() ;
  bioUInt dimension() ;
  void setParameters(std::vector<bioReal>* p) ;
  void setFixedParameters(std::vector<bioReal>* p) ;
  void setData(std::vector< std::vector<bioReal> >* d) ;
  void setMissingData(bioReal md) ;
  void setDataMap(std::vector< std::vector<bioUInt> >* dm) ;
  void setDraws(std::vector< std::vector< std::vector<bioReal> > >* d) ;
  
 private:
  std::vector<bioThreadArgSimul> inputStructures ;
  std::vector<bioSeveralFormulas> theFormulas ;

};

#endif
