//-*-c++-*------------------------------------------------------------
//
// File name : bioThreadMemory.h
// @date   Mon Apr 23 09:02:14 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioThreadMemory_h
#define bioThreadMemory_h

#include <pthread.h> 
#include <vector>
#include <map>
#include "bioTypes.h"
#include "bioString.h"
#include "bioFormula.h"

class bioExpression ;

typedef struct{
  bioUInt threadId ;
  bioBoolean calcGradient ;
  bioBoolean calcHessian ;
  bioBoolean calcBhhh ;
  std::vector<bioReal> grad;
  std::vector< std::vector<bioReal> > hessian ;
  std::vector< std::vector<bioReal> > bhhh ;
  std::vector< std::vector<bioReal> >* data ;
  std::vector< std::vector<bioUInt> >* dataMap ;
  bioReal missingData ;
  bioReal result ;
  bioUInt startData ;
  bioUInt endData ;
  bioFormula* theLoglike ;
  bioFormula* theWeight ;
  std::vector<bioUInt>* literalIds ;
  bioBoolean panel ;
} bioThreadArg ;


class bioThreadMemory {

 public:
  bioThreadMemory(bioUInt nThreads,bioUInt dim) ;
  ~bioThreadMemory() ;
  bioThreadArg* getInput(bioUInt t) ;
  void setLoglike(std::vector<bioString> f) ;
  void setWeight(std::vector<bioString> w) ;
  bioUInt numberOfThreads() ;
  bioUInt dimension() ;
  void setParameters(std::vector<bioReal>* p) ;
  void setFixedParameters(std::vector<bioReal>* p) ;
  void setData(std::vector< std::vector<bioReal> >* d) ;
  void setMissingData(bioReal md) ;
  void setDataMap(std::vector< std::vector<bioUInt> >* dm) ;
  void setDraws(std::vector< std::vector< std::vector<bioReal> > >* d) ;
  
 private:
  std::vector<bioThreadArg> inputStructures ;
  std::vector<bioFormula*> loglikes ;
  std::vector<bioFormula*> weights ;

};
#endif
