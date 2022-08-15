//-*-c++-*------------------------------------------------------------
//
// File name : bioThreadMemoryOneExpression.h
// @date   Tue Oct 19 14:21:00 2021
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#ifndef bioThreadMemoryOneExpression_h
#define bioThreadMemoryOneExpression_h

#include <pthread.h> 
#include <vector>
#include <map>
#include "bioTypes.h"
#include "bioString.h"
#include "bioFormula.h"
#include "bioVectorOfDerivatives.h"

class bioExpression ;

typedef struct{
  bioUInt threadId ;
  bioBoolean calcGradient ;
  bioBoolean calcHessian ;
  bioBoolean calcBhhh ;
  bioBoolean aggregation ;
  bioVectorOfDerivatives theDerivatives ;
  std::vector< std::vector<bioReal> >* data ;
  std::vector< std::vector<bioUInt> >* dataMap ;
  bioReal missingData ;
  bioUInt startData ;
  bioUInt endData ;
  bioFormula theFormula ;
  std::vector<bioUInt>* literalIds ;
  bioBoolean panel ;
} bioThreadArgOneExpression ;


class bioThreadMemoryOneExpression {

 public:
  bioThreadMemoryOneExpression() ;
  ~bioThreadMemoryOneExpression() ;
  void resize(bioUInt nThreads, bioUInt dim, bioUInt dataSize) ;
  bioThreadArgOneExpression* getInput(bioUInt t) ;
  void setFormula(std::vector<bioString> f) ;
  bioUInt numberOfThreads() ;
  bioUInt dimension() ;
  void setParameters(std::vector<bioReal>* p) ;
  void setFixedParameters(std::vector<bioReal>* p) ;
  void setData(std::vector< std::vector<bioReal> >* d) ;
  void setMissingData(bioReal md) ;
  void setDataMap(std::vector< std::vector<bioUInt> >* dm) ;
  void setDraws(std::vector< std::vector< std::vector<bioReal> > >* d) ;
  void clear() ;
  
 private:
  std::vector<bioThreadArgOneExpression> inputStructures ;
  std::vector<bioFormula> formulasPerThread ;

};

#endif
