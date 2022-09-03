//-*-c++-*------------------------------------------------------------
//
// File name : evaluateExpressions.h
// @date   Thu Oct 14 13:41:49 2021
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#ifndef evaluateExpressions_h
#define evaluateExpressions_h

#include <vector>
#include "bioString.h"
#include "bioVectorOfDerivatives.h"
#include "bioThreadMemoryOneExpression.h"


class evaluateOneExpression {
 public:
  evaluateOneExpression() ;
  void setExpression(std::vector<bioString> f) ;
  void setFreeBetas(std::vector<bioReal> freeBetas) ;
  void setFixedBetas(std::vector<bioReal> fixedBetas) ;
  void setData(std::vector< std::vector<bioReal> >& d) ;
  void setDataMap(std::vector< std::vector<bioUInt> >& dm) ;
  void setDraws(std::vector< std::vector< std::vector<bioReal> > >& draws) ;
  void setMissingData(bioReal md) ;
  void setNumberOfThreads(bioUInt n) ;

  void calculate(bioBoolean gradient,
		 bioBoolean hessian,
		 bioBoolean bhhh,
		 bioBoolean aggregation) ;
  void getResults(bioReal* f, bioReal* g, bioReal* h, bioReal* bhhh) ;
  bioUInt getDimension() const ;
  bioUInt getSampleSize() const ;
private:
  void prepareData() ;
  void applyTheFormula() ;
private:
  bioVectorOfDerivatives results ;
  bioBoolean with_data ;
  bioBoolean with_g ;
  bioBoolean with_h ;
  bioBoolean with_bhhh ;
  bioBoolean aggregation ;
  std::vector<bioThreadArgOneExpression*> theInput ;
  std::vector<bioUInt> literalIds ;
  std::vector<bioString> expression ;
  std::vector<bioReal> theFreeBetas;
  std::vector<bioReal> theFixedBetas;
  std::vector< std::vector<bioReal> > theData ;
  std::vector< std::vector<bioUInt> > theDataMap ;
  std::vector< std::vector< std::vector<bioReal> > > theDraws ;
  bioReal missingData ;
  bioBoolean panel ;
  bioBoolean gradientCalculated ;
  bioBoolean hessianCalculated ;
  bioBoolean bhhhCalculated ;
  bioThreadMemoryOneExpression theThreadMemory ;
  bioUInt nbrOfThreads ;
  
};
#endif
