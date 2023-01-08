//-*-c++-*------------------------------------------------------------
//
// File name : biogeme.h
// @date   Wed Apr  4 18:01:51 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef biogeme_h
#define biogeme_h

#include <vector>
#include <set>
#include <map>
#include "bioTypes.h"
#include "bioString.h"
#include "bioThreadMemory.h"
#include "bioThreadMemorySimul.h"

class bioExpression ;
class bioThreadMemory ;
class bioThreadMemorySimul ;

class biogeme {
 public:
  biogeme();
  ~biogeme() ;
  void setPanel(bioBoolean p=true) ;
  // bioString cfsqp(std::vector<bioReal>& beta,
  // 		  std::vector<bioReal>& fixedBeta,
  // 		  std::vector<bioUInt>& betaIds,
  // 		  bioUInt& nit,
  // 		  bioUInt& nf,
  // 		  bioUInt& mode,
  // 		  bioUInt& iprint,
  // 		  bioUInt& miter,
  // 		  bioReal eps) ;
  
  bioReal calculateLikelihood(std::vector<bioReal> beta,
			      std::vector<bioReal> fixedBeta) ;
  // This version is called from Python.
  bioReal calculateLikeAndDerivatives(std::vector<bioReal> beta,
				      std::vector<bioReal> fixedBeta,
				      std::vector<bioUInt> betaIds,
				      bioReal* g,
 				      bioReal* h,
 				      bioReal* bh,
				      bioBoolean hessian,
				      bioBoolean bhhh) ;

  // This version is called from C++ (by CFSQP).
  // bioReal calculateLikeAndDerivatives(std::vector<bioReal>& beta,
  // 				      std::vector<bioReal>& fixedBeta,
  // 				      std::vector<bioUInt>& betaIds,
  // 				      std::vector<bioReal>& g,
  // 				      std::vector< std::vector<bioReal> >& h,
  // 				      std::vector< std::vector<bioReal> >& bh,
  // 				      bioBoolean hessian,
  // 				      bioBoolean bhhh) ;
  // It is the same functions, where the betaIds and the fixedBetas are
  // reused from the last call.
  bioReal repeatedCalculateLikelihood(std::vector<bioReal>& beta) ;
  bioReal repeatedCalcLikeAndDerivatives(std::vector<bioReal>& beta,
				      std::vector<bioReal>& g,
				      std::vector< std::vector<bioReal> >& h,
				      std::vector< std::vector<bioReal> >& bh,
				      bioBoolean hessian,
				      bioBoolean bhhh) ;
  void setFixedBetas(std::vector<bioReal>& fixedBeta,
		     std::vector<bioUInt>& betaIds) ;

  bioReal simulateSimpleFormula(std::vector<bioString> formula,
				std::vector<bioReal> beta,
				std::vector<bioReal> fixedBeta,
				bioBoolean gradient,
				bioBoolean hessian,
				bioReal* g,
				bioReal* h) ;

  void simulateFormula(std::vector<bioString> formula,
		       std::vector<bioReal> beta,
		       std::vector<bioReal> fixedBeta,
		       std::vector< std::vector<bioReal> > data,
		       bioReal* results) ;

  void simulateSeveralFormulas(std::vector<std::vector<bioString> > formula,
			       std::vector<bioReal> beta,
			       std::vector<bioReal> fixedBeta,
			       bioUInt t,
			       std::vector< std::vector<bioReal> > data,
			       bioReal* results) ;

  
  void setExpressions(std::vector<bioString> ll,
		      std::vector<bioString> w,
		      bioUInt t) ;
  void setData(std::vector< std::vector<bioReal> >& d) ;
  void setDataMap(std::vector< std::vector<bioUInt> >& dm) ;
  void setMissingData(bioReal md) ;
  void setDraws(std::vector< std::vector< std::vector<bioReal> > >& draws) ;
  bioUInt getDimension() const ;
  void setBounds(std::vector<bioReal> lb, std::vector<bioReal> ub) ;
  std::vector<bioReal> getLowerBounds() ;
  std::vector<bioReal> getUpperBounds() ;
  void resetFunctionEvaluations() ;

  bioReal getNumericalEpsilon() ;
  bioReal getNumericalMin() ;
  bioReal getNumericalMax() ;
private: // methods
  void prepareData() ;
  void prepareDataSimul() ;
  void prepareMemoryForThreads(bioBoolean force = false) ;
  void prepareSimulMemoryForThreads(bioBoolean force = false) ;
  bioReal applyTheFormula(std::vector<bioReal>* g = NULL,
			  std::vector< std::vector<bioReal> >* h = NULL,
			  std::vector< std::vector<bioReal> >* bh = NULL) ;

private: // data
  std::vector<bioString> theLoglikeString ;
  std::vector<bioString> theWeightString ;
  bioUInt nbrOfThreads ;
  std::vector<bioUInt> literalIds ;
  std::vector<bioReal> theFixedBetas;
  bioBoolean fixedBetasDefined ;
  bioBoolean calculateHessian ;
  bioBoolean calculateBhhh ;
  bioThreadMemory theThreadMemory ;
  bioThreadMemorySimul theThreadMemorySimul ;
  std::vector< std::vector<bioReal> > theData ;
  std::vector< std::vector<bioUInt> > theDataMap ;
  std::vector< std::vector< std::vector<bioReal> > > theDraws ;
  bioReal missingData ;
  std::vector<bioThreadArg*> theInput ;
  std::vector<bioThreadArgSimul*> theSimulInput ;
  std::vector<bioReal> lowerBounds ;
  std::vector<bioReal> upperBounds ;
  bioUInt nbrFctEvaluations ;
  bioBoolean panel ;
  bioBoolean forceDataPreparation ; 

};
  

#endif
