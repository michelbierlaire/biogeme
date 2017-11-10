//-*-c++-*------------------------------------------------------------
//
// File name : bioPrecompiledFunction.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Sat Apr 23 09:28:13 2011
//
//--------------------------------------------------------------------

// We declare here the user defined objects

#ifndef bioPrecompiledFunction_h
#define bioPrecompiledFunction_h

#include "bioPythonWrapper.h"
#include "bioStatistics.h"
#include "bioConstraints.h"
#include "bioIteratorSpan.h"

class bioExpression ;
class bioArithListOfExpressions ;
class bioExpressionRepository ;
typedef struct{
  patULong threadId ;
  trVector* grad;
  trHessian* hessian ;
  trHessian* bhhh ;
  patBoolean* success ;
  patError* err ;
  bioSample* sample ;
  patReal result ;
  bioIteratorSpan* subsample ; 
  bioExpressionRepository* theExpressionRepository ;
  patULong theExpressionId ;
  bioArithListOfExpressions* theFunctionAndDerivatives ;
  vector<patULong> literalIds ;
} bioThreadArg ;

class bioPrecompiledFunction: public bioPythonWrapper {

public :

  bioPrecompiledFunction(bioExpressionRepository* rep, patULong expId, patError*& err) ;
  ~bioPrecompiledFunction() ;
  patReal computeFunction(trVector* x,
			  patBoolean* success,
			  patError*& err) ;
  patReal computeFunctionAndDerivatives(trVector* x,
					trVector* grad,
					trHessian* hessian,
					patBoolean* success,
					patError*& err); 

  trHessian* computeCheapHessian(trHessian* hessian,
				 patError*& err) ;

  patBoolean isCheapHessianAvailable() ;

  trVector* computeHessianTimesVector(trVector* x,
				      const trVector* v,
				      trVector* r,
				      patBoolean* success,
				      patError*& err)  ;
  
  patBoolean isGradientAvailable() const ;

  patBoolean isHessianAvailable()  const ;

  patBoolean isHessianTimesVectorAvailable() const ;

  unsigned long getDimension() const ;
  
  trVector getCurrentVariables() const ;
  
  patBoolean isUserBased() const ;

  void generateCppCode(ostream& str, patError*& err) ;


  void generateDerivatives(patError*& err) ;
 private:
  vector<bioExpressionRepository*> theExpressionRepository ;
  patULong theExpressionId ;
  trHessian bhhh ;
  //  trHessian hessian ;
  vector<trVector> threadGrad ;
  vector<trHessian> threadBhhh ;
  vector<patBoolean> threadSuccess ;
  vector<trHessian> threadHessian ;
  vector<vector<patULong> > threadChildren ;
  vector<bioIteratorSpan> threadSpans ;

  vector<patULong> betaIds ;
  pthread_t *threads;
  bioThreadArg *input;

};

class myBioStatistics: public bioStatistics {
public:
  
  void bio__generateStatistics(bioSample* sample, patError*& err) ;
};

class myBioConstraints: public bioConstraints {
public:
  
  void addConstraints() ;
};



#endif
