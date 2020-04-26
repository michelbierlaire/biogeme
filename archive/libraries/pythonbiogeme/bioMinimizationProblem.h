//-*-c++-*------------------------------------------------------------
//
// File name : bioMinimizationProblem.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Fri Aug  7 10:19:01 2009
//
//--------------------------------------------------------------------

#ifndef bioMinimizationProblem_h
#define bioMinimizationProblem_h

#include "patMyMatrix.h"
#include "patNonLinearProblem.h"
#include "trBounds.h"
#include "trParameters.h"

class bioMinimizationProblem : public patNonLinearProblem {
public:
  bioMinimizationProblem(trFunction* f,trBounds b,trParameters theTrParameters) ;
  virtual unsigned long nVariables()  ;

  virtual unsigned long nNonLinearIneq()  ;
  virtual unsigned long nNonLinearEq() ;
  virtual unsigned long nLinearIneq()  ;
  virtual unsigned long nLinearEq() ;
  virtual patVariables getLowerBounds(patError*& err) ;
  virtual patVariables getUpperBounds(patError*& err) ;
  virtual trFunction* getObjective(patError*& err) ;
  virtual trFunction* getNonLinInequality(unsigned long i,
				    patError*& err) ;
  virtual trFunction* getNonLinEquality(unsigned long i,
					patError*& err) ;

  virtual pair<patVariables,patReal> 
  getLinInequality(unsigned long i,
		   patError*& err) ;
  virtual pair<patVariables,patReal> 
  getLinEquality(unsigned long i,
		 patError*& err) ;

  virtual patString getProblemName() ;
  void addEqualityConstraint(trFunction* c) ;
  patBoolean isFeasible(trVector& x, patError*& err) const ; 
  patULong getSizeOfVarCovar() ;
  patBoolean computeVarCovar(trVector* x,
			     patMyMatrix* varCovar,
			     patMyMatrix* robustVarCovar,
			     map<patReal,patVariables>* eigVec,
			     patReal* smallSV,
			     patError*& err) ;
private:
  trFunction* theFunction ;
  trBounds theBounds ;
  vector<trFunction*> equalityConstraints ;
  patMyMatrix* bigMatrix ;
  patMyMatrix* matHess ;
  trParameters theTrParameters ;
};

#endif
