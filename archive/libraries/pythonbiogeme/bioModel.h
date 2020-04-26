//-*-c++-*------------------------------------------------------------
//
// File name : bioModel.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Mon Apr 27 15:41:49 2009
//
//--------------------------------------------------------------------

#ifndef bioModel_h
#define bioModel_h

#include "bioExpression.h"
#include "patHybridMatrix.h"
#include <map>

/*!
Class defining a model in the general sense. 
*/

class bioArithPrint ;
class bioArithBayes ;
class bioModel {

public:
  bioModel();
  void setFormula(patULong theFormula) ;
  void setExcludeExpr(patULong excludeExpr) ;
  void setWeightExpr(patULong weightExpr) ;
  void setStatistics(map<patString, patULong>* stat) ;
  void setFormulas(map<patString, patULong>* stat) ;
  void setConstraints(map<patString, patULong>* constr) ;
  void setSimulation(bioArithPrint* simul) ;
  void setBayesian(bioArithBayes* b) ;
  void setSensitivityAnalysis(vector<patString> params, 
			      patHybridMatrix* simul) ;
  void setRepository(bioExpressionRepository* rep) ;
  patULong getFormula() ;
  patULong getExcludeExpr() ;
  patULong getWeightExpr() ;
  map<patString, patULong>* getStatistics() ;
  map<patString, patULong>* getFormulas() ;
  map<patString, patULong>* getConstraints() ;
  vector<pair<patString, patULong> >* getUserExpressions() ;
  bioArithPrint* getSimulation() ;
  bioArithBayes* getBayesian() ;

  patBoolean mustEstimate() const ;
  patBoolean mustBayesEstimate() const ;
  patBoolean mustSimulate() const ;
  patBoolean mustPerformSensitivityAnalysis() const ;
  patHybridMatrix* getVarCovarForSensitivity() ;
  vector<patString> getNamesForSensitivity() ;
  bioExpressionRepository* getRepository() ;
  void addUserExpression(patString name, patULong exprId) ;
  patBoolean involvesMonteCarlo() ;
protected:
  patULong theFormula ;
  map<patString, patULong>* statistics ;
  map<patString, patULong>* formulas ;
  map<patString, patULong>* constraints ;
  vector<pair<patString, patULong> > userExpressions ;
  bioArithPrint* simulation ;
  patULong excludeExpr ;
  patULong weightExpr ;
  bioExpressionRepository* theRepository ;
  patHybridMatrix* varCovarForSensitivity;
  vector<patString> namesForSensitivity ;
  patBoolean performSensitivityAnalysis ;
  bioArithBayes* theBayesianExpr ;
};


#endif
