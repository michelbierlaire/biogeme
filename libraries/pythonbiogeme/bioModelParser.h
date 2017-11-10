//-*-c++-*------------------------------------------------------------
//
// File name : bioModelParser.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Thu May  7 09:52:29 2009
//
//--------------------------------------------------------------------

#ifndef bioModelParser_h
#define bioModelParser_h

#include <Python.h> // Before all other includes


#include "bioModel.h"
#include "bioLiteralRepository.h"
#include "bioRandomDraws.h"

class bioExpression ;
class bioArithPrint ;
class bioArithBayes ;

class bioModelParser {
public:
  bioModelParser(const patString& fname, patError*& err) ;
  bioModel* readModel(patError*& err) ;
  void setSampleFile(patString f) ;
  bioExpressionRepository* getRepository() ;
private:


  bioModel *theModel;
  patString filename;
  patString theDataFile ;
  bioExpression* buildExpression(PyObject* pExpression, patError*& err) ;

  bioExpression* buildMultiSum(PyObject* pExpression, patError*& err) ;

  pair<vector<patString>,patHybridMatrix* > getMatrix(PyObject* pMatrix,
						      patError*& err) ;
  
  patULong getFormula(PyObject* pBioObject, patError*& err);
  map<patString, patULong>* getStatistics(PyObject* pBioObject, patError*& err) ;
  map<patString, patULong>* getFormulas(PyObject* pBioObject, patError*& err) ;
  map<patString, bioRandomDraws::bioDraw >* getDraws(PyObject* pBioObject, patError*& err) ;
  bioArithPrint* getSimulation(PyObject* pBioObject, patError*& err) ;
  bioArithBayes* getBayesian(PyObject* pBioObject, patError*& err) ;
  void getUserExpressions(PyObject* pBioObject, patError*& err) ;
  pair<vector<patString>,patHybridMatrix* > getVarCovarMatrix(PyObject* pBioObject, patError*& err) ;
  map<patString, patULong>* getConstraints(PyObject* pBioObject, patError*& err) ;
  patULong getExclude(PyObject* pBioObject, patError*& err) ;
  patULong getWeight(PyObject* pBioObject, patError*& err) ;
  map<patString, patString>* getParameters(PyObject* pBioObject, patError*& err) ;

  void buildIteratorInfo(PyObject* pModule, patError*& err) ;

  bioLiteralRepository *theLiteralRepository;

  map<patString,patULong> userExpr ;
  vector<map<patString,patReal> > simulatedValues ;
  bioExpressionRepository* theExpRepository ;
  
};

#endif

