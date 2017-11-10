//-*-c++-*------------------------------------------------------------
//
// File name : bioModelParser.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Thu May  7 09:53:57 2009
//
//--------------------------------------------------------------------

#include <iostream>
#include <list>
#include <new>

#include "patString.h"
#include "patError.h"
#include "patDisplay.h"
#include "patFileExists.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "patEnvPathVariable.h"
#include "bioArithPrint.h"

#include "patHybridMatrix.h"
#include "bioParameters.h"

#include "bioModelParser_i.h"
#include "bioModelParser.h"

#include "bioArithAbs.h"
#include "bioArithAnd.h"
#include "bioArithUnaryMinus.h"
#include "bioArithBinaryMinus.h"
#include "bioArithBinaryPlus.h"
#include "bioArithDivide.h"
#include "bioArithEqual.h"
#include "bioArithExp.h"
#include "bioArithGreater.h"
#include "bioArithGreaterEqual.h"
#include "bioArithLess.h"
#include "bioArithLessEqual.h"
#include "bioArithLog.h"
#include "bioArithNormalCdf.h"
#include "bioArithMult.h"
#include "bioArithNotEqual.h"
#include "bioArithOr.h"
#include "bioArithPower.h"
#include "bioArithConstant.h"
#include "bioArithSum.h"
#include "bioArithMonteCarlo.h"
#include "bioArithProd.h"
#include "bioArithPrint.h"
#include "bioArithGHIntegral.h"
#include "bioArithDerivative.h"
#include "bioArithElem.h"
#include "bioArithLogLogit.h"
#include "bioArithNormalPdf.h"
#include "bioArithRandom.h"
#include "bioArithUnifRandom.h"
#include "bioArithMax.h"
#include "bioArithMin.h"
#include "bioArithVariable.h"
#include "bioArithFixedParameter.h"
#include "bioArithRandomVariable.h"
#include "bioArithCompositeLiteral.h"
#include "bioExpressionRepository.h"
#include "bioRandomDraws.h"
#include "bioRowIterator.h"
#include "bioIteratorInfo.h"
#include "bioIteratorInfoRepository.h"
#include "bioArithMultinaryPlus.h"
#include "bioArithBayesMH.h"
#include "bioArithBayesMean.h"

// #define DEBUG_MESSAGE(s) GENERAL_MESSAGE("[INFO] " << s)
// #define DEBUG_MESSAGE(s) 

/**
   Constructor
   \fname python module file (no .py extension) which contains the
   model specification
*/
bioModelParser::bioModelParser(const patString& fname,patError*& err) :  filename(fname), theExpRepository(NULL) {

  // Filename must not contain any path
  if (filename.find('/')!=string::npos) {
    err = new patErrMiscError("Filename contains directory path!") ;
    WARNING(err->describe()) ;
    return ;
  }
  theLiteralRepository = bioLiteralRepository::the() ;
  theExpRepository = new bioExpressionRepository(0) ;
}


bioModel *bioModelParser::readModel(patError*& err) {
  PyObject* pModule(NULL), *pBioObject(NULL) ;
  PyObject* pOperator(NULL) ;
  bioExpression* theFormula ;

  // Check that the file exists
  stringstream fileWithExtension ;
  fileWithExtension << filename << ".py" ;
  if (!patFileExists()(fileWithExtension.str())) {
    stringstream str ;
    str << "File " << fileWithExtension.str() << " does not exist" ;
    err = new patErrMiscError(str.str()) ;
    return NULL ;
  }
  else {
    DEBUG_MESSAGE(fileWithExtension.str() << " exists") ;
  }


  // Remove .py extension from the file name if present
  patString fn(filename) ;
  patString pythonFileExtension(".py") ;
  int extensionPos = fn.size()-pythonFileExtension.size() ;
  if (fn.find(pythonFileExtension, extensionPos)!=string::npos) {
    fn.erase(extensionPos) ;
  }


   patEnvPathVariable pythonPath("PYTHONPATH") ;
   if (!pythonPath.readFromSystem()) {
     err = new patErrMiscError("Environment variable PYTHONPATH is undefined") ;
     WARNING(err->describe()) ;
   }


  // Init Python
#ifdef STANDALONE_EXECUTABLE
  Py_NoSiteFlag = 1;
#endif 
  Py_Initialize() ;
  // Load the module and check if it has loaded
  //GENERAL_MESSAGE("loading python module");
  pModule = PyImport_ImportModule(const_cast<char *>(fn.c_str())) ;
  if (PY_FAIL(pModule)) {
    PY_ERR_PRINT() ;
    stringstream str ;
    str << "Failed to load " << filename ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL;
  }
  //GENERAL_MESSAGE("python module loaded:");
  //PyObject_Print(pModule, stdout, Py_PRINT_RAW);


  GET_PY_OBJ(pModule, OPERATOR_OBJECT, pOperator) ;
  setOperators(pOperator) ;


  buildIteratorInfo(pModule,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }

  GET_PY_OBJ(pModule, BIOGEME_OBJECT, pBioObject) ;


  theModel = new bioModel() ;
  theModel->setRepository(theExpRepository) ;
  map<patString, patString>* parameters = getParameters(pBioObject,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  
  bioParameters::the()->setParameters(parameters,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  
  // Get expression for statistics
  map<patString, patULong>* statistics = getStatistics(pBioObject,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  theModel->setStatistics(statistics) ;

  // Get expression for draws

  map<patString, bioRandomDraws::bioDraw >* draws = getDraws(pBioObject,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }

  bioRandomDraws::the()->setDraws(draws,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }



  // Get formulas to be reported
  map<patString, patULong>* formulas = getFormulas(pBioObject,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  theModel->setFormulas(formulas) ;








  // Get expression for exclusion
  patULong excludeExpr = getExclude(pBioObject,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  theModel->setExcludeExpr(excludeExpr) ;


  // Get expression for weight
  patULong weightExpr = getWeight(pBioObject,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  theModel->setWeightExpr(weightExpr) ;

  // Check if STAN files must be generated
  // patBoolean stan = getStan(pBioObject,err) ;
  // if (err != NULL) {
  //   WARNING(err->describe()) ;
  //   return NULL ;
  // }
  // theModel->setStan(stan) ;

  // The likelihood function
  patULong theFormulaId = getFormula(pBioObject,err) ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  theFormula = theExpRepository->getExpression(theFormulaId) ;



  if (theFormula != NULL) {
    
    // A formula for the likelihood function is present. Estimation will be performed
    theModel->setFormula(theFormulaId) ;
    
    
    // Get expressions for CONSTRAINTS 
    map<patString, patULong>* constraints = getConstraints(pBioObject,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    theModel->setConstraints(constraints) ;
  }
  else {

    // Check if Bayesian estimation is requested
    bioArithBayes* theBayesian = getBayesian(pBioObject,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    if (theBayesian != NULL) {
      theModel->setBayesian(theBayesian) ;
    }
    else {
      bioArithPrint* theSimulFormula = getSimulation(pBioObject,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
  	return NULL ;
      }
      theModel->setSimulation(theSimulFormula) ;
      pair<vector<patString>,patHybridMatrix*> varCovMatrix = getVarCovarMatrix(pBioObject, err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
      if (varCovMatrix.second != NULL) {
	GENERAL_MESSAGE("Sensitivity analysis will be performed") ;
	theModel->setSensitivityAnalysis(varCovMatrix.first,varCovMatrix.second) ;
      }
    }
  }


  Py_DECREF(pModule) ;

  
  //  DEBUG_MESSAGE("Model specification read!") ;

  // Stop Python
  PyErr_Clear(); 
  Py_Finalize() ;
  
  return theModel ;
}


patULong bioModelParser::getFormula(PyObject* pBioObject,patError*& err) {
  bioExpression* theFormula ;
  PyObject* pFormula(NULL) ;

  GET_PY_OBJ(pBioObject, ESTIMATE, pFormula) ;
  if (pFormula != NULL) {
    if (pFormula != Py_None) {
    

      theFormula = buildExpression(pFormula,err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return patBadId ;
      }
       
      if (theFormula == NULL) {
        err = new patErrNullPointer("bioExpression") ;
        WARNING(err->describe()) ;
        return patULong() ;
      }
      return theFormula->getId() ;
    }else {
      //      DEBUG_MESSAGE("No formula (None) for the loglikelihood has been provided") ;
      return patBadId ;
    }
  }
  else {
    //    DEBUG_MESSAGE("No formula for the loglikelihood has been provided") ;
    return patBadId ;
  }
}


patULong bioModelParser::getExclude(PyObject* pBioObject,patError*& err) {
  bioExpression* theFormula ;

  PyObject* pFormula(NULL) ;

  GET_PY_OBJ(pBioObject, EXCLUDE, pFormula) ;
  if (pFormula != NULL) {
    if (pFormula != Py_None) {
      theFormula = buildExpression(pFormula,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patBadId ;
      }
      //      DEBUG_MESSAGE("EXCLUDE: " << *theFormula) ;
      return theFormula->getId() ;
    }
    else {
      //      DEBUG_MESSAGE("No formula (None) for the exclude condition has been provided") ;
      return patBadId ;

    }
  }
  else {
    //    DEBUG_MESSAGE("No formula for the exclude condition has been provided") ;
    return patBadId ;
  }
}


patULong bioModelParser::getWeight(PyObject* pBioObject,patError*& err) {

  bioExpression* theFormula ;

  PyObject* pFormula(NULL) ;

  GET_PY_OBJ(pBioObject, WEIGHT, pFormula) ;
  if (pFormula != NULL) {
    if (pFormula != Py_None) {
      theFormula = buildExpression(pFormula,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patBadId ;
      }
      return theFormula->getId() ;
    }
    else {
      //      DEBUG_MESSAGE("No formula (None) for the weights has been provided") ;
      return patBadId ;

    }
  }
  else {
    //    DEBUG_MESSAGE("No formula for the weights has been provided") ;
    return patBadId ;
  }
}


map<patString, patULong>* bioModelParser::getStatistics(PyObject* pBioObject, patError*& err ) {
  PyObject* pStatistics(NULL) ;
  PyObject* pKeyList(NULL) ;
  map<patString,patULong>* statisticsDic = new map<patString,patULong> ;
  
  GET_PY_OBJ(pBioObject, STATISTICS, pStatistics) ;
  pKeyList = PyDict_Keys(pStatistics) ;

  Py_ssize_t listSize = PyList_Size(pKeyList) ;
  for (Py_ssize_t pos=0 ; pos<listSize ; ++pos) {
    PyObject* pName = PyList_GetItem(pKeyList, pos) ;
    PyObject* pStat = PyDict_GetItem(pStatistics, pName) ;

    patString statName = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pName))) ;
    //DEBUG_MESSAGE("Build expression for " << statName) ;
    bioExpression *expr = buildExpression(pStat,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    statisticsDic->insert(pair<patString,patULong>(statName, expr->getId())) ;
  }
  return statisticsDic ;
}


map<patString, patULong>* bioModelParser::getFormulas(PyObject* pBioObject, patError*& err ) {
  PyObject* pFormulas(NULL) ;
  PyObject* pKeyList(NULL) ;
  map<patString,patULong>* formulasDic = new map<patString,patULong> ;
  
  GET_PY_OBJ(pBioObject, FORMULAS, pFormulas) ;
  pKeyList = PyDict_Keys(pFormulas) ;

  Py_ssize_t listSize = PyList_Size(pKeyList) ;
  for (Py_ssize_t pos=0 ; pos<listSize ; ++pos) {
    PyObject* pName = PyList_GetItem(pKeyList, pos) ;
    PyObject* pForm = PyDict_GetItem(pFormulas, pName) ;

    patString formulaName = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pName))) ;
    bioExpression *expr = buildExpression(pForm,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    formulasDic->insert(pair<patString,patULong>(formulaName, expr->getId())) ;
  }
  return formulasDic ;
}



map<patString, bioRandomDraws::bioDraw >* bioModelParser::getDraws(PyObject* pBioObject, patError*& err ) {
  PyObject* pDraws(NULL) ;
  PyObject* pKeyList(NULL) ;
  map<patString, bioRandomDraws::bioDraw >* drawsDic = new map<patString, bioRandomDraws::bioDraw > ;
  
  GET_PY_OBJ(pBioObject, DRAWS, pDraws) ;
  pKeyList = PyDict_Keys(pDraws) ;

  Py_ssize_t listSize = PyList_Size(pKeyList) ;
  for (Py_ssize_t pos=0 ; pos<listSize ; ++pos) {
    PyObject* pName = PyList_GetItem(pKeyList, pos) ;
    PyObject* pTupleDraw = PyDict_GetItem(pDraws, pName) ;
    
    patString drawName = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pName))) ;
    Py_ssize_t tupleSize =  PyTuple_Size(pTupleDraw) ;

    patString defaultHeader = bioParameters::the()->getValueString(patString("HeaderOfRowId"),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    patString theDrawId ;
    patString theDrawType ;
    PyObject* drawType(NULL); 
    switch (tupleSize) {
    case -1:
      {
	// If there is only one element, and there is no trailing comma,
	// Python does not recognize it as a tuple. Then it is just a
	// string.
	theDrawType = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pTupleDraw))) ;
	theDrawId = defaultHeader ;
	break ;
      }
    case 1:
      {
      drawType = PyTuple_GetItem(pTupleDraw, 0) ;
      if (drawType == NULL) {
	err = new patErrNullPointer("PyObject") ;
	WARNING(err->describe()) ;
	return drawsDic;
      }
      theDrawType = patString(PyBytes_AsString(PyUnicode_AsASCIIString(drawType))) ;
      theDrawId = defaultHeader ;
      break ;
      }
    case 2:
      {
      drawType = PyTuple_GetItem(pTupleDraw, 0) ;
      if (drawType == NULL) {
	err = new patErrNullPointer("PyObject") ;
	WARNING(err->describe()) ;
	return drawsDic;
      }
      theDrawType = patString(PyBytes_AsString(PyUnicode_AsASCIIString(drawType))) ;
      PyObject* drawId = PyTuple_GetItem(pTupleDraw, 1) ;
      if (drawId == NULL) {
	err = new patErrNullPointer("PyObject") ;
	WARNING(err->describe()) ;
	return drawsDic;
      }
      theDrawId = patString(PyBytes_AsString(PyUnicode_AsASCIIString(drawId))) ;
      break ;
      }
    default:
      {
      stringstream str ;
      str << "Two strings are required for the definition of the draw " << drawName << ". One for the type of draw, ond one for the ID identifying the observations or individuals. If only one string is provided, the second one is assumed to be '" << defaultHeader << "'" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return drawsDic;
      }
    }
    
    bioRandomDraws::bioDrawsType dt = bioRandomDraws::UNDEFINED ;
    if (theDrawType == "NORMAL") {
      dt = bioRandomDraws::NORMAL ;
    }
    else if (theDrawType == "TRUNCNORMAL") {
      dt = bioRandomDraws::TRUNCNORMAL ;
    }
    else if (theDrawType == "UNIFORM") {
      dt = bioRandomDraws::UNIFORM ;
    }
    else if (theDrawType == "UNIFORMSYM") {
      dt = bioRandomDraws::UNIFORMSYM ;
    }
    else {
      stringstream str ;
      str << "Type " << theDrawType << " is unknown. Valid types are: NORMAL, TRUNCNORMAL, UNIFORM and UNIFORMSYM" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    bioRandomDraws::bioDraw theDraw(drawName,dt,theDrawId) ;
    //      DEBUG_MESSAGE(theDraw)
    // //DEBUG_MESSAGE("Build expression for " << statName) ;
    // bioExpression *expr = buildExpression(pStat,err) ;
    // if (err != NULL) {
    //   WARNING(err->describe()) ;
    //   return NULL ;
    // }
    drawsDic->insert(pair<patString,bioRandomDraws::bioDraw>(drawName,theDraw)) ;
  }
  return drawsDic ;
}




bioArithPrint* bioModelParser::getSimulation(PyObject* pBioObject, patError*& err ) {
  bioExpression* theFormula ;

  PyObject* pFormula(NULL) ;

  GET_PY_OBJ(pBioObject, SIMULATE, pFormula) ;
  if (pFormula != NULL) {
    if (pFormula != Py_None) {
      theFormula = buildExpression(pFormula,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
      
      if (!theFormula->isSimulator()) {
	err = new patErrMiscError("Expression not adequate for simulation. Requires a Enumerate statement.") ;
	WARNING(err->describe()) ;
	return NULL ;
      }
      return (bioArithPrint*)theFormula ;
    }
    else {
      //      DEBUG_MESSAGE("No formula (None) for the simulation has been provided") ;
      return NULL ;

    }
  }
  else {
    //    DEBUG_MESSAGE("No formula for the simulation has been provided") ;
    return NULL ;
  }
}

// patBoolean bioModelParser::getStan(PyObject* pBioObject,
// 				   patError*& err ) {
  
//   err = new patErrMiscError("To beimplemented") ;
//   WARNING(err->describe()) ;

  // PyObject* pStanObject(NULL) ;

  // GET_PY_OBJ(pBioObject, STAN, pStanObject) ;
  // if (pStanObject != NULL) {
  //   PyObject* pCategorical ;
  //   GET_PY_OBJ(pStanObject, "CATEGORICAL", pCategorical) ;
  //   PyObject* pKeyList = PyDict_Keys(pCategorical) ;
  //   Py_ssize_t listSize = PyList_Size(pKeyList) ;
  //   for (Py_ssize_t pos=0 ; pos<listSize ; ++pos) {
  //     PyObject* pName = PyList_GetItem(pKeyList, pos) ;
  //     PyObject* pCat = PyDict_GetItem(pCategorical, pName) ;
      
  //     patString theName = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pName))) ;
  //     DEBUG_MESSAGE("Categorical: " << theName) ;
  //     Py_ssize_t tupleSize =  PyTuple_Size(pCat) ;
  //     DEBUG_MESSAGE("Tuple size: " << tupleSize) ;
  //     if (tupleSize != 2) {
  // 	err = new patErrMiscError("The syntax for CATEGORICAL requires a tuple wth two elemens: the endogenous variable amd the model") ;
  // 	WARNING(err->describe()) ;
  // 	return patFALSE ;
  //     }
  //     PyObject* endogVariable = PyTuple_GetItem(pCat,0) ;
  //     DEBUG_MESSAGE("Build expression for endog variable") ;
  //     bioExpression* theEndogVariable = buildExpression(endogVariable,err) ;
  //     if (err != NULL) {
  // 	WARNING(err->describe()) ;
  // 	return patFALSE ;
  //     }
  //     DEBUG_MESSAGE("Endog variable: " << *theEndogVariable) ;
  //     PyObject* modelDict = PyTuple_GetItem(pCat,1) ;
  //     pKeyList = PyDict_Keys(modelDict) ;
  //     Py_ssize_t listSize = PyList_Size(pKeyList) ;
  //     for (Py_ssize_t pos = 0 ; pos < listSize ; ++pos) {
  // 	PyObject* pName = PyList_GetItem(pKeyList, pos) ;
  // 	PyObject* pExpr = PyDict_GetItem(modelDict, pName) ;
	
  // 	patString constraintName = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pName))) ;
  // 	bioExpression *expr = buildExpression(pConstraint,err) ;
  // 	if (err != NULL) {
  // 	  WARNING(err->describe()) ;
  // 	  return NULL ;
  // 	}
  //     }
  //   }
  // }
//   return patFALSE ;
  
// }

bioArithBayes* bioModelParser::getBayesian(PyObject* pBioObject,
					   patError*& err ) {

  bioExpression* theFormula ;

  PyObject* pFormula(NULL) ;

  GET_PY_OBJ(pBioObject, BAYESIAN, pFormula) ;
  if (pFormula != NULL) {
    if (pFormula != Py_None) {
      theFormula = buildExpression(pFormula,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
      //      DEBUG_MESSAGE("Formula: " << *theFormula) ;
      
      if (!theFormula->isBayesian()) {
	err = new patErrMiscError("Expression not adequate for Bayesian estimation. Requires a MH statement.") ;
	WARNING(err->describe()) ;
	return NULL ;
      }
      return (bioArithBayes*)theFormula ;
    }
    else {
      //      DEBUG_MESSAGE("No formula (None) for the Bayesian estimation has been provided") ;
      return NULL ;

    }
  }
  else {
    //    DEBUG_MESSAGE("No formula for the simulation has been provided") ;
    return NULL ;
  }
}


map<patString, patULong>* bioModelParser::getConstraints(PyObject* pBioObject, patError*& err) {
  PyObject* pConstraints(NULL) ;
  PyObject* pKeyList(NULL) ;
  map<patString,patULong>* constraintsDic = new map<patString,patULong> ;
  
  GET_PY_OBJ(pBioObject, CONSTRAINTS, pConstraints) ;
  pKeyList = PyDict_Keys(pConstraints) ;

  Py_ssize_t listSize = PyList_Size(pKeyList) ;
  for (Py_ssize_t pos=0 ; pos<listSize ; ++pos) {
    PyObject* pName = PyList_GetItem(pKeyList, pos) ;
    PyObject* pConstraint = PyDict_GetItem(pConstraints, pName) ;

    patString constraintName = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pName))) ;
    bioExpression *expr = buildExpression(pConstraint,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    constraintsDic->insert(pair<patString,patULong>(constraintName, expr->getId())) ;
  }
  return constraintsDic ;
}


map<patString, patString>* bioModelParser::getParameters(PyObject* pBioObject, patError*& err) {
  PyObject* pParameters(NULL) ;
  PyObject* pKeyList(NULL) ;
  map<patString,patString>* parametersDic = new map<patString,patString> ;
  
  GET_PY_OBJ(pBioObject, PARAMETERS, pParameters) ;
  pKeyList = PyDict_Keys(pParameters) ;

  Py_ssize_t listSize = PyList_Size(pKeyList) ;
  for (Py_ssize_t pos=0 ; pos<listSize ; ++pos) {
    PyObject* pName = PyList_GetItem(pKeyList, pos) ;
    PyObject* pParam = PyDict_GetItem(pParameters, pName) ;

    patString paramName = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pName))) ;

    if (!PyUnicode_Check(pParam)) {
      stringstream str ;
      str << "Parameters must be specified as strings. It is not the case for parameter " << paramName ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return NULL;
    }

    patString paramValue = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pParam))) ;
    parametersDic->insert(pair<patString,patString>(paramName, paramValue)) ;
  }
  return parametersDic ;
}


void bioModelParser::buildIteratorInfo(PyObject* pModule, patError*& err) {
  
  PyObject *pIteratorList(NULL) ;
  GET_PY_OBJ(pModule, ITERATOR_LIST, pIteratorList) ;
  int li = PyList_Size(pIteratorList) ;

  //  DEBUG_MESSAGE("Get infos for " << li << " iterators") ;
  // Process each iterator in the list
  for (int pos=0 ; pos<li ; ++pos) {

    bioIteratorInfo *iterInfo ; 
    PyObject *pIteratorName(NULL), *pColName(NULL), *pTypeIter(NULL) ;
    patString iteratorName, colName, typeIter ;
    PyObject* pIterator(NULL) ;

    pIterator = PyList_GetItem(pIteratorList, pos) ;
    if (PY_FAIL(pIterator)) {
      PY_ERR_PRINT() ;
      err = new patErrNullPointer("pIterator") ;
      WARNING(err->describe());
      return ;
    }

    //DEBUG_MESSAGE("Get Iterators " << ITERATOR_NAME) ;
    GET_PY_OBJ(pIterator, ITERATOR_NAME, pIteratorName ) ;

    iteratorName = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pIteratorName))) ;

    GET_PY_OBJ(pIterator, INDEX_VARIABLE, pColName) ;
    colName = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pColName))) ;

    GET_PY_OBJ(pIterator, ITERATOR_TYPE, pTypeIter) ;
    typeIter = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pTypeIter))) ;

    //    DEBUG_MESSAGE("[" << pos << "] Iterator " << typeIter << "[" << colName << "]") ;

    if (typeIter == patString(DRAW_TYPE)) {
      PyObject* pChild(NULL) ;
      GET_PY_OBJ(pIterator, CHILD, pChild) ;
      patString childIteratorName(PyBytes_AsString(PyUnicode_AsASCIIString(pChild))) ;
      PyObject* pName(NULL) ;
      GET_PY_OBJ(pIterator, STRUCTURE_NAME, pName) ;  // pName can be a filename or an Iterator object

      if (PyUnicode_Check(pName)) {   // Iterator
        patString iteratorParentName(PyBytes_AsString(PyUnicode_AsASCIIString(pName))) ;
	if (iteratorParentName == "__dataFile__") {
	  iterInfo = new bioIteratorInfo(theDataFile,patString(""),childIteratorName, colName, DRAW) ;
	  //	  DEBUG_MESSAGE("iterInfo created: " << iterInfo->getInfo()) ;
	}
	else {
	  stringstream str ;
	  str << "A draw iterator must span the whole file. It cannot depend on iterator " << iteratorParentName ;
	  err = new patErrMiscError(str.str()) ;
	  WARNING(err->describe()) ;
	  return ;
	} 
      }
      else {  // Iterator
	err = new patErrMiscError("Argument of drawIterator must be a string") ;
	WARNING(err->describe()) ;
	return ;
      }
      //      DEBUG_MESSAGE("*** Adding to the repository: " << *iterInfo) ;
      bioIteratorInfoRepository::the()->addIteratorInfo(iteratorName, *iterInfo, err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
    else if (typeIter == patString(META_TYPE)) {
      //      DEBUG_MESSAGE("[" << pos << "] Build MetaIterator ");
      PyObject* pChild(NULL) ;
      PyObject* pName(NULL) ;

      GET_PY_OBJ(pIterator, CHILD, pChild) ;
      patString childIteratorName(PyBytes_AsString(PyUnicode_AsASCIIString(pChild))) ;
      GET_PY_OBJ(pIterator, STRUCTURE_NAME, pName) ;  // pName may be a filename or an Iterator

      if (PyUnicode_Check(pName)) {  // Iterator
        patString iteratorParentName(PyBytes_AsString(PyUnicode_AsASCIIString(pName))) ;
	if (iteratorParentName == "__dataFile__") {
	  iterInfo = new bioIteratorInfo(theDataFile,patString(""),childIteratorName, colName, META) ;
	  //	  DEBUG_MESSAGE("iterInfo created: " << iterInfo->getInfo()) ;
	}
	else {
	  iterInfo = new bioIteratorInfo(patString(""),iteratorParentName, childIteratorName, colName, META) ;
	}
      } 
      else { 
	err = new patErrMiscError("Argument of metaIterator must be a string") ;
	WARNING(err->describe()) ;
	return ;
      }
      //      DEBUG_MESSAGE("*** Adding to the repository: " << *iterInfo) ;
      bioIteratorInfoRepository::the()->addIteratorInfo(iteratorName, *iterInfo,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
    else if (typeIter == patString(ROW_TYPE)) {
      //DEBUG_MESSAGE("[" << pos << "] Build RowIterator ");
      PyObject* pChild(NULL) ;
      GET_PY_OBJ(pIterator, CHILD, pChild) ;
      patString childIteratorName(PyBytes_AsString(PyUnicode_AsASCIIString(pChild))) ;
      PyObject* pName(NULL) ;
      GET_PY_OBJ(pIterator, STRUCTURE_NAME, pName) ;  // pName can be a filename or an Iterator object

      if (PyUnicode_Check(pName)) {   // Iterator
        patString iteratorParentName(PyBytes_AsString(PyUnicode_AsASCIIString(pName))) ;
	if (iteratorParentName == "__dataFile__") {
	  iterInfo = new bioIteratorInfo(theDataFile,patString(""),childIteratorName, colName, ROW) ;
	  //	  DEBUG_MESSAGE("iterInfo created: " << iterInfo->getInfo()) ;
	}
	else {
	  iterInfo = new bioIteratorInfo(patString(""),iteratorParentName,childIteratorName, colName, ROW) ;
	  //	  DEBUG_MESSAGE("iterInfo created: " << iterInfo->getInfo()) ;
	} 
      }
      else {  // Iterator
	err = new patErrMiscError("Argument of rowIterator must be a string") ;
	WARNING(err->describe()) ;
	return ;
      }
      //      DEBUG_MESSAGE("*** Adding to the repository: " << *iterInfo) ;
      bioIteratorInfoRepository::the()->addIteratorInfo(iteratorName, *iterInfo, err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
  }
}


// Build the corresponding bioExpression object from the object "Expression" (in the python module)
bioExpression* bioModelParser::buildExpression(PyObject *pExpression, patError*& err) {
 
  //get ID from python object

  if (pExpression == NULL) {
    err = new patErrNullPointer("PyObject") ;
    WARNING(err->describe()) ;
    return NULL ;
  }

  PyObject* id = PyObject_CallMethod(pExpression, "getID", NULL);
  patString pyID("");

  if(id != NULL){
    PyObject* exprAsStr = PyObject_Str(id);

    PyObject* pyStr = PyUnicode_AsEncodedString(exprAsStr, "utf-8","Error ~");
    const char *str = PyBytes_AS_STRING(pyStr);
    patString s(str);
    Py_XDECREF(id);
    Py_XDECREF(exprAsStr);
    Py_XDECREF(pyStr);

    //if the expression has a specific ID
    if(s.substr(s.find("-")+1,-1).compare("no ID") != 0){

      pyID = s;

      
      //check if the expression is already known
      bioExpression* exp = theExpRepository->getExpressionByPythonID(pyID);
      if(exp != NULL){
	//        fprintf(stdout, "already there : %s\n", pyID.substr(0,40).c_str());
        return exp;
      }else{  
	//        fprintf(stdout, "new thing in map : %s\n", pyID.substr(0,40).c_str());
      }
    }
  }


  bioExpression* e ;


  

  PyObject *pIndexType(NULL) ;
  int indexType ;
  GET_PY_OBJ(pExpression, OPERATOR_INDEX, pIndexType) ;
  // patString theString(OPERATOR_INDEX) ;
  // pIndexType = PyObject_GetAttrString(pExpression, theString.c_str()); 
  // if (PY_FAIL(pIndexType)) { 
  //   PY_ERR_PRINT() ;
  //   FATAL("Unable to get the object " << OPERATOR_INDEX) ;
  // }
  if (pIndexType == NULL) {
    err = new patErrNullPointer("PyObject") ;
    WARNING(err->describe()) ;
    return(NULL) ;
  }
  indexType = PyLong_AsLong(pIndexType) ;

  //  DEBUG_MESSAGE("Index type = " << indexType) ;

  // Numeric: 0
  if (indexType==operators.indexByNames[OP_NUM]) {

    PyObject *pValue(NULL) ;
    GET_PY_OBJ(pExpression, NUMERIC_VALUE, pValue) ;
    if (pValue == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patReal val(PyFloat_AsDouble(pValue)) ;
    bioArithConstant* constantNode = new bioArithConstant(theExpRepository,patBadId,val) ;
    if (constantNode == NULL) {
      err = new patErrNullPointer("bioArithConstant") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    
    e =  constantNode;
  
  }

  // Variable
  else if (indexType==operators.indexByNames[OP_VAR]) { 
    bioArithVariable *variable ;
    PyObject *pVariableName(NULL) ;
    GET_PY_OBJ(pExpression, VARIABLE_NAME, pVariableName) ;
    if (pVariableName == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patString name = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pVariableName))) ;

    pair<patULong,patULong> variableIds = theLiteralRepository->getVariable(name,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    variable = new bioArithVariable(theExpRepository,patBadId, variableIds.first,variableIds.second) ;
    if (variable == NULL) {
      err = new patErrNullPointer("bioArithVariable") ;
      WARNING(err->describe()) ;
      return NULL ;
      
    }

    e =  variable;
  }

  // Random variable Exactly the same as a regular variable, except
  // that it does not have to be declared by an expression ior in the
  // data file. It is used only for integration.
  else if (indexType==operators.indexByNames[OP_RV]) { 
    bioArithRandomVariable *variable ;
    PyObject *pVariableName(NULL) ;
    GET_PY_OBJ(pExpression, VARIABLE_NAME, pVariableName) ;
    if (pVariableName == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patString name = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pVariableName))) ;
    pair<patULong,patULong> ids = theLiteralRepository->getRandomVariable(name, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    // The literal does not exist yet. It must be added in the literal repository.
    if (ids.first == patBadId) {
      ids = theLiteralRepository->addRandomVariable(name,err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
    }
    variable = new bioArithRandomVariable(theExpRepository,patBadId, ids.first, ids.second) ;
    if (variable == NULL) {
      err = new patErrNullPointer("bioArithRamdomVariable") ;
      WARNING(err->describe()) ;
      return NULL ;
    }

    e =  variable;
  }

  else if (indexType==operators.indexByNames[OP_DRAWS]) { // Draws for Monte Carlo
    PyObject *pVariableName(NULL) ;

    GET_PY_OBJ(pExpression, DRAWS_NAME, pVariableName) ;
    if (pVariableName == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patString name = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pVariableName))) ;

    patULong id = bioRandomDraws::the()->getColId(name,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    patString index = bioRandomDraws::the()->getIndex(id,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    bioRandomDraws::bioDrawsType theType = bioRandomDraws::the()->getType(id,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }


    pair<patULong,patULong> theIndexIds =  bioLiteralRepository::the()->getVariable(index,err)  ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    

    bioArithVariable* theIndex = new bioArithVariable(theExpRepository,patBadId,theIndexIds.first, theIndexIds.second) ;
    if (theIndex == NULL) {
      err = new patErrNullPointer("bioArithVariable") ;
      WARNING(err->describe()) ;
      return NULL ;
    }

    bioArithRandom* theDraws = new bioArithRandom(theExpRepository,
						  patBadId, 
						  theIndex->getId(), 
						  id,
						  theType,
						  err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (theDraws == NULL) {
      err = new patErrNullPointer("bioArithRandom") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    e =  theDraws;

  }
  else if (indexType==operators.indexByNames[OP_UNIFDRAWS]) { // Uniform draws recyled from Monte Carlo
    bioParameters::the()->setValueInt("saveUniformDraws",1,err) ; 
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    PyObject *pVariableName(NULL) ;

    GET_PY_OBJ(pExpression, DRAWS_NAME, pVariableName) ;
    if (pVariableName == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patString name = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pVariableName))) ;

    patULong id = bioRandomDraws::the()->getColId(name,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    patString index = bioRandomDraws::the()->getIndex(id,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    bioRandomDraws::bioDrawsType theType = bioRandomDraws::the()->getType(id,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    pair<patULong,patULong> theIndexIds =  bioLiteralRepository::the()->getVariable(index,err)  ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    

    bioArithVariable* theIndex = new bioArithVariable(theExpRepository,patBadId,theIndexIds.first, theIndexIds.second) ;
    if (theIndex == NULL) {
      err = new patErrNullPointer("bioArithVariable") ;
      WARNING(err->describe()) ;
      return NULL ;
    }

    bioArithUnifRandom* theDraws = new bioArithUnifRandom(theExpRepository,
							  patBadId, 
							  theIndex->getId(), 
							  id,
							  theType,
							  err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (theDraws == NULL) {
      err = new patErrNullPointer("bioArithUnifRandom") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    e =  theDraws;

  }
  // Normal Random number
  else if (indexType==operators.indexByNames[OP_NORMAL]) { 
    err = new patErrMiscError("Deprecated syntax. Use bioDraws instead, together with the section DRAWS of the BIOGEME_OBJECT") ;
    WARNING(err->describe()) ;
    return NULL ;
    PyObject *pVariableName(NULL) ;
    PyObject *pIndexName(NULL) ;

    GET_PY_OBJ(pExpression, RAND_NORMAL_NAME, pVariableName) ;
    if (pVariableName == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patString name = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pVariableName))) ;
    GET_PY_OBJ(pExpression, RAND_NORMAL_INDEX, pIndexName) ;
    if (pIndexName == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patString index = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pIndexName))) ;

    patULong id = bioRandomDraws::the()->addRandomVariable(name,
							   bioRandomDraws::NORMAL,index,NULL,patString(""),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    pair<patULong,patULong> theIndexIds =  bioLiteralRepository::the()->getVariable(index,err)  ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    

    bioArithVariable* theIndex = new bioArithVariable(theExpRepository,patBadId,theIndexIds.first, theIndexIds.second) ;
    if (theIndex == NULL) {
      err = new patErrNullPointer("bioArithVariable") ;
      WARNING(err->describe()) ;
      return NULL ;
    }

    bioArithRandom* theNormal = new bioArithRandom(theExpRepository,
						   patBadId, 
						   theIndex->getId(), 
						   id,
						   bioRandomDraws::NORMAL,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (theNormal == NULL) {
      err = new patErrNullPointer("bioArithRandom") ;
      WARNING(err->describe()) ;
      return NULL ;
    }

    e =  theNormal;
  }

  // Uniform Random number
  else if (indexType==operators.indexByNames[OP_UNIFORM]) { 
    err = new patErrMiscError("Deprecated syntax. Use bioDraws instead, together with the section DRAWS of the BIOGEME_OBJECT") ;
    WARNING(err->describe()) ;
    return NULL ;
    PyObject *pVariableName(NULL) ;
    PyObject *pIndexName(NULL) ;

    GET_PY_OBJ(pExpression, RAND_UNIFORM_NAME, pVariableName) ;
    if (pVariableName == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patString name = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pVariableName))) ;
    GET_PY_OBJ(pExpression, RAND_UNIFORM_INDEX, pIndexName) ;
    if (pIndexName == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patString index = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pIndexName))) ;

    patULong id = bioRandomDraws::the()->addRandomVariable(name,bioRandomDraws::UNIFORM,index,NULL,patString(""),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    pair<patULong,patULong> theIndexIds =  bioLiteralRepository::the()->getVariable(index,err)  ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    bioArithVariable* theIndex = new bioArithVariable(theExpRepository,
						      patBadId,
						      theIndexIds.first, 
						      theIndexIds.second) ;

    if (theIndex == NULL) {
      err = new patErrNullPointer("bioArithVariable") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    bioArithRandom* theUniform = new bioArithRandom(theExpRepository,
						    patBadId, 
						    theIndex->getId(), 
						    id,
						    bioRandomDraws::UNIFORM,
						    err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (theUniform == NULL) {
      err = new patErrNullPointer("bioArithRandom") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    
    e =  theUniform;
  }
  // Uniform Random number
  else if (indexType==operators.indexByNames[OP_UNIFORMSYM]) { 
    err = new patErrMiscError("Deprecated syntax. Use bioDraws instead, together with the section DRAWS of the BIOGEME_OBJECT") ;
    WARNING(err->describe()) ;
    return NULL ;
    PyObject *pVariableName(NULL) ;
    PyObject *pIndexName(NULL) ;

    GET_PY_OBJ(pExpression, RAND_UNIFORMSYM_NAME, pVariableName) ;
    if (pVariableName == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patString name = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pVariableName))) ;
    GET_PY_OBJ(pExpression, RAND_UNIFORMSYM_INDEX, pIndexName) ;
    if (pIndexName == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patString index = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pIndexName))) ;

    patULong id = bioRandomDraws::the()->addRandomVariable(name,bioRandomDraws::UNIFORMSYM,index,NULL,patString(""),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    pair<patULong,patULong> theIndexIds =  bioLiteralRepository::the()->getVariable(index,err)  ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    bioArithVariable* theIndex = new bioArithVariable(theExpRepository,
						      patBadId,
						      theIndexIds.first, 
						      theIndexIds.second) ;


    if (theIndex == NULL) {
      err = new patErrNullPointer("bioArithVariable") ;
      WARNING(err->describe()) ;
      return NULL ;
    }

    bioArithRandom* theUniformSym = new bioArithRandom(theExpRepository,
						       patBadId, 
						       theIndex->getId(),
						       id,
						       bioRandomDraws::UNIFORMSYM,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (theUniformSym == NULL) {
      err = new patErrNullPointer("bioArithRandom") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    e =  theUniformSym ;
  }
  // user expressions
  else if (indexType == operators.indexByNames[OP_USEREXPR]) { 
    PyObject *pName(NULL) ;
    GET_PY_OBJ(pExpression, "name", pName) ;
    if (pName == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    patString name = 
      patString(PyBytes_AsString(PyUnicode_AsASCIIString(pName))) ;
    PyObject *pExpr(NULL) ;
    GET_PY_OBJ(pExpression, "expression", pExpr) ;
    if (pExpr == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    bioExpression* theUserExpression = buildExpression(pExpr,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    pair<patBoolean,pair<patULong,patULong> > variableIds = theLiteralRepository->addUserExpression(name,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    bioArithVariable* variable = new bioArithVariable(theExpRepository,patBadId, variableIds.second.first,variableIds.second.second) ;
    if (variable == NULL) {
      err = new patErrNullPointer("bioArithVariable") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (variableIds.first) {
      theModel->addUserExpression(name,theUserExpression->getId()) ;
    }
    e =  variable ;
 
  }
  // user draws
  else if (indexType == operators.indexByNames[OP_USERDRAWS]) { 
    PyObject *pName(NULL) ;
    GET_PY_OBJ(pExpression, "name", pName) ;
    if (pName == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    patString name = 
      patString(PyBytes_AsString(PyUnicode_AsASCIIString(pName))) ;
    PyObject *pExpr(NULL) ;
    GET_PY_OBJ(pExpression, "expression", pExpr) ;
    if (pExpr == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    bioExpression* theUserExpression = buildExpression(pExpr,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    
    PyObject *pIteratorInfo(NULL);
    GET_PY_OBJ(pExpression, "iteratorName", pIteratorInfo) ;
    if (pIteratorInfo == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patString iteratorName = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pIteratorInfo))) ;


    patULong colId = bioRandomDraws::the()->addUserDraws(name,theUserExpression,iteratorName,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    patString index = bioRandomDraws::the()->getIndex(colId,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    
    pair<patULong,patULong> theIndexIds =  bioLiteralRepository::the()->getVariable(index,err)  ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    

    bioArithVariable* theIndex = new bioArithVariable(theExpRepository,patBadId,theIndexIds.first, theIndexIds.second) ;
    if (theIndex == NULL) {
      err = new patErrNullPointer("bioArithVariable") ;
      WARNING(err->describe()) ;
      return NULL ;
    }

    bioArithRandom* theDraws = new bioArithRandom(theExpRepository,
						  patBadId, 
						  theIndex->getId(), 
						  colId,
						  bioRandomDraws::USERDEFINED,
						  err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (theDraws == NULL) {
      err = new patErrNullPointer("bioArithRandom") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    e =  theDraws;
  }
  // Parameters (Beta)
  else if (indexType==operators.indexByNames[OP_PARAM]) { 
    PyObject *pParamName(NULL) ;
    PyObject *pLowerBound(NULL) ;
    PyObject *pUpperBound(NULL) ;
    PyObject *pFixed(NULL) ;
    PyObject *pCurVal(NULL) ;
    PyObject *pDesc(NULL) ;

    GET_PY_OBJ(pExpression, "name", pParamName) ;
    if (pParamName == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patString name = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pParamName))) ;

    GET_PY_OBJ(pExpression,"lb", pLowerBound) ;
    if (pLowerBound == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patReal lb = PyFloat_AsDouble(pLowerBound) ;
    
    GET_PY_OBJ(pExpression, "ub", pUpperBound) ;
    if (pUpperBound == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patReal ub = PyFloat_AsDouble(pUpperBound) ;
    
    GET_PY_OBJ(pExpression, "st", pFixed) ;
    if (pFixed == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patBoolean fixed = patBoolean(PyLong_AsLong(pFixed)==1) ;

    GET_PY_OBJ(pExpression, "val", pCurVal) ;
    if (pCurVal == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patReal val = PyFloat_AsDouble(pCurVal) ;

    GET_PY_OBJ(pExpression, "desc", pDesc) ;
    if (pDesc == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patString desc = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pDesc))) ;
    
    pair<patULong,patULong> paramIds = theLiteralRepository->addFixedParameter(name, val, lb, ub, fixed, desc, err) ;

    bioArithFixedParameter *beta = new bioArithFixedParameter(theExpRepository,
							      patBadId, 
							      paramIds.first, 
							      paramIds.second) ;

    if (beta == NULL) {
      err = new patErrNullPointer("bioArithFixedParameter") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    
    e =  beta;
  }

  else if ((indexType>=operators.minBinopIndex) && (indexType<=operators.maxBinopIndex)) { // Bin Op
    bioExpression *binOpExpr ;
    PyObject *pLeft(NULL) ;

    GET_PY_OBJ(pExpression, "left", pLeft) ;
    if (pLeft == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    bioExpression *left = buildExpression(pLeft,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (left == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    PyObject *pRight(NULL) ;
    GET_PY_OBJ(pExpression, "right", pRight) ;
    if (pRight == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    
    bioExpression *right = buildExpression(pRight,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (right == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return NULL ;
    }


    
    if (indexType==operators.indexByNames[OP_ADD]) { // add
      binOpExpr = new bioArithBinaryPlus(theExpRepository,patBadId, left->getId(), right->getId(),err) ; 
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (binOpExpr == NULL) {
        err = new patErrNullPointer("bioArithBinaryPlus") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    }
    else if (indexType==operators.indexByNames[OP_SUB]) { // sub
      binOpExpr = new bioArithBinaryMinus(theExpRepository,patBadId, left->getId(), right->getId(),err) ; 
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (binOpExpr == NULL) {
        err = new patErrNullPointer("bioArithBinaryMinus") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    }
    else if (indexType==operators.indexByNames[OP_MUL]) { // mul
      binOpExpr = new bioArithMult(theExpRepository,patBadId, left->getId(), right->getId(),err) ; 
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (binOpExpr == NULL) {
        err = new patErrNullPointer("bioArithMult") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    } 
    else if (indexType==operators.indexByNames[OP_DIV]) { // div
      binOpExpr = new bioArithDivide(theExpRepository,patBadId, left->getId(), right->getId(),err) ; 
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (binOpExpr == NULL) {
        err = new patErrNullPointer("bioArithDivide") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    }
    else if (indexType==operators.indexByNames[OP_POW]) { // power
      binOpExpr = new bioArithPower(theExpRepository,patBadId, left->getId(), right->getId(),err) ; 
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (binOpExpr == NULL) {
        err = new patErrNullPointer("bioArithPower") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
      
    }
    else if (indexType==operators.indexByNames[OP_AND]) { // and
      binOpExpr = new bioArithAnd(theExpRepository,patBadId, left->getId(), right->getId(),err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (binOpExpr == NULL) {
        err = new patErrNullPointer("bioArithAnd") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    }
    else if (indexType==operators.indexByNames[OP_OR]) { // or
      binOpExpr = new bioArithOr(theExpRepository,patBadId, left->getId(), right->getId(),err) ; 
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (binOpExpr == NULL) {
        err = new patErrNullPointer("bioArithOr") ;
        WARNING(err->describe()) ;
        return NULL ;
      }

    } 
    else if (indexType==operators.indexByNames[OP_EQ]) { // ==
      binOpExpr = new bioArithEqual(theExpRepository,patBadId, left->getId(), right->getId(),err) ; 
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (binOpExpr == NULL) {
        err = new patErrNullPointer("bioArithEqual") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    } 
    else if (indexType==operators.indexByNames[OP_NEQ]) { // <>
      binOpExpr = new bioArithNotEqual(theExpRepository,patBadId, left->getId(), right->getId(),err) ; 
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (binOpExpr == NULL) {
        err = new patErrNullPointer("bioArithNotEqual") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    } 
    else if (indexType==operators.indexByNames[OP_GT]) { // >
      binOpExpr = new bioArithGreater(theExpRepository,patBadId, left->getId(), right->getId(),err) ; 
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (binOpExpr == NULL) {
        err = new patErrNullPointer("bioArithGreater") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    } 
    else if (indexType==operators.indexByNames[OP_GE]) { // >=
      binOpExpr = new bioArithGreaterEqual(theExpRepository,patBadId, left->getId(), right->getId(),err) ; 
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (binOpExpr == NULL) {
        err = new patErrNullPointer("bioArithGreaterEqual") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    } 
    else if (indexType==operators.indexByNames[OP_LT]) { // <
      binOpExpr = new bioArithLess(theExpRepository,patBadId, left->getId(), right->getId(),err) ; 
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (binOpExpr == NULL) {
        err = new patErrNullPointer("bioArithLess") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    } 
    else if (indexType==operators.indexByNames[OP_LE]) { // <=
      binOpExpr = new bioArithLessEqual(theExpRepository,patBadId, left->getId(), right->getId(),err) ; 
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (binOpExpr == NULL) {
        err = new patErrNullPointer("bioArithLessEqual") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    } 
    else if (indexType==operators.indexByNames[OP_MAX]) {
      binOpExpr = new bioArithMax(theExpRepository,patBadId,left->getId(),right->getId(),err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (binOpExpr == NULL) {
        err = new patErrNullPointer("bioArithMax") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    }
    else if (indexType==operators.indexByNames[OP_MIN]) {
      binOpExpr = new bioArithMin(theExpRepository,patBadId,left->getId(),right->getId(),err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (binOpExpr == NULL) {
        err = new patErrNullPointer("bioArithMin") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    }
    else if (indexType==operators.indexByNames[OP_MOD]) {
      err = new patErrMiscError("mod operator not yet implemented") ;
      WARNING(err->describe()) ;
      return NULL ;
      //      binOpExpr = new bioArithMod(patBadId,left->getId(),right->getId()) ;
    }
    else {
      stringstream str ;
      str << "Binary Operator not yet implemented. indexType [" << indexType << "]" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return NULL;
    }

    e =  binOpExpr;
  }
  else if ((indexType>=operators.minUnopIndex)&&(indexType<=operators.maxUnopIndex)) { // UnOp

    bioExpression *unOpExpr ;
    PyObject *pExp(NULL) ;
    GET_PY_OBJ(pExpression, "expression", pExp) ;
    if (pExp == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    bioExpression *exp = buildExpression(pExp,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    if (indexType==operators.indexByNames[OP_ABS]) { // abs
      unOpExpr = new bioArithAbs(theExpRepository,patBadId, exp->getId(),err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (unOpExpr == NULL) {
        err = new patErrNullPointer("bioArithAbs") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    }
    else if (indexType==operators.indexByNames[OP_NEG]) { // negOp (unaryMinus)
      unOpExpr = new bioArithUnaryMinus(theExpRepository,patBadId, exp->getId(),err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (unOpExpr == NULL) {
        err = new patErrNullPointer("bioArithUnaryMinus") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    }
    else if (indexType==operators.indexByNames[OP_MC]) { // Monte Carlo
      unOpExpr = new bioArithMonteCarlo(theExpRepository,patBadId, exp->getId(),err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (unOpExpr == NULL) {
        err = new patErrNullPointer("bioArithMonteCarlo") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    }
    else if (indexType==operators.indexByNames[OP_LOG]) { // Logarithm function
      unOpExpr = new bioArithLog(theExpRepository,patBadId, exp->getId(),err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (unOpExpr == NULL) {
        err = new patErrNullPointer("bioArithLog") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    }
    else if (indexType==operators.indexByNames[OP_NORMALCDF]) { // Normal CDF function
      unOpExpr = new bioArithNormalCdf(theExpRepository,patBadId, exp->getId(),err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (unOpExpr == NULL) {
        err = new patErrNullPointer("bioArithNormalCdf") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    }
    else if (indexType==operators.indexByNames[OP_EXP]) { // Exponential function
      unOpExpr = new bioArithExp(theExpRepository,patBadId, exp->getId(),err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (unOpExpr == NULL) {
        err = new patErrNullPointer("bioArithExp") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    }
    else {
      stringstream str ;
      str << "Unary Operator not yet implemented. indexType [" << indexType << "]" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return NULL;
    }

    e =  unOpExpr;  
  }
  else if (indexType==operators.indexByNames[OP_MCCV]) {
    PyObject* pExpr(NULL), *pIntegrand(NULL), *pIntegral(NULL) ;
    GET_PY_OBJ(pExpression, "expression", pExpr ) ;
    if (pExpr == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    bioExpression* expr = buildExpression(pExpr,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    GET_PY_OBJ(pExpression, "integrand", pIntegrand ) ;
    if (pIntegrand == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }

    bioExpression* integrand = buildExpression(pIntegrand,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    GET_PY_OBJ(pExpression, "integral", pIntegral ) ;
    if (pIntegral == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    bioExpression* integral = buildExpression(pIntegral,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    bioExpression* theExpression = new bioArithMonteCarlo(theExpRepository,
							  patBadId, 
							  expr->getId(),
							  integrand->getId(),
							  integral->getId(),
							  err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (theExpression == NULL) {
      err = new patErrNullPointer("bioArithMonteCarlo") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    e = theExpression ;


  }
  else if (indexType==operators.indexByNames[OP_INTEGRAL]) {
    PyObject* pExpr(NULL), *pVariable(NULL) ;
    GET_PY_OBJ(pExpression, "function", pExpr ) ;
    if (pExpr == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    bioExpression* expr = buildExpression(pExpr,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    GET_PY_OBJ(pExpression,"variable", pVariable) ;
    if (pVariable == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patString variableName = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pVariable))) ;
    bioExpression* theExpression = new bioArithGHIntegral(theExpRepository,patBadId, expr->getId(), variableName,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (theExpression == NULL) {
      err = new patErrNullPointer("bioArithGHIntegral") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    
    e =  theExpression;
  }
  else if (indexType==operators.indexByNames[OP_NORMALPDF]) {
    PyObject* pExpr(NULL) ;
    GET_PY_OBJ(pExpression, "function", pExpr ) ;
    if (pExpr == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    bioExpression* expr = buildExpression(pExpr,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    bioExpression* theExpression = new bioArithNormalPdf(theExpRepository,patBadId, expr->getId(), err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (theExpression == NULL) {
      err = new patErrNullPointer("bioArithNormalPdf") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    e =  theExpression ;
  }

  else if (indexType==operators.indexByNames[OP_DERIVATIVE]) {
    PyObject* pExpr(NULL), *pVariable(NULL) ;
    GET_PY_OBJ(pExpression, "function", pExpr ) ;
    if (pExpr == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    bioExpression* expr = buildExpression(pExpr,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    GET_PY_OBJ(pExpression,"variable", pVariable) ;
    if (pVariable == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patString variableName = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pVariable))) ;
    bioExpression* theExpression = new bioArithDerivative(theExpRepository,patBadId, expr->getId(), variableName,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (theExpression == NULL) {
      err = new patErrNullPointer("bioArithDerivative") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    e =  theExpression ;
  }
  else if ((indexType>=operators.minIteratoropIndex)&&
           (indexType<=operators.maxIteratoropIndex)) {
    bioExpression *theExpression ;
    PyObject *pExpr(NULL), *pIteratorInfo(NULL);
    bioExpression *expr ; 

    GET_PY_OBJ(pExpression, "function", pExpr ) ;
    if (pExpr == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    expr = buildExpression(pExpr,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    GET_PY_OBJ(pExpression, "iteratorName", pIteratorInfo) ;
    if (pIteratorInfo == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patString iteratorName = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pIteratorInfo))) ;
    if (indexType==operators.indexByNames[OP_SUM]) {
      //      DEBUG_MESSAGE("Iter info: " << *iterInfo) 
      patULong weightExpr = theModel->getWeightExpr() ;
      theExpression = new bioArithSum(theExpRepository,patBadId, expr->getId(), iteratorName,weightExpr,err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (theExpression == NULL) {
        err = new patErrNullPointer("bioArithSum") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    }
    else if (indexType==operators.indexByNames[OP_PROD]) {
      PyObject *pPositive(NULL) ;
      GET_PY_OBJ(pExpression, "positive", pPositive ) ;
      if (pPositive == NULL) {
        err = new patErrNullPointer("PyObject") ;
        WARNING(err->describe()) ;
        return(NULL) ;
      }
      int isPositive = PyLong_AsUnsignedLong(pPositive) ;
      //      DEBUG_MESSAGE("[bioModelParser::buildExpression] Create new bioArithProd") ;
      theExpression = new bioArithProd(theExpRepository,patBadId, expr->getId(), iteratorName,patBoolean(isPositive != 0),err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      if (theExpression == NULL) {
        err = new patErrNullPointer("bioArithProd") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    }
    else {
      stringstream str ;
      str << "Operator not yet implemented. indexType [" << indexType << "]" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return NULL;
    }
    e =  theExpression ;

  }
  else if (indexType == operators.indexByNames[OP_ENUM]) {
    PyObject* pTerm(NULL) ;
    PyObject* pKeyList(NULL) ;
    map<patString,patULong> theDict ;
    Py_ssize_t length ;
    

    DEBUG_MESSAGE("Build dictionary") ;
    GET_PY_OBJ(pExpression, "theDict", pTerm) ;
    if (pTerm == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    pKeyList = PyDict_Keys(pTerm) ;
    if (pKeyList == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe());
      return NULL ;
    }
    length = PyList_Size(pKeyList) ;
    for (Py_ssize_t pos=0 ; pos<length ; ++pos) {
      patString theKey ;
      PyObject* pKey = PyList_GetItem(pKeyList, pos) ;
      if (pKey == NULL) {
	err = new patErrNullPointer("PyObject") ;
	WARNING(err->describe()) ;
	return NULL;
      }
      PyObject* pExpression = PyDict_GetItem(pTerm, pKey) ;
      if (pExpression == NULL) {
	err = new patErrMiscError("Should never happen !!!") ;
	WARNING(err->describe()) ;
	return NULL ;
      }
      theKey = PyBytes_AsString(PyUnicode_AsASCIIString(pKey)) ;
      //      DEBUG_MESSAGE("key = " << theKey) ;
      bioExpression* dictEntry = buildExpression(pExpression,err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      theDict[theKey] = dictEntry->getId() ;
    }
    DEBUG_MESSAGE("Done") ;

    //    DEBUG_MESSAGE("ENUM: dictionary has been built") ;
    PyObject* pIteratorInfo(NULL) ;
    GET_PY_OBJ(pExpression, "iteratorName", pIteratorInfo) ;
    if (pIteratorInfo == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patString iteratorName = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pIteratorInfo))) ;
    bioExpression* theExpression = new bioArithPrint(theExpRepository,patBadId,theDict,iteratorName,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (theExpression == NULL) {
      err = new patErrNullPointer("bioArithPrint") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    e =  theExpression ;
  }
  else if (indexType==operators.indexByNames[OP_ELEM]) {
    PyObject* pProb(NULL) ;
    PyObject* pChoice(NULL) ;
    PyObject* pDefault(NULL) ;
    
    bioExpression* choice ;
    bioExpression* defExpression ;
    PyObject* pKeyList(NULL) ;
    map<patULong,patULong> theDict ;
    Py_ssize_t length ;
    bioExpression* theExpression = NULL ;
    
    GET_PY_OBJ(pExpression, "prob", pProb) ;
    if (pProb == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    pKeyList = PyDict_Keys(pProb) ;
    GET_PY_OBJ(pExpression, "default", pDefault) ;
    if (pDefault == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    defExpression = buildExpression(pDefault,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    GET_PY_OBJ(pExpression, "choice", pChoice) ;
    if (pChoice == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    choice = buildExpression(pChoice,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    length = PyList_Size(pKeyList) ;
    //    DEBUG_MESSAGE("Length = " << length) ;
    for (Py_ssize_t pos=0 ; pos<length ; ++pos) {
      patULong theKey ;
      PyObject* pKey = PyList_GetItem(pKeyList, pos) ;
      if (pKey == NULL) {
	err = new patErrNullPointer("PyObject") ;
	WARNING(err->describe()) ;
	return NULL ;
      }
      // DEBUG_MESSAGE("Key=" << PyLong_AsLong(pKey)) ;
      //       DEBUG_MESSAGE("Key = " << PyBytes_AsString(PyUnicode_AsASCIIString(PyObject)_Str(pKey)) );
      //       DEBUG_MESSAGE("Type Key = " << PyBytes_AsString(PyUnicode_AsASCIIString(PyObject)_Str(PyObject_Type(pKey))));

      //      if (PyLong_Check(pKey)) {
      PyObject* pExpression = PyDict_GetItem(pProb, pKey) ;
      if (pExpression == NULL) {
	err = new patErrMiscError("Should never happen !!!") ;
	WARNING(err->describe()) ;
	return NULL ;
      }
      theKey = PyLong_AsLong(pKey) ;
      bioExpression* expr = buildExpression(pExpression,err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      theDict[theKey] = expr->getId() ;
      //       }
      //       else {
      //         stringstream str ;
      //         str << "Problem with key: " << PyBytes_AsString(PyUnicode_AsASCIIString(pKey)) ;
      //         err = new patErrMiscError(str.str()) ;
      //         WARNING(err->describe()) ;
      //         return NULL;
      //       }
    }

    theExpression = new bioArithElem(theExpRepository,patBadId,choice->getId(),theDict,defExpression->getId(),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (theExpression == NULL) {
      err = new patErrNullPointer("bioArithElem") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    e =  theExpression ;
  }
  else if (indexType==operators.indexByNames[OP_LOGIT]) {
    PyObject* pProb(NULL) ;
    PyObject* pAv(NULL) ;
    PyObject* pChoice(NULL) ;
    
    bioExpression* choice ;
    PyObject* pKeyList(NULL) ;
    PyObject* pKeyListAv(NULL) ;
    map<patULong,patULong> theDict ;
    map<patULong,patULong> theAvDict ;
    Py_ssize_t length ;
    
    GET_PY_OBJ(pExpression, "prob", pProb) ;
    if (pProb == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    pKeyList = PyDict_Keys(pProb) ;

    GET_PY_OBJ(pExpression, "av", pAv) ;
    if (pAv == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    pKeyListAv = PyDict_Keys(pAv) ;

    GET_PY_OBJ(pExpression, "choice", pChoice) ;
    if (pChoice == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    choice = buildExpression(pChoice,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    length = PyList_Size(pKeyList) ;
    //    DEBUG_MESSAGE("Length = " << length) ;
    for (Py_ssize_t pos=0 ; pos<length ; ++pos) {
      patULong theKey ;
      PyObject* pKey = PyList_GetItem(pKeyList, pos) ;
      if (pKey == NULL) {
	err = new patErrNullPointer("PyObject") ;
	WARNING(err->describe()) ;
	return NULL ;
      }
      PyObject* pExpression = PyDict_GetItem(pProb, pKey) ;
      if (pExpression == NULL) {
	err = new patErrMiscError("Should never happen !!!") ;
	WARNING(err->describe()) ;
	return NULL ;
      }
      theKey = PyLong_AsLong(pKey) ;
      bioExpression* expr = buildExpression(pExpression,err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      theDict[theKey] = expr->getId() ;
    }

    length = PyList_Size(pKeyListAv) ;
    for (Py_ssize_t pos=0 ; pos<length ; ++pos) {
      patULong theKey ;
      PyObject* pKey = PyList_GetItem(pKeyListAv, pos) ;
      if (pKey == NULL) {
	err = new patErrNullPointer("PyObject") ;
	WARNING(err->describe()) ;
	return NULL ;
      }
      PyObject* pExpression = PyDict_GetItem(pAv, pKey) ;
      if (pExpression == NULL) {
	err = new patErrMiscError("Should never happen !!!") ;
	WARNING(err->describe()) ;
	return NULL ;
      }
      theKey = PyLong_AsLong(pKey) ;
      bioExpression* expr = buildExpression(pExpression,err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      theAvDict[theKey] = expr->getId() ;
    }
    bioExpression* theExpression = new bioArithLogLogit(theExpRepository,patBadId,choice->getId(),theDict,theAvDict,patFALSE,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (theExpression == NULL) {
      err = new patErrNullPointer("bioArithLogLogit") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    e =  theExpression ;
  }
  else if (indexType==operators.indexByNames[OP_LOGLOGIT]) {
    PyObject* pProb(NULL) ;
    PyObject* pAv(NULL) ;
    PyObject* pChoice(NULL) ;
    
    bioExpression* choice ;
    PyObject* pKeyList(NULL) ;
    PyObject* pKeyListAv(NULL) ;
    map<patULong,patULong> theDict ;
    map<patULong,patULong> theAvDict ;
    Py_ssize_t length ;
    
    GET_PY_OBJ(pExpression, "prob", pProb) ;
    if (pProb == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    pKeyList = PyDict_Keys(pProb) ;

    GET_PY_OBJ(pExpression, "av", pAv) ;
    if (pAv == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    pKeyListAv = PyDict_Keys(pAv) ;

    GET_PY_OBJ(pExpression, "choice", pChoice) ;
    if (pChoice == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    choice = buildExpression(pChoice,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    length = PyList_Size(pKeyList) ;
    //    DEBUG_MESSAGE("Length = " << length) ;
    for (Py_ssize_t pos=0 ; pos<length ; ++pos) {
      patULong theKey ;
      PyObject* pKey = PyList_GetItem(pKeyList, pos) ;
      if (pKey == NULL) {
	err = new patErrNullPointer("PyObject") ;
	WARNING(err->describe()) ;
	return NULL ;
      }
      PyObject* pExpression = PyDict_GetItem(pProb, pKey) ;
      if (pExpression == NULL) {
	err = new patErrMiscError("Should never happen !!!") ;
	WARNING(err->describe()) ;
	return NULL ;
      }
      theKey = PyLong_AsLong(pKey) ;
      bioExpression* expr = buildExpression(pExpression,err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      theDict[theKey] = expr->getId() ;
    }

    length = PyList_Size(pKeyListAv) ;
    for (Py_ssize_t pos=0 ; pos<length ; ++pos) {
      patULong theKey ;
      PyObject* pKey = PyList_GetItem(pKeyListAv, pos) ;
      if (pKey == NULL) {
	err = new patErrNullPointer("PyObject") ;
	WARNING(err->describe()) ;
	return NULL ;
      }
      PyObject* pExpression = PyDict_GetItem(pAv, pKey) ;
      if (pExpression == NULL) {
	err = new patErrMiscError("Should never happen !!!") ;
	WARNING(err->describe()) ;
	return NULL ;
      }
      theKey = PyLong_AsLong(pKey) ;
      bioExpression* expr = buildExpression(pExpression,err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      theAvDict[theKey] = expr->getId() ;
    }
    bioExpression* theExpression = new bioArithLogLogit(theExpRepository,patBadId,choice->getId(),theDict,theAvDict,patTRUE,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (theExpression == NULL) {
      err = new patErrNullPointer("bioArithLogLogit") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    e =  theExpression ;
  }
  else if (indexType==operators.indexByNames[OP_MULTSUM]) {
    e =  buildMultiSum(pExpression, err);  
  }
  else if (indexType==operators.indexByNames[OP_BAYESMEAN]) {
    
    PyObject* pError(NULL) ;
    GET_PY_OBJ(pExpression, "error", pError) ;
    if (pError != NULL) {
      if (pError != Py_None) {
        patString theError(PyBytes_AsString(PyUnicode_AsASCIIString(pError))) ;
        err = new patErrMiscError(theError) ;
        WARNING(err->describe()) ;
        return NULL ;
      }
    }
    PyObject* pMean(NULL) ;
    //    PyObject* pTypeMean(NULL) ;
    
    vector<patULong> theMeans ;

    
    GET_PY_OBJ(pExpression, "mean", pMean) ;
    if (pMean == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    Py_ssize_t vectSize = PyList_Size(pMean) ; 
    for (Py_ssize_t pos=0 ; pos < vectSize ; ++pos) {
      PyObject* pTheMean = PyList_GetItem(pMean, pos) ;
      bioExpression* theMean = buildExpression(pTheMean,err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      theMeans.push_back(theMean->getId()) ;
    }
   
    // Second argument
    PyObject* pRealizations(NULL) ;

    vector<patULong> theRealizations ;
    
    GET_PY_OBJ(pExpression, "realizations", pRealizations) ;
    if (pRealizations == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    vectSize = PyList_Size(pRealizations) ; 
    for (Py_ssize_t pos=0 ; pos < vectSize ; ++pos) {
      PyObject* pTheRealizations = PyList_GetItem(pRealizations, pos) ;
      bioExpression* theReal = buildExpression(pTheRealizations,err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
	}
      theRealizations.push_back(theReal->getId()) ;
    }


    // Third argument
    PyObject* pVarcovar(NULL) ;

    vector<vector<patULong> > theVarcovar ;


    GET_PY_OBJ(pExpression, "varcovar", pVarcovar) ;
    if (pVarcovar == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    vectSize = PyList_Size(pVarcovar) ; 
    for (Py_ssize_t pos=0 ; pos < vectSize ; ++pos) {
      vector<patULong> oneRow ;
      PyObject* pOneRow = PyList_GetItem(pVarcovar, pos) ;
      Py_ssize_t l = PyList_Size(pOneRow) ;
      if (l != vectSize) {
      	stringstream str ;
      	str << "The matrix is not square. It has " << vectSize << " rows and row " << pos+1 << " has " << l << " entries" ;
      	err = new patErrMiscError(str.str()) ;
      	WARNING(err->describe()) ;
      	return NULL ;
      }
      for (Py_ssize_t j=0 ; j < l ; ++j) {
      	PyObject* pCell = PyList_GetItem(pOneRow,j) ;
      	bioExpression* theReal = buildExpression(pCell,err) ;
      	if (err != NULL) {
      	  WARNING(err->describe()) ;
      	  return NULL ;
      	}
      	oneRow.push_back(theReal->getId()) ;
      	
      }
      theVarcovar.push_back(oneRow) ;
    }

    bioExpression* theExpression = new bioArithBayesMean(theExpRepository,
							 patBadId,
							 theMeans,
							 theRealizations,
							 theVarcovar,
							 err) ;

    DEBUG_MESSAGE("Bayes mean expression: " << *theExpression) ;
    if (theExpression == NULL) {
      err = new patErrNullPointer("bioArithBayesMean") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    e =  theExpression ;
   
    
  }
  else if (indexType==operators.indexByNames[OP_MH]) {
    PyObject* pBetas(NULL) ;
    PyObject* pType(NULL) ;
    
    vector<patULong> theBetas ;
    
    GET_PY_OBJ(pExpression, "type", pType) ;
    if (pType == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patULong theType = PyLong_AsLong(pType) ;
    GET_PY_OBJ(pExpression, "beta", pBetas) ;
    if (pBetas == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    if (theType == 2) {
      PyObject* pKeyList = PyDict_Keys(pBetas) ;
      if (pKeyList == NULL) {
      	err = new patErrNullPointer("PyObject") ;
      	WARNING(err->describe()) ;
      	return NULL ;
      }
      
      Py_ssize_t length = PyList_Size(pKeyList) ;
      for (Py_ssize_t pos=0 ; pos<length ; ++pos) {
      	PyObject* pKey = PyList_GetItem(pKeyList, pos) ;
      	if (pKey == NULL) {
      	  err = new patErrNullPointer("PyObject") ;
      	  WARNING(err->describe()) ;
      	  return NULL ;
      	}
      	PyObject* pTheBeta = PyDict_GetItem(pBetas, pKey) ;
      	if (pTheBeta == NULL) {
      	  err = new patErrNullPointer("PyObject") ;
      	  WARNING(err->describe()) ;
      	  return NULL ;
      	}
      	bioExpression* theBeta = buildExpression(pTheBeta,err) ;
      	if (err != NULL) {
      	  WARNING(err->describe()) ;
      	  return NULL ;
      	}
      	theBetas.push_back(theBeta->getId()) ;
      }

    }
    else if (theType == 1) {
      Py_ssize_t vectSize = PyList_Size(pBetas) ; 
      for (Py_ssize_t pos=0 ; pos < vectSize ; ++pos) {
      	PyObject* pTheBeta = PyList_GetItem(pBetas, pos) ;
      	bioExpression* theBeta = buildExpression(pTheBeta,err) ;
      	if (err != NULL) {
      	  WARNING(err->describe()) ;
      	  return NULL ;
      	}
      	theBetas.push_back(theBeta->getId()) ;
      }
    }
    else {
      err = new patErrMiscError("Argument of Draw is not a list nor a dictionary") ;
      WARNING(err->describe()) ;
      return NULL ;
    }

    // Density 
    PyObject *pDensity(NULL) ;
    GET_PY_OBJ(pExpression, "density", pDensity) ;
    if (pDensity == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    bioExpression *densityExpr = buildExpression(pDensity,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    // MH parameters
    PyObject *pWarmup(NULL) ;
    PyObject *pSteps(NULL) ;
    GET_PY_OBJ(pExpression, "warmup", pWarmup) ;
    if (pWarmup == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patULong warmup = PyLong_AsLong(pWarmup) ;
    GET_PY_OBJ(pExpression, "steps", pSteps) ;
    if (pSteps == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return(NULL) ;
    }
    patULong steps = PyLong_AsLong(pSteps) ;

    bioExpression *expr = new bioArithBayesMH(theExpRepository,
					      patBadId,
					      theBetas,
					      densityExpr->getId(),
					      warmup,
					      steps,
					      err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (expr == NULL) {
      err = new patErrNullPointer("bioArithBayesMH") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    e =  expr ;
  }

  else if  (indexType==operators.indexByNames[OP_DEFINE]) {
    err = new patErrMiscError("Operator 'define' must be reimplemented") ;
    WARNING(err->describe()) ;
    return NULL ;

    // PyObject *pName ;
    // PyObject *pExpr ;

    // GET_PY_OBJ(pExpression, "name", pName) ;
    // patString name = patString(PyBytes_AsString(PyUnicode_AsASCIIString(pName))) ;
    // patULong literalId = bioLiteralRepository::the()->getLiteralId(name,err) ;
    // if (err != NULL) {
    //   WARNING(err->describe()) ;
    //   return NULL ;
    // }

    // if (literalId == patBadId) {
    // 	// THe literal has not been defined yet. Create the expression.
    //   GET_PY_OBJ(pExpression, "expr", pExpr) ;
    //   bioExpression* theExpr = buildExpression(pExpr, err) ;
    //   if (err != NULL) {
    // 	WARNING(err->describe()) ;
    // 	return NULL ;
    //   }
      
    //   bioArithNamedExpression* namedExpr = new bioArithNamedExpression(theExpRepository,patBadId, name, theExpr->getId(),err) ;
    //   if (err != NULL) {
    // 	WARNING(err->describe()) ;
    // 	return NULL ;
    //   }
    //   theNamedExpressions.push_back(namedExpr) ;
    //   literalId = bioLiteralRepository::the()->getLiteralId(name,err) ;
    //   if (err != NULL) {
    // 	WARNING(err->describe()) ;
    // 	return NULL ;
    //   }
    // }
    // // The literal exists
    // pair<patULong,patULong> theCompositeIds = 
    //   bioLiteralRepository::the()->getCompositeLiteral(literalId,err) ;
    // if (err != NULL) {
    //   WARNING(err->describe()) ;
    //   return NULL ;
    // }
    // // First, check if it is a composite literal
    // if (theCompositeIds.second == patBadId) {
    //   stringstream str ;
    //   str << "Literal " << name << " cannot be used to define an expression, as it already used." ;
    //   err = new patErrMiscError(str.str()) ;
    //   WARNING(err->describe()) ;
    //   return NULL ;
    // }
    // bioArithCompositeLiteral* theCompositeLiteral = 
    //   new bioArithCompositeLiteral(theExpRepository,
    // 				   patBadId,
    // 				   theCompositeIds.first, 
    // 				   theCompositeIds.second) ;
    // e =  theCompositeLiteral ;
  }
  else {
    stringstream str ;
    str << "Not yet implemented node indexType [" << indexType << "]" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL ;
  }

  if(!pyID.empty()){
    theExpRepository->addIDtoMap(pyID, e->getId());
  }

  //  DEBUG_MESSAGE("Expression: " << *e) ;
  return e ;
}

bioExpression* bioModelParser::buildMultiSum(PyObject* pExpression, patError*& err){

  //sleep(1);
  //PyObject_Print(pExpression, stdout, Py_PRINT_RAW);

  PyObject* pTerms(NULL) ;
  PyObject* pType(NULL) ;

  vector<patULong> theTerms ;

  GET_PY_OBJ(pExpression, "type", pType) ;
  if (pType == NULL) {
    err = new patErrNullPointer("PyObject") ;
    WARNING(err->describe()) ;
    return(NULL) ;
  }

  //WARNING("type obtained:");

  patULong theType = PyLong_AsLong(pType) ;
  GET_PY_OBJ(pExpression, "terms", pTerms) ;
  if (pTerms == NULL) {
    err = new patErrNullPointer("PyObject") ;
    WARNING(err->describe()) ;
    return(NULL) ;
  }


  if (theType == 2) {
    /*
    PyObject* pKeyList = PyDict_Keys(pTerms) ;
    if (pKeyList == NULL) {
      err = new patErrNullPointer("PyObject") ;
      WARNING(err->describe()) ;
      return NULL ;
    }

    Py_ssize_t length = PyList_Size(pKeyList) ;
    //WARNING(length);

    for (Py_ssize_t pos=0 ; pos<length ; ++pos) {
      PyObject* pKey = PyList_GetItem(pKeyList, pos) ;
      if (pKey == NULL) {
        err = new patErrNullPointer("PyObject") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
      PyObject* pTheTerm = PyDict_GetItem(pTerms, pKey) ;
      if (pTheTerm == NULL) {
        err = new patErrNullPointer("PyObject") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
      bioExpression* theTerm = buildExpression(pTheTerm,err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      theTerms.push_back(theTerm->getId()) ;
    }*/
    PyObject *key, *pTheTerm;
    Py_ssize_t pos = 0;

    while (PyDict_Next(pTerms, &pos, &key, &pTheTerm)) {
       if (pTheTerm == NULL) {
        err = new patErrNullPointer("PyObject") ;
        WARNING(err->describe()) ;
        return NULL ;
      }
      bioExpression* theTerm = buildExpression(pTheTerm,err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      theTerms.push_back(theTerm->getId()) ;
    }
    //Py_DECREF(key);

  }else if (theType == 1) {
    Py_ssize_t vectSize = PyList_Size(pTerms) ; 
    for (Py_ssize_t pos=0 ; pos < vectSize ; ++pos) {
      PyObject* pTheTerm = PyList_GetItem(pTerms, pos) ;
      bioExpression* theTerm = buildExpression(pTheTerm,err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return NULL ;
      }
      theTerms.push_back(theTerm->getId()) ;
    }
  }else {
    err = new patErrMiscError("Argument of bioMultSum is not a list nor a dictionary") ;
    WARNING(err->describe()) ;
    return NULL ;
  }


  //WARNING("multisum parsed");
  //sleep(1);



  bioExpression* theExpression = 
  new bioArithMultinaryPlus(theExpRepository,patBadId,theTerms,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  if (theExpression == NULL) {
    err = new patErrNullPointer("bioArithMultinaryPlus") ;
    WARNING(err->describe()) ;
    return NULL ;
  }

  //WARNING("multisum built");
  //sleep(1);
  //attempt to clear things
  
  Py_XDECREF(pTerms);
  Py_XDECREF(pType);

  // if(!pyID.empty()){
  //   theExpRepository->addIDtoMap(pyID, theExpression->getId());
  // }

  return theExpression ;


}


void bioModelParser::setSampleFile(patString f) {
  theDataFile = f ;
}

bioExpressionRepository* bioModelParser::getRepository() {
  return theExpRepository ;
}

pair<vector<patString>,patHybridMatrix* > bioModelParser::getVarCovarMatrix(PyObject* pBioObject, patError*& err) {

  PyObject* pMatrix(NULL) ;

   GET_PY_OBJ(pBioObject, VARCOVAR, pMatrix) ;
   if (pMatrix == NULL) {
     err = new patErrNullPointer("PyObject") ;
     WARNING(err->describe()) ;
     return pair<vector<patString>,patHybridMatrix* >() ;
   }

   pair<vector<patString>,patHybridMatrix* > result = getMatrix(pMatrix,err) ;
   if (err != NULL) {
     WARNING(err->describe()) ;
     return pair<vector<patString>,patHybridMatrix* >() ;
   }
   return result ;
 }


pair<vector<patString>,patHybridMatrix* > bioModelParser::getMatrix(PyObject* pMatrix, patError*& err) {

   if (pMatrix == NULL) {
     return pair<vector<patString>,patHybridMatrix* >(vector<patString>(),NULL) ;
   }
   if (pMatrix == Py_None) {
     
     return pair<vector<patString>,patHybridMatrix* >(vector<patString>(),NULL) ;
   }
   else {
     PyObject* pDim(NULL) ;
     GET_PY_OBJ(pMatrix, "dim", pDim) ;
     if (pDim == NULL) {
       err = new patErrNullPointer("PyObject") ;
       WARNING(err->describe()) ;
       return pair<vector<patString>,patHybridMatrix* >(vector<patString>(),NULL) ;
     }
     //     patULong theDim = PyLong_AsLong(pDim) ;
     PyObject* pNames(NULL) ;
     GET_PY_OBJ(pMatrix, "names", pNames) ;
     if (pNames == NULL) {
       err = new patErrNullPointer("PyObject"); 
       WARNING(err->describe()) ;
       return pair<vector<patString>,patHybridMatrix* >(vector<patString>(),NULL) ;
       }
     patULong size = PyList_Size(pNames) ;
     vector<patString> theNames ;
     for (patULong i = 0 ; i < size ; ++i) {
       PyObject* n = PyList_GetItem(pNames,i) ;
       patString nn = patString(PyBytes_AsString(PyUnicode_AsASCIIString(n))) ;
       theNames.push_back(nn) ;
       }
     
     patHybridMatrix* theMatrix = new patHybridMatrix(size) ;
     if (theMatrix == NULL) {
       err = new patErrNullPointer("patHybridMatrix") ;
       WARNING(err->describe()) ;
       return pair<vector<patString>,patHybridMatrix* >(vector<patString>(),NULL) ;
     }
     
     PyObject* pValues(NULL) ;
     GET_PY_OBJ(pMatrix, "matrix", pValues) ;
     if (pValues == NULL) {
       err = new patErrNullPointer("PyObject") ;
       WARNING(err->describe()) ;
       return pair<vector<patString>,patHybridMatrix* >(vector<patString>(),NULL) ;
     }
     patULong nRows = PyList_Size(pValues) ;
     if (nRows != size) {
       stringstream str ;
       str << "The number of rows of the matrix (" << nRows << ") does not match the number of names (" << size << ")" ;
       err = new patErrMiscError(str.str()) ;
       WARNING(err->describe()) ;
       return pair<vector<patString>,patHybridMatrix* >(vector<patString>(),NULL) ;
     }
     for (patULong i = 0 ; i < size ; ++i) {
       PyObject* aRow = PyList_GetItem(pValues,i) ;
       patULong nCols = PyList_Size(aRow) ;
       if (nCols != size) {
	 stringstream str ;
	 str << "The number of columns of the matrix (" << nCols << ") does not match the number of names (" << size << ")" ;
	 err = new patErrMiscError(str.str()) ;
	 WARNING(err->describe()) ;
	 return pair<vector<patString>,patHybridMatrix* >(vector<patString>(),NULL) ;
       }
       for (patULong j = 0 ; j < size ; ++j) {
	 PyObject* aCell = PyList_GetItem(aRow,j) ;
	 patReal value = PyFloat_AsDouble(aCell) ;
	 theMatrix->setElement(i,j,value,err) ;
	 if (err != NULL) {
	   WARNING(err->describe()) ;
	 }
       }
     }
     return pair<vector<patString>,patHybridMatrix* >(theNames,theMatrix) ;
   }

}
