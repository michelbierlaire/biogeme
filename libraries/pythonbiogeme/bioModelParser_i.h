//-*-c++-*------------------------------------------------------------
//
// File name : bioModelParser_i.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Mon May 18 16:16:43 2009
//
//--------------------------------------------------------------------

#ifndef bioModelParser_i_h
#define bioModelParser_i_h

#include <Python.h> // Before all other includes
#include <limits.h>

#include <map>
#include "patString.h"

#define PY_FAIL(v)          ((v)==(NULL))
#define PY_ERR_PRINT()      PyErr_PrintEx(1)

#define GET_PY_OBJ(baseClass, name, result) { result = PyObject_GetAttrString(baseClass, const_cast<char *>(name)); \
                                      if (PY_FAIL(result)) { \
                                        PY_ERR_PRINT() ; \
                                        FATAL("Unable to get the object " << name) ; } \
                                     }


/***********************************************************
Name of classes, functions or variables in 'bio_iterator.py'
************************************************************/
#define ITERATOR_LIST     "listOfIterator"
#define ITERATOR_NAME     "iteratorName"
#define INDEX_VARIABLE    "variable"
#define ITERATOR_TYPE     "type"
#define DRAW_TYPE         "DRAW"
#define ROW_TYPE          "ROW"
#define META_TYPE         "META"
#define CHILD             "child"
#define STRUCTURE_NAME    "name"
#define FILENAME          "filename"


/******************************************************
Name of classes, functions or variables in 'biogeme.py'
*******************************************************/
#define BIOGEME_OBJECT  "BIOGEME_OBJECT"
#define ESTIMATE        "ESTIMATE"
#define DRAWS           "DRAWS"
#define UNIFDRAWS       "UNIFDRAWS"
#define BAYESIAN        "BAYESIAN"
#define STATISTICS      "STATISTICS"
#define FORMULAS        "FORMULAS"
#define EXCLUDE         "EXCLUDE"
#define WEIGHT          "WEIGHT"
#define SIMULATE        "SIMULATE"
#define EXPRESSIONS     "EXPRESSIONS"
#define VARCOVAR        "VARCOVAR"
#define CONSTRAINTS     "CONSTRAINTS"
#define PARAMETERS      "PARAMETERS"

#define OPERATOR_OBJECT "Operator"


/*************************************************************
Name of classes, functions or variables in 'bio_expression.py'
**************************************************************/

/* Class Operator */
// Name of the operators and the functions
#define OP_NUM    "num"
#define OP_VAR    "var"
#define OP_USEREXPR "userexpr"
#define OP_USERDRAWS "userdraws"
#define OP_RV    "rv"
#define OP_NORMAL "normal"
#define OP_UNIFORMSYM "uniformSym"
#define OP_UNIFORM "uniform"
#define OP_PARAM  "param"
#define OP_ABS    "absOp"
#define OP_NEG    "negOp"
#define OP_EXP    "exp"
#define OP_LOG    "log"
#define OP_NORMALCDF   "bioNormalCdf"
#define OP_ADD    "add"
#define OP_SUB    "sub"
#define OP_MUL    "mul"
#define OP_DIV    "div"
#define OP_POW    "power"
#define OP_AND    "andOp"
#define OP_OR     "orOp"
#define OP_EQ     "equal"
#define OP_NEQ    "notEqual"
#define OP_GT     "greater"
#define OP_GE     "greaterEq"
#define OP_LT     "less"
#define OP_LE     "lessEq"
#define OP_MIN    "minOp"
#define OP_MAX    "maxOp"
#define OP_MOD    "mod"
#define OP_SUM    "sumOp"
#define OP_DRAWS  "mcdraws"
#define OP_UNIFDRAWS  "mcunifdraws"
#define OP_MC  "monteCarloOp"
#define OP_MCCV  "monteCarloCVOp"
#define OP_PROD   "prodOp"
#define OP_INTEGRAL   "integralOp"
#define OP_DERIVATIVE   "derivativeOp"
#define OP_ELEM   "elemOp"
#define OP_LOGIT   "logitOp"
#define OP_LOGLOGIT   "loglogitOp"
#define OP_MULTSUM   "multSumOp"
#define OP_MULTPROD   "multProdOp"
#define OP_NORMALPDF   "bioNormalPdf"
#define OP_ENUM   "enumOp"
#define OP_DEFINE   "defineOp"
#define OP_MH   "mhOp"
#define OP_BAYESMEAN "bayesMeanOp"

// Bounds
#define MIN_UNOP_INDEX        "MIN_UNOP_INDEX"
#define MAX_UNOP_INDEX        "MAX_UNOP_INDEX"
#define MIN_BINOP_INDEX       "MIN_BINOP_INDEX"
#define MAX_BINOP_INDEX       "MAX_BINOP_INDEX"
#define MIN_ITERATOROP_INDEX  "MIN_ITERATOROP_INDEX"
#define MAX_ITERATOROP_INDEX  "MAX_ITERATOROP_INDEX"
#define UNDEF_OP              "UNDEF_OP"

// Name of the function returning the index associated to 
// an operator
#define GET_OPERATOR_FUNC "getOpIndex"

/* end Class Operator */

/* Class Expression */
// Name of the variable containing the operator index 
// associated with the current expression
#define OPERATOR_INDEX    "operatorIndex"
/* end Class Expression */

/* Class Numeric */
#define NUMERIC_VALUE   "number"
/* class Numeric */

/* Class Variable */
#define VARIABLE_NAME  "name"
/* Class Numeric */

/* Class bioDraws */
#define DRAWS_NAME  "name"

/* Class bioNormalDraws */
#define RAND_NORMAL_NAME    "name"
#define RAND_NORMAL_INDEX   "index"
/* end Class bioNormal */

/* class bioUniformDraws */
#define RAND_UNIFORM_NAME    "name"
#define RAND_UNIFORM_INDEX   "index"
/* end Class bioUniform */

/* class bioUniformSymmetricDraws */
#define RAND_UNIFORMSYM_NAME    "name"
#define RAND_UNIFORMSYM_INDEX   "index"
/* end Class bioUniform */



patString listOfOperatorName[] = {
  OP_NUM,   
  OP_VAR,     
  OP_USEREXPR,
  OP_USERDRAWS,
  OP_RV,      
  OP_PARAM,   
  OP_NORMAL,  
  OP_UNIFORM, 
  OP_UNIFORMSYM, 
  OP_ABS,     
  OP_NEG,     
  OP_EXP,
  OP_LOG,
  OP_NORMALCDF,
  OP_ADD,
  OP_SUB,
  OP_MUL,
  OP_DIV,
  OP_POW,
  OP_AND,
  OP_OR,
  OP_EQ,
  OP_NEQ,
  OP_GT,
  OP_GE,
  OP_LT,
  OP_LE,
  OP_MIN,
  OP_MAX,
  OP_MOD,
  OP_SUM,
  OP_DRAWS,
  OP_UNIFDRAWS,
  OP_MC,
  OP_MCCV,
  OP_PROD,
  OP_INTEGRAL,
  OP_DERIVATIVE,
  OP_ELEM,
  OP_LOGIT,
  OP_LOGLOGIT,
  OP_MULTSUM,
  OP_MULTPROD,
  OP_NORMALPDF,
  OP_ENUM,
  OP_DEFINE,
  OP_MH,
  OP_BAYESMEAN
};


struct operators_t {
  int minUnopIndex ;
  int maxUnopIndex ;
  int minBinopIndex ; 
  int maxBinopIndex ;
  int minIteratoropIndex ;
  int maxIteratoropIndex ;
  int undefOp ;
  map<patString,int> indexByNames ;
} operators;


// Build the 'operators' structure from Biogeme 'Operator' Python object
void setOperators (PyObject* pOperator) {

  int i;
  int len=sizeof(listOfOperatorName)/sizeof(listOfOperatorName[0]);
  PyObject* pOperatorValue;
  PyObject* pOpIndex;

  operators.minUnopIndex       = PyLong_AsLong(PyObject_GetAttrString(pOperator, MIN_UNOP_INDEX));
  operators.maxUnopIndex       = PyLong_AsLong(PyObject_GetAttrString(pOperator, MAX_UNOP_INDEX));
  operators.minBinopIndex      = PyLong_AsLong(PyObject_GetAttrString(pOperator, MIN_BINOP_INDEX));
  operators.maxBinopIndex      = PyLong_AsLong(PyObject_GetAttrString(pOperator, MAX_BINOP_INDEX));
  operators.minIteratoropIndex = PyLong_AsLong(PyObject_GetAttrString(pOperator, MIN_ITERATOROP_INDEX));
  operators.maxIteratoropIndex = PyLong_AsLong(PyObject_GetAttrString(pOperator, MAX_ITERATOROP_INDEX));
  operators.undefOp            = PyLong_AsLong(PyObject_GetAttrString(pOperator, UNDEF_OP));

  for (i=0; i<len; ++i) {
    pOperatorValue = PyObject_GetAttrString(pOperator, const_cast<char *>(listOfOperatorName[i].c_str()));
    patString theOperator(PyBytes_AsString(PyUnicode_AsASCIIString(pOperatorValue))) ;
    char format[] = "s" ;
    char op[] = GET_OPERATOR_FUNC ;
    pOpIndex = PyObject_CallMethod(pOperator, op, format, theOperator.c_str());
    operators.indexByNames[listOfOperatorName[i]] = PyLong_AsLong(pOpIndex);
  }
}

#endif
