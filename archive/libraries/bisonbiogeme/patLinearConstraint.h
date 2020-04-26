//-*-c++-*------------------------------------------------------------
//
// File name : patLinearConstraint.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri Oct 25 14:57:35 2002
//
//--------------------------------------------------------------------

#ifndef patLinearConstraint_h
#define patLinearConstraint_h

#include <list>
#include "patType.h"
#include "patVariables.h"

/**
 */
struct patConstraintTerm {
  /**
   */
  patReal fact ;
  /**
   */
  patString param ;
};

/**
 */
typedef list<patConstraintTerm> patConstraintEquation ;


/**
 */
class patLinearConstraint {
public:
  /**
   */
  typedef enum {patEQUAL, patLESSEQUAL, patGREATEQUAL } patConstraintType ;
  /**
   */
  patConstraintEquation theEquation ;
  /**
   */
  patConstraintType theType ;
  /**
   */
  patReal theRHS ;

  /**
   */
  patString getFormForPython() ;
};

/**
 */
typedef list<patLinearConstraint> patListLinearConstraint ;

/**
 */
ostream& operator<<(ostream &str, const patLinearConstraint& x) ;

/**
 */
/**
 */
typedef pair<patVariables,patReal> patProblemLinearConstraint ;

/**
 */
typedef list<patProblemLinearConstraint> patListProblemLinearConstraint ;

#endif
