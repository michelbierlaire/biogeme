//-*-c++-*------------------------------------------------------------
//
// File name : patSolvOpt.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Tue Feb  5 11:38:44 2002
//
//--------------------------------------------------------------------

#ifndef patSolvOpt_h
#define patSolvOpt_h 

/**
@doc This class defines a C++ interface to the SolvOpt routine, by Alexcei Kuntsevich and Franz Kappel. \URL{http://www.kfunigraz.ac.at/imawww/kuntsevich/solvopt}
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Tue Feb  5 11:38:55 2002) 

*/


#include "patError.h"
#include "trNonLinearAlgo.h"
#include "solvoptParameters.h"

class patNonLinearProblem ;

class patSolvOpt : public trNonLinearAlgo {

  friend patReal ObjectFunctionValue(patReal x[]) ;
  friend void ObjectFunctionGradient(patReal x[], patReal g[]) ;
  friend patReal MaxResidual(patReal x[]) ;
  friend void GradMaxResConstr(patReal x[], patReal gc[]) ;

public:
  typedef enum {patNonLinEq,
		patLinEq,
		patNonLinIneq,
		patLinIneq,
		patUpperBound,
		patLowerBound} patMaxConstraintType ;
public:
  /**
   */
  patSolvOpt(solvoptParameters theParameters, 
	     patNonLinearProblem* aProblem = NULL) ;
  /**
   */
  virtual ~patSolvOpt() ;
  /**
   */
  void setProblem(patNonLinearProblem* aProblem) ;
  /**
   */
  patNonLinearProblem* getProblem() ;

  /**
   */
  void defineStartingPoint(const patVariables& x0) ;

  /**
   */
  patVariables getStartingPoint() ;
  /**
   */
  patVariables getSolution(patError*& err) ;

  /**
   */
  patReal getValueSolution(patError*& err) ;

  /**
   */
  patString getName() ;

  /**
     @return Diagnostic from cfsqp
   */
  patString run(patError*& err) ;
  /**
     @return number of iterations. If there is any error, 0 is returned. 
   */
  patULong nbrIter() ;

  /**
   */
   patVariables getLowerBoundsLambda() ;
  /**
   */
   patVariables getUpperBoundsLambda() ;

private:
  solvoptParameters theParameters ;
  patVariables startingPoint ;
  patVariables solution ;
  patULong nIter ;
  
};

/**
 @doc User functions required by solvopt
 */
  patReal ObjectFunctionValue(patReal x[]) ;
/**
 @doc User functions required by solvopt
 */
  void ObjectFunctionGradient(patReal x[], patReal g[]) ;
/**
 @doc User functions required by solvopt
 */
  patReal MaxResidual(patReal x[]) ;
/**
 @doc User functions required by solvopt
 */
  void GradMaxResConstr(patReal x[], patReal gc[]) ;

#endif
