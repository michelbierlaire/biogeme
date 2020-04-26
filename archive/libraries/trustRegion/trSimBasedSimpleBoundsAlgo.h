//-*-c++-*------------------------------------------------------------
//
// File name : trSimBasedSimpleBoundsAlgo.h
// Date :      Sun Nov 26 15:10:53 2000
//
//--------------------------------------------------------------------


#ifndef trSimBasedSimpleBoundsAlgo_h
#define trSimBasedSimpleBoundsAlgo_h

#include <fstream>
#include <sstream>
#include "patError.h"
#include "trVector.h"
#include "trNonLinearAlgo.h"

class patSimBasedOptimizationProblem ;
class trFunction ;
class trBounds ;
class trMatrixVector;
class trPrecond ;

class trHessian ;
class trSecantUpdate ;

/** 
  @doc Implementation of an algorithm to minimize a non linear function subject to simple bounds constraints.
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Sun Nov 26 15:10:53 2000)
*/

class trSimBasedSimpleBoundsAlgo : public trNonLinearAlgo {
public :  

  /**
   */
  enum trTermStatus {
    /**
     */
    trUNKNOWN , 
    /**
     */
    trMAXITER, 
    /**
     */
    trCONV,
    /**
     */
    trMINRADIUS
  }   ;

  /**
     @param f pointer to the function to minimize
     @param b pointer to the description of bounds
     @param initSolution initial solution
     @param _radius initial radius of the trust region
     @param err ref. of the pointer to the error object.
   */
  trSimBasedSimpleBoundsAlgo(patSimBasedOptimizationProblem* aProblem,
			     const trVector& initSolution,
			     trParameters theParameters,
			     patError*& err) ;
  /**
   */
  virtual ~trSimBasedSimpleBoundsAlgo() ;

  /**
   */
  void defineStartingPoint(const patVariables& x0) ;

  /**
     Executes the algorithm
     @param err ref. of the pointer to the error object.
     @return termination status
   */
  patString run(patError*& err) ;
  /**
     @return number of iterations. If there is any error, 0 is returned. 
   */
  patULong nbrIter() ;


  /**
   */
  trTermStatus getTermStatus() ;
  /**
   */
  trVector getSolution(patError*& err) ;  /**
   */
  patReal getValueSolution(patError*& err) ;
  /**
   */
   patVariables getLowerBoundsLambda() ;
  /**
   */
   patVariables getUpperBoundsLambda() ;
private:
  patBoolean stop(patReal& gMax, patError*& err) ;
  patBoolean checkOpt(const trVector& x,
		      const trVector& g,
		      patReal& gmax,
		      patError*& err) ;

  /**
     previousIterate, previousGradient and currentGradient  are used only if a  BFGS update is requested.
  */
  trMatrixVector* computeHessian( trVector& previousIterate,
				  trVector& previousGradient,
				  trVector& currentIterate,
				  trVector& currentGradient,
				  patError*& err) ;

  /**
   */
  patString getName() ;


  private :

  patReal computeRhoK(patReal fold, 
		      patReal fnew,
		      patReal modelImprovement) ;

private :

  patSimBasedOptimizationProblem* theSimBasedProblem ;
  trTermStatus status ;
  trVector solution ;
  patReal radius ;
  // Specific hessian. Usually, only one will be non zero.  The generic
  // pointer is used to avoid to check which one is non zero when no specific
  // function is required.
  trHessian* trueHessian ;
  trSecantUpdate* quasiNewton ; 
  trPrecond* precond ;
  stringstream* iterInfo ;
  patBoolean mustInitBFGS ;
  patULong iter ;

  ofstream reportFile ;
  trFunction* f ;
  trBounds* bounds ;
  patReal function ;
  trVector gk ;
  // Generic hessian, unaware of its real nature
  trMatrixVector* hessian ;

  trVector oldSolution ;
  trVector oldGk ;


  patVariables lowerLambda ;
  patVariables upperLambda ;

  patBoolean exactHessian ;
  patBoolean cheapHessian ;

  unsigned long userNbrDraws ;

  trParameters theParameters ;
};

#endif
