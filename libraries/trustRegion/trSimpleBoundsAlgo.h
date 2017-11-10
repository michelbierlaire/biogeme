//-*-c++-*------------------------------------------------------------
//
// File name : trSimpleBoundsAlgo.h
// Date :      Sun Nov 26 15:10:53 2000
//
//--------------------------------------------------------------------


#ifndef trSimpleBoundsAlgo_h
#define trSimpleBoundsAlgo_h

#include <sstream>
#include <fstream>
#include "patError.h"
#include "trVector.h"
#include "trNonLinearAlgo.h"
#include "trParameters.h"


class patNonLinearProblem ;
class trFunction ;
class trBounds ;
class trMatrixVector;
class trPrecond ;

class trHessian ;
class trSecantUpdate ;
class patIterationBackup ;
/** 
  @doc Implementation of an algorithm to minimize a non linear function subject to simple bounds constraints.
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Sun Nov 26 15:10:53 2000)
*/

class trSimpleBoundsAlgo : public trNonLinearAlgo {
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
    trMINRADIUS,
    /**
     */
    trUSER
  }   ;

  /**
     @param f pointer to the function to minimize
     @param b pointer to the description of bounds
     @param initSolution initial solution
     @param _radius initial radius of the trust region
     @param err ref. of the pointer to the error object.
   */
  trSimpleBoundsAlgo(patNonLinearProblem* aProblem,
		     const trVector& initSolution,
		     trParameters theParameters,
		     patIterationBackup* inter,
		     patError*& err) ;
  /**
   */
  virtual ~trSimpleBoundsAlgo() ;

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
  trVector getSolution(patError*& err) ;
  /**
   */
  patReal getValueSolution(patError*& err) ;
  /**
   */
   patVariables getLowerBoundsLambda() ;
  /**
   */
   patVariables getUpperBoundsLambda() ;
private:
  patBoolean stop(patReal& gMax,patError*& err) ;
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

  trTermStatus status ;
  trVector solution ;
  trHessian* trueHessian ;
  trHessian* bhhhHessian ;
  trHessian* initSecantUpdate ;
  patReal radius ;
  trSecantUpdate* quasiNewton ; 
  trPrecond* precond ;
  stringstream* iterInfo ;
  patBoolean mustInitBFGS ;
  trParameters theParameters ;
  patIterationBackup* theInteraction ;
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




};

#endif
