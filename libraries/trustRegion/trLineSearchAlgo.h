//-*-c++-*------------------------------------------------------------
//
// File name : trLineSearchAlgo.h
// Date :      Tue Nov 16 10:25:37 2004
//
//--------------------------------------------------------------------


#ifndef trLineSearchAlgo_h
#define trLineSearchAlgo_h

#include <fstream>
#include <sstream>
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

/** 
  @doc Implementation of an algorithm to minimize a non linear function subject to simple bounds constraints.
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Sun Nov 26 15:10:53 2000)
*/

class trLineSearchAlgo : public trNonLinearAlgo {
public :  

  /**
     @param f pointer to the function to minimize
     @param b pointer to the description of bounds
     @param initSolution initial solution
     @param _radius initial radius of the trust region
     @param err ref. of the pointer to the error object.
   */
  trLineSearchAlgo(patNonLinearProblem* aProblem,
		   const trVector& initSolution,
		   trParameters theParameter,
		   patError*& err) ;
  /**
   */
  virtual ~trLineSearchAlgo() ;

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
   */
  patString getName() ;


private :

  trVector solution ;
  trParameters theParameters ;
  // Generic hessian, unaware of its real nature
  trHessian* hessian ;
  stringstream* iterInfo ;
  patULong iter ;

  patReal valCandidate ;
  trVector gradientCandidate ;
  ofstream reportFile ;
  trFunction* f ;
  patReal function ;
  trVector gk ;

  trVector oldSolution ;
  trVector oldGk ;



};

#endif
