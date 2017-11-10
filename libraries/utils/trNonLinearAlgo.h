//-*-c++-*------------------------------------------------------------
//
// File name : trNonLinearAlgo.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri Aug 31 11:46:43 2001
//
//--------------------------------------------------------------------

#ifndef trNonLinearAlgo_h
#define trNonLinearAlgo_h

/**
   @doc Defines a generic interface for algorithms solving the maximum likelihood
   problem 
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi} (Fri Apr 27 12:04:33 2001) 
*/

#include "patError.h"
#include "patVariables.h"

class patNonLinearProblem ;
class patIterationBackup ;

class trNonLinearAlgo {

public:
  /**
   */
  trNonLinearAlgo(patNonLinearProblem* aProblem = NULL) ;
  /**
   */
  virtual ~trNonLinearAlgo() ;  
  /**
     @return Diagnostic from algorithm
   */
  virtual patString run(patError*& err) = PURE_VIRTUAL ;
  /**
     @return number of iterations. If there is any error, 0 is returned. 
   */
  virtual patULong nbrIter() = PURE_VIRTUAL ;
  /**
   */
  virtual patVariables getSolution(patError*& err) = PURE_VIRTUAL ;
  /**
   */
  virtual patReal getValueSolution(patError*& err) = PURE_VIRTUAL ;
  /**
   */
  virtual patVariables getLowerBoundsLambda() = PURE_VIRTUAL ;
  /**
   */
  virtual patVariables getUpperBoundsLambda() = PURE_VIRTUAL ;
  /**
   */
  virtual void defineStartingPoint(const patVariables& x0) = PURE_VIRTUAL ;

  /**
   */
  virtual patString getName() = PURE_VIRTUAL ;
  /**
   */
  virtual void setBackup(patIterationBackup* aBackup) ;
  /**
   */
  virtual patBoolean isAvailable() const ;


protected:
  patNonLinearProblem* theProblem ;
  patIterationBackup* theBackup ;

};
#endif 
 
