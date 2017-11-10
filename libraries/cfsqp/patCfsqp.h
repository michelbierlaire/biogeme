//-*-c++-*------------------------------------------------------------
//
// File name : patCfsqp.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri Apr 27 07:34:12 2001
//
//--------------------------------------------------------------------

#ifndef patCfsqp_h
#define patCfsqp_h 

/**
@doc This class defines a C++ interface to the cfsqp routine, by Lawrence, Zhou and Tits (1998). \URL{http://gachinese.com/aemdesign/FSQPframe.htm}
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Fri Apr 27 07:34:12 2001) 

*/


#include "patError.h"
#include "trNonLinearAlgo.h"

class patNonLinearProblem ;
class patIterationBackup ;

class patCfsqp : public trNonLinearAlgo {

public:
  /**
   */
  patCfsqp(patIterationBackup* i, patNonLinearProblem* aProblem = NULL) ;
  /**
   */
  virtual ~patCfsqp() ;

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
  patVariables getLowerBoundsLambda() ;
  /**
   */
  patVariables getUpperBoundsLambda() ;
  /**
   */
  patVariables getNonLinIneqLambda() ;
  /**
   */
  patVariables getLinIneqLambda() ;
  /**
   */
  patVariables getNonLinEqLambda() ;
  /**
   */
  patVariables getLinEqLambda() ;

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

  void setParameters(int _mode,
		     int _iprint,
		     int _miter,
		     patReal _eps,
		     patReal _epseqn,
		     patReal _udelta,
		     patString sf) ;

private:

  patVariables startingPoint ;
  patVariables solution ;
  patVariables lowerBoundsLambda ;
  patVariables upperBoundsLambda ;
  patVariables nonLinIneqLambda ;
  patVariables linIneqLambda ;
  patVariables nonLinEqLambda ;
  patVariables linEqLambda ;

  int mode;
  int iprint;
  int miter;
  patReal eps;
  patReal epseqn;
  patReal udelta ;
  patIterationBackup* theInteraction ;
  patString stopFile ;
  int nIter ;
};

/**
 @doc User functions required by cfsqp. See User's guide.
 */
void obj(int nparam, int j, patReal* x, patReal* fj, void* cd) ;
/**
 @doc User functions required by cfsqp. See User's guide.
 */
void constr(int nparam,int j,patReal* x, patReal* gj, void* cd) ;
/**
 @doc User functions required by cfsqp. See User's guide.
 */
void gradob(int nparam, 
	    int j,
	    patReal* x,
	    patReal* gradfj, 
	    void (* dummy)(int, int, patReal *, patReal *, void *),
	    void* cd) ;
/**
 @doc User functions required by cfsqp. See User's guide.
 */
void gradcn(int nparam, 
	    int j,
	    patReal* x,
	    patReal* gradgj, 
	    void (* dummy)(int, int, patReal *, patReal *, void *),
	    void* cd) ;



#endif
