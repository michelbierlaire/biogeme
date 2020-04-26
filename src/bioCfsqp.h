//-*-c++-*------------------------------------------------------------
//
// File name : bioCfsqp.h
// Author :    Michel Bierlaire
// Date :      Tue Aug 13 08:59:34 2019
//
//--------------------------------------------------------------------

#ifndef bioCfsqp_h
#define bioCfsqp_h 

/**
 This class defines a C++ interface to the cfsqp routine, by Lawrence, Zhou and Tits (1998). 
*/

#include "biogeme.h"

#include <iostream>
#define DEBUG_MESSAGE(message)  {std::cout << message << std::endl ;}
#define GENERAL_MESSAGE(message)  {std::cout << message << std::endl ;}
#define DETAILED_MESSAGE(message)  {std::cout << message << std::endl ;}
#define WARNING(message)  {std::cout << message << std::endl ;}
#define FATAL(message)  {std::cout << message << std::endl ;}

class bioCfsqp {

public:
  /**
   */
  bioCfsqp(biogeme* bio) ;
  /**
   */
  virtual ~bioCfsqp() ;

  /**
   */
  void defineStartingPoint(const std::vector<bioReal>& x0) ;

  /**
   */
  std::vector<bioReal> getStartingPoint() ;
  /**
   */
  std::vector<bioReal> getSolution() ;

  /**
   */
  bioReal getValueSolution() ;

  /**
   */
  std::vector<bioReal> getLowerBoundsLambda() ;
  /**
   */
  std::vector<bioReal> getUpperBoundsLambda() ;
  /**
     @return Diagnostic from cfsqp
   */
  bioString run() ;
  /**
     @return number of iterations. If there is any error, 0 is returned. 
   */
  bioUInt nbrIter() ;

  void setParameters(int _mode,
		     int _iprint,
		     int _miter,
		     bioReal _eps,
		     bioReal _epseqn,
		     bioReal _udelta) ;

private:

  std::vector<bioReal> startingPoint ;
  std::vector<bioReal> solution ;
  std::vector<bioReal> lowerBoundsLambda ;
  std::vector<bioReal> upperBoundsLambda ;
  std::vector<bioReal> lowerBounds ;
  std::vector<bioReal> upperBounds ;
  std::vector<bioReal> fixedBetas ;
  biogeme* theBiogeme ;

  int mode;
  int iprint;
  int miter;
  bioReal eps;
  bioReal epseqn;
  bioReal udelta ;
  int nIter ;
};

/**
  User functions required by cfsqp. See User's guide.
 */
void obj(int nparam, int j, bioReal* x, bioReal* fj, void* cd) ;
/**
  User functions required by cfsqp. See User's guide.
 */
void constr(int nparam,int j,bioReal* x, bioReal* gj, void* cd) ;
/**
  User functions required by cfsqp. See User's guide.
 */
void gradob(int nparam, 
	    int j,
	    bioReal* x,
	    bioReal* gradfj, 
	    void (* dummy)(int, int, bioReal *, bioReal *, void *),
	    void* cd) ;
/**
  User functions required by cfsqp. See User's guide.
 */
void gradcn(int nparam, 
	    int j,
	    bioReal* x,
	    bioReal* gradgj, 
	    void (* dummy)(int, int, bioReal *, bioReal *, void *),
	    void* cd) ;

#endif
