//-*-c++-*------------------------------------------------------------
//
// File name : patFunction.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri Jan 29 11:50:06 1999
//
//--------------------------------------------------------------------

#ifndef patFunction_h
#define patFunction_h

#include "patConst.h"
#include "patVariables.h"
#include "patError.h"
#include <map>

/**
   @doc Defines an interface for objective functions used in derivate-free optinization
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Fri Jan 29 11:50:06 1999)
 */
class patFunction {
public:
  /**
     Purely virtual
   */
  virtual string getName() const = PURE_VIRTUAL ;
  /**
   */
  virtual ~patFunction() {}
  /**
   */
  patReal operator()(const patVariables& x, 
		     patError*& err) ;
  /**
     Purely virtual
   */
  virtual unsigned long getDimension()  = PURE_VIRTUAL ;
  /**
   */
  virtual void reset() ;
  /**
   */
  virtual unsigned long getNbrEval() ;
protected:
  virtual patReal evaluate(const patVariables& x, 
			   patError*) = PURE_VIRTUAL;

private:
  map<patVariables,patReal,less<patVariables> >  functionValue ; 
};

#endif


