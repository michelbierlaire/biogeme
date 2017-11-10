//-*-c++-*------------------------------------------------------------
//
// File name : trQuadratic.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Jan 26 14:44:05 2000
//
//--------------------------------------------------------------------

#ifndef trQuadratic_h
#define trQuadratic_h

#include "trFunction.h"

class trQuadratic : public trFunction {

public:
  trQuadratic(const trVector& x) ;
  virtual ~trQuadratic() {} ;
  virtual patReal getFunction(const trVector& x,
		      patError*& err) const ;
  virtual trVector getGradient(const trVector& x,
			       patError*& err) const ;
  virtual trHessian* computeHessian(const patVariables& x,
				    trHessian& hessian,
				    patError*& err) const ;
  virtual trVector getHessianTimesVector(const trVector& x,
					 const trVector& v,
					 patError*& err)
    const ;
  virtual patBoolean isGradientAvailable() const {return patTRUE ;}
  virtual patBoolean isHessianAvailable() const {return patTRUE ;}
  virtual patBoolean isHessianTimesVectorAvailable() const {return patTRUE ;}
  virtual unsigned long getDimension() const ;

private :
  trVector scaling ;

};

#endif
