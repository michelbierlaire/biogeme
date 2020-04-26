#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <iomanip>
#include "patTestNormal.h"


patReal patTestNormal::getFunction(trVector* x,
				   patBoolean* success,
				   patError*& err) {
  *success = patTRUE ;
  patReal result = patNormalCdf::the()->compute((*x)[0],err) ;
  //  DEBUG_MESSAGE(setprecision(7) << setiosflags(ios::scientific|ios::showpos) <<"f(" << (*x)[0] << ")=" << result) ;
  return result ;
}

trVector* patTestNormal::getGradient(trVector* x,
		      trVector* grad,
		      patBoolean* success,
		      patError*& err) {
  *success = patTRUE ;
  (*grad)[0] = patNormalCdf::the()->derivative((*x)[0],err) ;
  return grad ;
}


trVector patTestNormal::getHessianTimesVector(trVector* x,
					      trVector* v,
					      patBoolean* success,
					      patError*& err) const {
  return trVector() ;
}

trHessian* patTestNormal::computeHessian(trVector* x,
					 trHessian& hessian,
					 patBoolean* success,
					 patError*& err) {
  return NULL ;
}

trHessian* patTestNormal::getCheapHessian(trVector* x,
					  trHessian& hessian,
					  patBoolean* success,
					  patError*& err) {
  return NULL ;
}

patBoolean patTestNormal::isCheapHessianAvailable() {
  return patFALSE ;
}

patBoolean patTestNormal::isGradientAvailable() const {
  return patTRUE ;
}

patBoolean patTestNormal::isHessianTimesVectorAvailable() const {
  return patFALSE ;
}

patBoolean patTestNormal::isHessianAvailable() const {
  return patFALSE ;
}

unsigned long patTestNormal::getDimension() const {
  return 1 ;
}
