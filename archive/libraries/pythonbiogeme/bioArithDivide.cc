//-*-c++-*------------------------------------------------------------
//
// File name : bioArithDivide.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Apr 28 20:58:50 2009
//
//--------------------------------------------------------------------

#include <sstream>
#include "patMath.h"
#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "bioLiteralRepository.h"
#include "bioArithDivide.h"
#include "bioArithMult.h"
#include "bioArithUnaryMinus.h"
#include "bioArithBinaryMinus.h"
#include "bioArithPower.h"
#include "bioArithConstant.h"
#include "bioArithLiteral.h"
#include "bioExpressionRepository.h"
#include "bioArithCompositeLiteral.h"

bioArithDivide::bioArithDivide(bioExpressionRepository* rep,
			       patULong par,
                               patULong left, 
                               patULong right,
patError*& err) : 
  bioArithBinaryExpression(rep,par,left,right,err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

}

bioArithDivide::~bioArithDivide() {}

patString bioArithDivide::getOperatorName() const {
  return ("/") ;
}

patReal bioArithDivide::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)   {
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){

    if (leftChild == NULL || rightChild == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return patReal();
    }
    
    patReal left ;
    left = leftChild->getValue(prepareGradient, currentLap, err) ;

    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }

    patReal result ;
    if (left == 0.0) {
      return 0.0 ;
    }else{
      patReal right ;
      right = rightChild->getValue(prepareGradient, currentLap, err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return patReal();
      }

      if (right == 0.0) {
        result = patMaxReal ;
        //return patMaxReal ;
      }else{

        result = left / right ;

        if (performCheck) {
          if (!patFinite(result)) {
            result = patMaxReal ;
            //return patMaxReal ;
          }
          /*else {
            //return result ;
          }
        } 
        else {
          //return result ;
        */
        }
      }
    }

    lastValue = result;
    lastComputedLap = currentLap;
    return lastValue;
  }else{
    return lastValue ;
  }
}

bioExpression* bioArithDivide::getDerivative(patULong aLiteralId, 
					     patError*& err) const {

  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    return NULL;
  }
  
  bioExpression* leftValue = leftChild->getDeepCopy(theRepository,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  bioExpression* rightValue = rightChild->getDeepCopy(theRepository,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  bioExpression* leftDeriv = leftChild->getDerivative(aLiteralId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  bioExpression* rightDeriv = rightChild->getDerivative(aLiteralId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  if (leftChild->dependsOf(aLiteralId)) {
    if (rightChild->dependsOf(aLiteralId)) {
      bioExpression* lderiv_r = new bioArithMult(theRepository,patBadId,leftDeriv->getId(),rightValue->getId(),err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
	bioExpression* rderiv_l = new bioArithMult(theRepository,patBadId,rightDeriv->getId(),leftValue->getId(),err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
	bioExpression* numerator = new bioArithBinaryMinus(theRepository,patBadId,lderiv_r->getId(),rderiv_l->getId(),err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
	bioExpression* denominator = new bioArithMult(theRepository,patBadId,rightValue->getId(),rightValue->getId(),err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
	bioExpression* result = new bioArithDivide(theRepository,patBadId,numerator->getId(),denominator->getId(),err);
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL;
      }
      return result ;
    }
    else {
      // Left variable, right constant
      bioExpression* l = new bioArithDivide(theRepository,patBadId,leftDeriv->getId(),rightValue->getId(),err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
      return l ;
    }
  }
  else {
    if (rightChild->dependsOf(aLiteralId)) {
      //Left constant, right variable
      bioExpression* r = new bioArithMult(theRepository,patBadId,leftValue->getId(),rightDeriv->getId(),err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL;
	}
	bioExpression* n = new bioArithUnaryMinus(theRepository,patBadId,r->getId(),err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
	bioExpression* d = new bioArithMult(theRepository,patBadId,rightValue->getId(),rightValue->getId(),err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
	bioExpression* result = new bioArithDivide(theRepository,patBadId,n->getId(),d->getId(),err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}

      return result ;
    }
    else {
      //Left constant, right constant
      bioExpression* result = new bioArithConstant(theRepository,patBadId,0) ;
      return result ;
    }
  }

  
  //  patReal result = (rightValue * leftDeriv - leftValue * rightDeriv) / 
  //                 (rightValue * rightValue) ;

}

bioArithDivide* bioArithDivide::getDeepCopy(bioExpressionRepository* rep,
					    patError*& err) const {
  bioExpression* leftClone(NULL) ;
  bioExpression* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(rep,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (leftClone == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(rep,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (rightClone == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
  }

  bioArithDivide* theNode = 
    new bioArithDivide(rep,patBadId,leftClone->getId(),rightClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}

bioArithDivide* bioArithDivide::getShallowCopy(bioExpressionRepository* rep,
					    patError*& err) const {
  bioArithDivide* theNode = 
    new bioArithDivide(rep,patBadId,leftChild->getId(),rightChild->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}




patBoolean bioArithDivide::isStructurallyZero() const {
  return (leftChild->isStructurallyZero()) ;
}

patString bioArithDivide::getExpressionString() const {
  stringstream str ;
  str << '/' ;
  if (leftChild != NULL) {
    str << '{' << leftChild->getExpressionString() << '}' ;
  }
  if (rightChild != NULL) {
    str << '{' << rightChild->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;

}



  

bioFunctionAndDerivatives* bioArithDivide::getNumericalFunctionAndGradient(vector<patULong> literalIds,patBoolean computeHessian,
							   patBoolean debugDeriv, patError*& err) {
  if (result.empty()) {
    result.resize(literalIds.size()) ;
  }
  bioFunctionAndDerivatives* l = leftChild->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  bioFunctionAndDerivatives* r = 
    rightChild->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  patReal rsquare = r->theFunction * r->theFunction ;
  patReal rcube = rsquare * r->theFunction ;
  if (l->theFunction == 0.0) {
    patReal r = rightChild->getValue(patFALSE, patLapForceCompute, err) ;
    // l = 0
    if (r  == 0.0) {
      // l= 0, r = 0
      result.theFunction =  0.0 ;
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	result.theGradient[i] = 0.0 ;
      }
    } 
    else if (r == 1.0) {
      // l = 0, r = 1
      result.theFunction = 0.0 ;
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	result.theGradient[i] = l->theGradient[i] ;
      }
    }
    else {
      // l=0, r != 0, r != 1
      result.theFunction =  0.0 ;
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	if (l->theGradient[i] == 0) {
	  result.theGradient[i] = 0.0 ;
	}
	else {
	  result.theGradient[i] = l->theGradient[i] / r ;
	}
      }
    }
  }
  else {
    // l != 0
    if (r->theFunction == 0.0) {
      // l != 0, r = 0
      result.theFunction =  patMaxReal ;
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	result.theGradient[i] = patMaxReal ;
      }
    } 
    else if (r->theFunction == 1.0) {
      // l != 0, r = 1
      result.theFunction =  l->theFunction ;
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	if (l->theGradient[i] == 0.0) {
	  if (r->theGradient[i] == 0.0) {
	    result.theGradient[i] = 0.0 ;
	  } 
	  else {
	    result.theGradient[i] = - r->theGradient[i] * l->theFunction ;
	  }
	}
	else {
	  if (r->theGradient[i] == 0.0) {
	    result.theGradient[i] = l->theGradient[i] ;
	  } 
	  else {
	    result.theGradient[i] = l->theGradient[i] - r->theGradient[i] * l->theFunction ;
	  }
	}
      }
    }
    else {
      // l != 0, r != 0, r != 1
      result.theFunction =  l->theFunction / r->theFunction ;
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	patReal num = (l->theGradient[i] * r->theFunction -
		       r->theGradient[i] * l->theFunction) ;
	if (num != 0.0) {
	  result.theGradient[i] = num / rsquare ;
	}
	else {
	  result.theGradient[i] = 0.0 ;
	}
      }
    }
    
  }
  if (performCheck) {
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      if (!patFinite(result.theGradient[i])) {
	result.theGradient[i] = patMaxReal ;
      }
    }
  }

  if (result.theHessian != NULL && computeHessian) {
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      for (patULong j = i ; j < literalIds.size() ; ++j) {
	patReal v ;
	if (l->theFunction != 0.0) {
	  patReal rhs = r->theHessian->getElement(i,j,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	  v = - l->theFunction * rhs / rsquare ;
	  if (r->theGradient[i] != 0.0 && r->theGradient[j] != 0.0) {
	    v += 2.0 * l->theFunction * r->theGradient[i] * r->theGradient[j] / rcube ;
	  }
	}
	else {
	  v = 0.0 ;
	}
	patReal lhs = l->theHessian->getElement(i,j,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
	if (lhs != 0.0) {
	  v += lhs / r->theFunction ;
	}
	if (l->theGradient[i] != 0.0 && r->theGradient[j] != 0.0) {
	  v -=  l->theGradient[i] * r->theGradient[j] / rsquare ;
	}
	if (l->theGradient[j] != 0.0 && r->theGradient[i] != 0.0) {
	  v -=  l->theGradient[j] * r->theGradient[i] / rsquare ;
	}
	if (performCheck) {
	  if (!patFinite(v)) {
	    v = patMaxReal ;
	  }
	}
	result.theHessian->setElement(i,j,v,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
      }
    }
  }
  // For debug only 
  // bioFunctionAndDerivatives* findiff = getNumericalFunctionAndFinDiffGradient(literalIds, err)  ;
  // if (err != NULL) {
  //   WARNING(err->describe()) ;
  //   return NULL ;
  // }
  // patVariables error = result.theGradient - findiff->theGradient ;
  // patReal norm = norm2(error) ;
  // if (!finite(norm) || norm > 1.0e-3) {
  //   DEBUG_MESSAGE("Gradient : " << result.theGradient) ;
  //   DEBUG_MESSAGE("Fin diff.: " << findiff->theGradient) ;
  //   DEBUG_MESSAGE("Error:     " << error) ;
  // }

  return &result ;
}

