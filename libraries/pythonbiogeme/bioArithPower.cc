//-*-c++-*------------------------------------------------------------
//
// File name : bioArithPower.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 12:05:09 2009
//
//--------------------------------------------------------------------

#include <sstream>

#include "patMath.h"
#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "bioArithPower.h"
#include "bioArithConstant.h"
#include "bioArithLog.h"
#include "bioArithCompositeLiteral.h"
#include "bioArithLiteral.h"
#include "bioArithMult.h"
#include "bioArithBinaryPlus.h"
#include "bioArithDivide.h"
#include "bioLiteralRepository.h"
#include "bioArithNotEqual.h"
#include "bioExpressionRepository.h"
#include "bioLiteralRepository.h"

bioArithPower::bioArithPower(bioExpressionRepository* rep,
			     patULong par,
                             patULong left,
                             patULong right,
			     patError*& err)
  : bioArithBinaryExpression(rep, par,left,right,err) {

}


bioArithPower::~bioArithPower() {}

patString bioArithPower::getOperatorName() const {
  return ("^") ;
}

patReal bioArithPower::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  {
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){

    patReal l = leftChild->getValue(prepareGradient, currentLap, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }

    patReal r = rightChild->getValue(prepareGradient, currentLap, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }
    if (performCheck) {
      if (!patFinite(l)) {
        lastValue = patMaxReal;
        lastComputedLap = currentLap;
        return(patMaxReal) ;
      }
    }
    if (l == 0) {
      lastValue = 0;
      lastComputedLap = currentLap;
      return 0 ;
    }
    if (performCheck) {
      if (!patFinite(r)) {
        lastValue = patMaxReal;
        lastComputedLap = currentLap;
        return(patMaxReal) ;
      }
    }
    if (r == 1.0) {
      lastValue = l;
      //return l ;
    }else{

      patULong rint = patULong(r) ;
      if (patReal(rint) == r) {
        patReal result = 1.0 ;
        for (patULong i = 0 ; i < rint ; ++i) {
          result *= l ;
        }
        lastValue = result ;
        //return result ;   
      } else {

        if (performCheck) {
          patReal result =  r * log(l) ;
          if (result >= patLogMaxReal::the()) {
            lastValue = patMaxReal;
            //return(patMaxReal) ;
          } else {
          	patReal ere = exp(result) ;
          	lastValue = ere;
            //return ere ;
          }
        } else {
          lastValue = pow(l,r) ;
          //return pow(l,r) ;
        }
      }
    }

    lastComputedLap = currentLap ;
  }
  return lastValue;
}


bioExpression* bioArithPower::getDerivative(patULong aLiteralId, 
					    patError*& err) const {

  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    WARNING(getInfo()) ;
    return NULL;
  }

  if (!dependsOf(aLiteralId) || leftChild->isStructurallyZero()) {
    bioExpression* result = theRepository->getZero() ;
    return result ;
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

  bioExpression* power = new bioArithPower(theRepository,patBadId,leftValue->getId(),rightValue->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  //  Derivative = 
  // power * [rightValue * leftDeriv / leftValue + rightDeriv * ln(leftValue)]
  
  patBoolean both(patTRUE) ;
  bioExpression* term1(NULL) ;
  bioExpression* term2(NULL) ;
  if (leftChild->dependsOf(aLiteralId)) {
    bioExpression* n = new bioArithMult(theRepository,patBadId,rightValue->getId(),leftDeriv->getId(),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    term1 = new bioArithDivide(theRepository,patBadId,n->getId(),leftValue->getId(),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
  }
  else {
    both = patFALSE ;
  }
  if (rightChild->dependsOf(aLiteralId)) {
    bioExpression* l= new bioArithLog(theRepository,patBadId,leftValue->getId(),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL;
    }
    term2 = new bioArithMult(theRepository,patBadId,rightDeriv->getId(),l->getId(),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
  }
  else {
    both = patFALSE ;
  }
  bioExpression* sum(NULL) ;
  if (both) {
    sum = new bioArithBinaryPlus(theRepository,patBadId,term1->getId(),term2->getId(),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    bioExpression* result = new bioArithMult(theRepository,patBadId,power->getId(),sum->getId(),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    return result ;
  }
  else {
    if (term1 != NULL) {
      bioExpression* result = new bioArithMult(theRepository,patBadId,power->getId(),term1->getId(),err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
      return result ;
    }
    else if (term2 != NULL) {
      bioExpression* result = new bioArithMult(theRepository,patBadId,power->getId(),term2->getId(),err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
      return result ;
    } 
    else {
      // Should never happen
      err = new patErrMiscError("Something's wrong... One of the two terms shoud be non null.");
      WARNING(err->describe()) ;
      return NULL ;
    }
  }

  // bioExpression* result = new bioArithMult(theRepository,patBadId,power->getId(),sum->getId(),err) ;
  // if (err != NULL) {
  //   WARNING(err->describe()) ;
  //   return NULL;
  // }
  // return result ;
}


bioArithPower* bioArithPower::getDeepCopy(bioExpressionRepository* rep,
					  patError*& err) const {
  bioExpression* leftClone(NULL) ;
  bioExpression* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(rep,err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(rep,err) ;
  }

  bioArithPower* theNode = 
    new bioArithPower(rep,patBadId,leftClone->getId(),rightClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return theNode ;
}

bioArithPower* bioArithPower::getShallowCopy(bioExpressionRepository* rep,
					  patError*& err) const {
  bioArithPower* theNode = 
    new bioArithPower(rep,patBadId,leftChild->getId(),rightChild->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return theNode ;
}


patBoolean bioArithPower::isStructurallyZero() const {
  return (leftChild->isStructurallyZero()) ;
}

patString bioArithPower::getExpressionString() const {
  stringstream str ;
  str << '^' ;
  if (leftChild != NULL) {
    str << '{' << leftChild->getExpressionString() << '}' ;
  }
  if (rightChild != NULL) {
    str << '{' << rightChild->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;

}




bioFunctionAndDerivatives* bioArithPower::getNumericalFunctionAndGradient(vector<patULong> literalIds,
patBoolean computeHessian,
									 patBoolean debugDeriv, patError*& err) {

  if (leftDependsOnLiteral.size() == 0) {
    for (vector<patULong>::iterator i = literalIds.begin() ;
	 i != literalIds.end() ;
	 ++i) {
      leftDependsOnLiteral.push_back(getLeftChild()->dependsOf(*i)) ;
      rightDependsOnLiteral.push_back(getRightChild()->dependsOf(*i)) ;
    }
  }

  if (result.empty()) {
    result.resize(literalIds.size()) ;
  }
  bioFunctionAndDerivatives* l = leftChild->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return  NULL ;
  }
  bioFunctionAndDerivatives* r = 
    rightChild->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  if (l->theFunction == 0) {
    result.theFunction = 0.0 ;
  }
  else if (r->theFunction == 0.0) {
    result.theFunction = 1.0 ;
  }
  else if (r->theFunction == 1.0) {
    result.theFunction = l->theFunction ;
  }
  else {
    patULong rint = patULong(r->theFunction) ;
    if (patReal(rint) == r->theFunction) {
      result.theFunction = 1.0 ;
      for (patULong i = 0 ; i < rint ; ++i) {
	result.theFunction *= l->theFunction ;
      }
    }
    else {
      if (performCheck) {
	patReal tmp =  r->theFunction * log(l->theFunction) ;
	if (tmp >= patLogMaxReal::the()) {
	  result.theFunction = patMaxReal ;
	}
	else {
	  result.theFunction = exp(tmp) ;
	}
      }
      else {
	result.theFunction = pow(l->theFunction,r->theFunction) ;
      }
    }
  }
  
  vector<patReal> G(result.theGradient.size(),0.0) ;
  for (patULong i = 0 ; i < literalIds.size() ; ++i) {
    result.theGradient[i] = 0.0 ;
    if (result.theFunction != 0.0) {
      if (leftDependsOnLiteral[i]) {
	if (l->theGradient[i] != 0.0 && r->theFunction != 0.0) {
	  patReal term = l->theGradient[i] * r->theFunction / l->theFunction ;  
  	  G[i] += term ;

	}
      }
      if (rightDependsOnLiteral[i]) {
	if (r->theGradient[i] != 0.0) {
	  G[i] += r->theGradient[i] * log(l->theFunction) ;
	}
      }
      result.theGradient[i] = result.theFunction * G[i] ;
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
	patReal v = G[i] * result.theGradient[j] ;
	if (result.theFunction != 0) {
	  patReal term(0.0) ;
	  patReal hright = r->theHessian->getElement(i,j,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	  if (hright != 0.0) {
	    term += hright * log(l->theFunction) ;
	  }
	  if (l->theGradient[j] != 0.0 && r->theGradient[i] != 0.0) {
	    term += l->theGradient[j] * r->theGradient[i] / l->theFunction ;
	  } 
	  if (l->theGradient[i] != 0.0 && r->theGradient[j] != 0.0) {
	    term += l->theGradient[i] * r->theGradient[j] / l->theFunction ;
	  } 
	  if (l->theGradient[i] != 0.0 && l->theGradient[j] != 0.0) {
	    patReal asquare = l->theFunction * l->theFunction ;
	    term -= l->theGradient[i] * l->theGradient[j] * r->theFunction / asquare ;
	  }
	  patReal hleft = l->theHessian->getElement(i,j,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	  if (hleft != 0.0) {
	    term += hleft * r->theFunction / l->theFunction ;
	  }
	  if (term != 0.0) {
	    v += term * result.theFunction ;
	  }
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
  return &result ;
}

