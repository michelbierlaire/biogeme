//-*-c++-*------------------------------------------------------------
//
// File name : patArithNormalRandom.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Tue Mar  4 09:39:26 2003
//
//--------------------------------------------------------------------
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <sstream>
#include "patDisplay.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "patArithNormalRandom.h"
#include "patBetaLikeParameter.h"
#include "patValueVariables.h"
#include "patModelSpec.h"

patArithNormalRandom::patArithNormalRandom(patArithNode* par) : 
  patArithRandom(par)
{
  
}

patArithNormalRandom::~patArithNormalRandom() {

}

patArithNode::patOperatorType patArithNormalRandom::getOperatorType() const {
  return patArithNode::RANDOM_OP ;
}
patString patArithNormalRandom::getOperatorName() const {
  stringstream str ;
  str << locationParameter << " [ " << scaleParameter << " ] " ;
  return patString(str.str()) ;
}

patReal patArithNormalRandom::getValue(patError*& err) const {

  //  DEBUG_MESSAGE("-------------------------") ;

  patReal x ;
  patReal result = 0.0 ;
  for (patUtilFunction::const_iterator i = linearExpression.begin() ;
       i != linearExpression.end() ;
       ++i) {

    patReal beta = patValueVariables::the()->getVariable(i->betaIndex,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    if (i == linearExpression.begin()) {
      // The first term is the mean, which must be multiplied by 1.0
      x = 1.0 ;
    }
    else {
      x = patValueVariables::the()->getRandomDrawValue(i->xIndex,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }
    }
    
    result += beta * x ;
  }
  return result ;
}

patReal patArithNormalRandom::getDerivative(unsigned long index, 
				      patError*& err) const {
  
  patReal x ;
  for (patUtilFunction::const_iterator i = linearExpression.begin() ;
       i != linearExpression.end() ;
       ++i) {
    if (i->betaIndex == index) {
      if (i == linearExpression.begin()) {
      // The first term is the mean, which must be multiplied by 1.0
	x = 1.0 ;
      }
      else {
	x = patValueVariables::the()->getRandomDrawValue(i->xIndex,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return patReal() ;
	}
      }
      return x ;
    }
  }
  return 0.0 ;
}
    
patString patArithNormalRandom::getExpression(patError*& err) const {
  stringstream str ;
  str << getOperatorName() ;
  return patString(str.str()) ;
}

void patArithNormalRandom::addCovariance(patString c) {
  covariance.push_back(c) ;
}

void patArithNormalRandom::setLinearExpression(const patUtilFunction& u) {
  linearExpression = u ;
}


void patArithNormalRandom::replaceInLiterals(patString subChain, patString with) {
  replaceAll(&locationParameter,subChain,with) ;
  replaceAll(&scaleParameter,subChain,with) ;
  for (vector<patString>::iterator i = covariance.begin() ;
       i != covariance.end() ;
       ++i) {
    replaceAll(&(*i),subChain,with) ;
  }
  for (list<patUtilTerm>::iterator i = linearExpression.begin() ;
       i != linearExpression.end() ;
       ++i) {
    replaceAll(&i->beta,subChain,with) ;
    replaceAll(&i->x,subChain,with) ;
  }
}

patArithNormalRandom* patArithNormalRandom::getDeepCopy(patError*& err) {
  patArithNormalRandom* theNode = new patArithNormalRandom(NULL) ;
  theNode->leftChild = NULL ;
  theNode->rightChild = NULL ;
  theNode->linearExpression = linearExpression ;
  theNode->locationParameter = locationParameter ;
  theNode->scaleParameter = scaleParameter ;
  theNode->covariance = covariance ;
  return theNode ;
}

patUtilFunction* patArithNormalRandom::getLinearExpression() {
  return &linearExpression ;
}

void patArithNormalRandom::addTermToLinearExpression(patUtilTerm term) {
  linearExpression.push_back(term) ;
}

patDistribType patArithNormalRandom::getDistribution() {
  return NORMAL_DIST ;
}

patString patArithNormalRandom::getGnuplot(patError*& err) const {
  stringstream str ;
  str << "No GNUPLOT syntax for random variables: " << *this ;
  err = new patErrMiscError(str.str()) ;
  WARNING(err->describe()) ;
  return patString() ;
}

patString patArithNormalRandom::getCppCode(patError*& err) {

  stringstream str ;
  patBoolean something(patFALSE) ;
  for (patUtilFunction::const_iterator i = linearExpression.begin() ;
       i != linearExpression.end() ;
       ++i) {
    
    patBoolean found ;
    patBetaLikeParameter beta = patModelSpec::the()->getBeta(i->beta,&found) ;
    if (!found) {
      stringstream str ;
      str << "Unknown parameter " << i->beta ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patString() ;
    }
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patString() ;
    }
    if (i == linearExpression.begin()) {
      // The first term is the mean, which must be multiplied by 1.0
      if (beta.isFixed) {
	if (beta.defaultValue != 0.0) {
	  str << beta.defaultValue ;
	  something = patTRUE ;
	}
      }
      else {
	str << " (*x)[" << beta.index << "]" ;
	something = patTRUE ;
      }
    }
    else {
      if (beta.isFixed) {
	if (beta.defaultValue != 0.0) {
	  str << " + " << beta.defaultValue ;
	  str << " * observation->draws[drawNumber-1][" << i->xIndex << "]" ;
	  something = patTRUE ;
	}
      }
      else {
	str << " + (*x)[" << beta.index << "]" ;
	str << " * observation->draws[drawNumber-1][" << i->xIndex << "]" ;
	something = patTRUE ;
      }
    }
  }

  if (!something) {
    return patString("0.0") ;
  }
  else {
    return patString(str.str()) ;
  }
}

patString patArithNormalRandom::getCppDerivativeCode(unsigned long index, patError*& err) {

  stringstream str ;
  patBoolean something(patFALSE) ;
  for (patUtilFunction::const_iterator i = linearExpression.begin() ;
       i != linearExpression.end() ;
       ++i) {
    
    patBoolean found ;
    patBetaLikeParameter beta = patModelSpec::the()->getBeta(i->beta,&found) ;
    if (!found) {
      stringstream str ;
      str << "Unknown parameter " << i->beta ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patString() ;
    }
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patString() ;
    }
    if (i == linearExpression.begin()) {
      // The first term is the mean, which is multiplied by 1.0
      if (!beta.isFixed) {
	if (beta.index == index) {
	  str << "1.0" ;
	  something = patTRUE ;
	}
      }
    }
    else {
      if (!beta.isFixed) {
	if (beta.index == index) {
	  str << "observation->draws[drawNumber-1][" << i->xIndex << "]" ;
	  something = patTRUE ;
	}
      }
    }
  }
  if (!something) {
    return patString("0.0") ;
  }
  else {
    return patString(str.str()) ;
  }

}

patBoolean patArithNormalRandom::isDerivativeStructurallyZero(unsigned long index, patError*& err) {
  return patFALSE ;
}
