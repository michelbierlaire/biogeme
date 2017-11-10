//-*-c++-*------------------------------------------------------------
//
// File name : patArithUnifRandom.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Tue Dec 23 13:59:00 2003
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include "patDisplay.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "patArithUnifRandom.h"

#include "patValueVariables.h"

patArithUnifRandom::patArithUnifRandom(patArithNode* par) : 
  patArithRandom(par)
{
  
}

patArithUnifRandom::~patArithUnifRandom() {

}

patArithNode::patOperatorType patArithUnifRandom::getOperatorType() const {
  return patArithNode::UNIRANDOM_OP ;
}
patString patArithUnifRandom::getOperatorName() const {
  stringstream str ;
  str << locationParameter << " { " << scaleParameter << " } " ;
  return patString(str.str()) ;
}

patReal patArithUnifRandom::getValue(patError*& err) const {



  //  DEBUG_MESSAGE("-------------------------") ;

  patReal result(0.0) ;
  for (patUtilFunction::const_iterator i = linearExpression.begin() ;
       i != linearExpression.end() ;
       ++i) {

    patReal beta = patValueVariables::the()->getVariable(i->betaIndex,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    patReal x ;
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
//       DEBUG_MESSAGE("Random index = " << i->xIndex) ;
//       DEBUG_MESSAGE("Beta index = " << i->betaIndex) ;
//       DEBUG_MESSAGE(i->beta << "*" << i->x << "=" << beta << "*" << x)
    }
    
    result += beta * x ;
  }
  return result ;
}

patReal patArithUnifRandom::getDerivative(unsigned long index, 
				      patError*& err) const {
  
  for (patUtilFunction::const_iterator i = linearExpression.begin() ;
       i != linearExpression.end() ;
       ++i) {
    if (i->betaIndex == index) {
      patReal x ;
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
    
patString patArithUnifRandom::getExpression(patError*& err) const {
  stringstream str ;
  str << getOperatorName() ;
//       str << "[" ;
//   for (vector<patString>::const_iterator i = covariance.begin() ;
//        i != covariance.end() ;
//        ++i) {
//     if (i != covariance.begin()) {
//       str << "," ;
//     }
//     str << *i ;
//   }
//   str << "]" ;
//   if (linearExpression!= NULL) {
//     str << " {LINEAR_EXP=" << *linearExpression << "}" ;
//   }
  return patString(str.str()) ;
}

void patArithUnifRandom::setLinearExpression(const patUtilFunction& u) {
  linearExpression = u ;
}


void patArithUnifRandom::replaceInLiterals(patString subChain, patString with) {
  replaceAll(&locationParameter,subChain,with) ;
  replaceAll(&scaleParameter,subChain,with) ;
  for (list<patUtilTerm>::iterator i = linearExpression.begin() ;
       i != linearExpression.end() ;
       ++i) {
    replaceAll(&i->beta,subChain,with) ;
    replaceAll(&i->x,subChain,with) ;
  }
}

patArithUnifRandom* patArithUnifRandom::getDeepCopy(patError*& err) {
  patArithUnifRandom* theNode = new patArithUnifRandom(NULL) ;
  theNode->leftChild = NULL ;
  theNode->rightChild = NULL ;
  theNode->linearExpression = linearExpression ;
  theNode->locationParameter = locationParameter ;
  theNode->scaleParameter = scaleParameter ;
  return theNode ;
}

patUtilFunction* patArithUnifRandom::getLinearExpression() {
  return &linearExpression ;
}

void patArithUnifRandom::addTermToLinearExpression(patUtilTerm term) {
  linearExpression.push_back(term) ;
}

patDistribType patArithUnifRandom::getDistribution() {
  return UNIF_DIST ;
}

void patArithUnifRandom::addCovariance(patString c) {  
  WARNING(*this << ": NO COVARIANCE ALLOWED FOR UNIFORM DISTRIBUTION") ;
  exit(-1) ;
}


patString patArithUnifRandom::getGnuplot(patError*& err) const {
  stringstream str ;
  str << "No GNUPLOT syntax for random variables: " << *this ;
  err = new patErrMiscError(str.str()) ;
  WARNING(err->describe()) ;
  return patString() ;
}

patString patArithUnifRandom::getCppCode(patError*& err) {
  err = new patErrMiscError("Not yet implemented") ;
  WARNING(err->describe()) ;
  return patString() ;
}

patString patArithUnifRandom::getCppDerivativeCode(unsigned long index, patError*& err) {
  err = new patErrMiscError("Not yet implemented") ;
  WARNING(err->describe()) ;
  return patString() ;

}
