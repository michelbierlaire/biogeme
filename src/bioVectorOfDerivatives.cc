//-*-c++-*------------------------------------------------------------
//
// File name : bioVectorOfDerivatives.cc
// @date   Wed Oct 20 18:01:24 2021
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#include "bioDebug.h"
#include "bioVectorOfDerivatives.h"

bioVectorOfDerivatives::bioVectorOfDerivatives():
  wg(false),
  wh(false),
  wbhhh(false) {

}
  
void bioVectorOfDerivatives::onlyOne(bioDerivatives d) {
  theDerivatives.clear() ;
  theDerivatives.push_back(d) ;
}

void bioVectorOfDerivatives::aggregate(bioVectorOfDerivatives deriv) {
  for (std::vector< bioDerivatives >::iterator d = deriv.theDerivatives.begin() ;
       d != deriv.theDerivatives.end() ;
       ++d) {
    aggregate(*d) ;
  }
}

void bioVectorOfDerivatives::disaggregate(bioVectorOfDerivatives deriv) {
  for (std::vector< bioDerivatives >::iterator d = deriv.theDerivatives.begin() ;
       d != deriv.theDerivatives.end() ;
       ++d) {
    disaggregate(*d) ;
  }
}


void bioVectorOfDerivatives::resizeAll(bioUInt n) {
  for (std::vector<bioDerivatives>::iterator d = theDerivatives.begin() ;
       d != theDerivatives.end() ;
       ++d) {
    d->resize(n) ;
  }
}

void bioVectorOfDerivatives::setEverythingToZero() {
  for (std::vector<bioDerivatives>::iterator d = theDerivatives.begin() ;
       d != theDerivatives.end() ;
       ++d) {
    d->setEverythingToZero() ;
  }

}

void bioVectorOfDerivatives::setDerivativesToZero() {
  for (std::vector<bioDerivatives>::iterator d = theDerivatives.begin() ;
       d != theDerivatives.end() ;
       ++d) {
    d->setDerivativesToZero() ;
  }

}

void bioVectorOfDerivatives::dealWithNumericalIssues() {
  for (std::vector<bioDerivatives>::iterator d = theDerivatives.begin() ;
       d != theDerivatives.end() ;
       ++d) {
    d->dealWithNumericalIssues() ;
  }
}

void bioVectorOfDerivatives::set_with_g(bioBoolean yes) {
  for (std::vector<bioDerivatives>::iterator d = theDerivatives.begin() ;
       d != theDerivatives.end() ;
       ++d) {
    d->with_g = yes ;
  }
  wg = yes ;
}

void bioVectorOfDerivatives::set_with_h(bioBoolean yes) {
  for (std::vector<bioDerivatives>::iterator d = theDerivatives.begin() ;
       d != theDerivatives.end() ;
       ++d) {
    d->with_h = yes ;
  }
  wh = yes ;
}

void bioVectorOfDerivatives::set_with_bhhh(bioBoolean yes) {
  for (std::vector<bioDerivatives>::iterator d = theDerivatives.begin() ;
       d != theDerivatives.end() ;
       ++d) {
    d->with_bhhh = yes ;
  }
  wbhhh = yes ;
}

bioBoolean bioVectorOfDerivatives::with_g() const {
  return wg ;
}

bioBoolean bioVectorOfDerivatives::with_h() const {
  return wh ;
}

bioBoolean bioVectorOfDerivatives::with_bhhh() const {
  return wbhhh ;
}

bioUInt bioVectorOfDerivatives::getSize() const {
  if (theDerivatives.size() == 0) {
    return 0 ;
  }
  return theDerivatives[0].getSize() ;
}

void bioVectorOfDerivatives::clear() {
  theDerivatives.clear() ;
}

void bioVectorOfDerivatives::aggregate(bioDerivatives d) {
  if (with_bhhh()) {
    d.computeBhhh() ;
  }
  if (theDerivatives.empty()) {
    theDerivatives.push_back(d) ;
  }
  else {
    theDerivatives[0] += d ;
  }
}

void bioVectorOfDerivatives::disaggregate(bioDerivatives d) {
  if (with_bhhh()) {
    d.computeBhhh() ;
  }
  theDerivatives.push_back(d) ;
}

