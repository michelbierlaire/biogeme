//-*-c++-*------------------------------------------------------------
//
// File name : patNetworkGevNode.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Jan 29 10:42:54 2003
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patNetworkGevNode.h"
#include "patErrMiscError.h"
#include "patDisplay.h"

patBoolean patNetworkGevNode::isAlternativeRelevant(unsigned long index) {
  std::set<unsigned long> relevant = getRelevantAlternatives() ;
  std::set<unsigned long>::iterator found = relevant.find(index) ;
  return (found != relevant.end()) ;
}

void patNetworkGevNode::setNbrParameters(unsigned long n) {
  parameters = n ;
}

unsigned long patNetworkGevNode::getNbrParameters() {
  return parameters ;
}

void patNetworkGevNode::compute(const patVariables* x,
				const patVariables* param,
				const patReal* mu, 
				const vector<patBoolean>& available,
				patBoolean computeSecondDerivatives,
				patError*& err) {

}

void patNetworkGevNode::generateCppCode(ostream& cppFile, 
					patBoolean derivatives, 
					patError*& err) {
  err = new patErrMiscError("Not yet implemented") ;
  WARNING(err->describe()) ;
  return ;
}
