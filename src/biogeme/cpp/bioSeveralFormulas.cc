//-*-c++-*------------------------------------------------------------
//
// File name : bioSeveralFormulas.cc
// @date   Wed Mar  3 17:31:51 2021
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioSeveralFormulas.h"

#include "bioMemoryManagement.h"
#include "bioDebug.h"
#include "bioSeveralExpressions.h"

bioSeveralFormulas::bioSeveralFormulas(): theFormulas(NULL) {

}

void bioSeveralFormulas::
setExpressions(std::vector<std::vector<bioString> > vectOfExpressionsStrings) {

  std::vector<bioExpression*> exprs ;
  // Process the formulas
  for (std::vector<std::vector<bioString> >::iterator k = vectOfExpressionsStrings.begin() ;
       k != vectOfExpressionsStrings.end() ;
       ++k) {
    bioExpression* lastExpression = NULL ;
    for (std::vector<bioString>::iterator i = k->begin() ;
	 i != k->end() ;
	 ++i) {
      // As the formula is the last in the list, it will
      // be correct at the end of the loop. The other assignments do not
      // matter.
      lastExpression = processFormula(*i) ;
    }
    exprs.push_back(lastExpression) ;
  }
  theFormulas = bioMemoryManagement::the()->get_bioSeveralExpressions(exprs) ;
}

void bioSeveralFormulas::resetExpressions() {
  theFormulas = NULL ;
}

bioSeveralFormulas::~bioSeveralFormulas() {
  
}

bioSeveralExpressions* bioSeveralFormulas::getExpressions() {
  return theFormulas ;
}


std::ostream& operator<<(std::ostream &str, const bioSeveralFormulas& x) {
  if (x.theFormulas != NULL) {
    str << x.theFormulas->print() ;
  }
  return str ;
}

