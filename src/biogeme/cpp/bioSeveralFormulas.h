//-*-c++-*------------------------------------------------------------
//
// File name : bioSeveralFormulas.h
// @date   Mon Apr 23 13:53:55 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioSeveralFormulas_h
#define bioSeveralFormulas_h

#include <vector>
#include "bioTypes.h"
#include "bioString.h"
#include "bioFormula.h"

class bioSeveralExpressions ;

class bioSeveralFormulas: public bioFormula {
  friend std::ostream& operator<<(std::ostream &str, const bioSeveralFormulas& x) ;

 public:
  bioSeveralFormulas() ;
  ~bioSeveralFormulas() ;
  void setExpressions(std::vector<std::vector<bioString> > vectOfExpressionsStrings) ;
  void resetExpressions() ;
  bioSeveralExpressions* getExpressions() ;
 private:
  bioSeveralExpressions* theFormulas ;

};


#endif
