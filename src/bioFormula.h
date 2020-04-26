//-*-c++-*------------------------------------------------------------
//
// File name : bioFormula.h
// @date   Mon Apr 23 13:53:55 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioFormula_h
#define bioFormula_h

#include <vector>
#include <map>
#include "bioTypes.h"
#include "bioString.h"

class bioExpression ;

class bioFormula {
  friend std::ostream& operator<<(std::ostream &str, const bioFormula& x) ;

 public:
  bioFormula(std::vector<bioString> expressionsStrings) ;
  ~bioFormula() ;
  bioExpression* getExpression() ;
  void setParameters(std::vector<bioReal>* p) ;
  void setFixedParameters(std::vector<bioReal>* p) ;
  void setRowIndex(bioUInt* r) ;
  void setIndividualIndex(bioUInt* i) ;
  void setData(std::vector< std::vector<bioReal> >* d) ;
  void setMissingData(bioReal md) ;
  void setDataMap(std::vector< std::vector<bioUInt> >* dm) ;
  void setDraws(std::vector< std::vector< std::vector<bioReal> > >* d) ;
 private:
  bioExpression* processFormula(bioString f) ;
  std::map<bioString,bioExpression*> expressions ;
  std::map<bioString,bioExpression*> literals ;
  bioExpression* theFormula ;
  bioReal missingData ;


};


#endif
