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
  bioFormula() ;
  virtual ~bioFormula() ;
  void setExpression(std::vector<bioString> expressionsStrings) ;
  void resetExpression() ;
  virtual bioBoolean isDefined() const ;
  bioExpression* getExpression() ;
  virtual void setParameters(std::vector<bioReal>* p) ;
  virtual void setFixedParameters(std::vector<bioReal>* p) ;
  virtual void setRowIndex(bioUInt* r) ;
  virtual void setIndividualIndex(bioUInt* i) ;
  virtual void setData(std::vector< std::vector<bioReal> >* d) ;
  virtual void setMissingData(bioReal md) ;
  virtual void setDataMap(std::vector< std::vector<bioUInt> >* dm) ;
  virtual void setDraws(std::vector< std::vector< std::vector<bioReal> > >* d) ;
protected:
  std::map<bioString,bioExpression*> expressions ;
  std::map<bioString,bioExpression*> literals ;
  bioReal missingData ;
  bioExpression* processFormula(bioString f) ;
private:
  bioExpression* theFormula ;


};


#endif
