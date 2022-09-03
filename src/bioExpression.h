//-*-c++-*------------------------------------------------------------
//
// File name : bioExpression.h
// @date   Thu Apr 12 11:21:21 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExpression_h
#define bioExpression_h

#include <vector>
#include <map>
#include "bioConst.h"
#include "bioTypes.h"
#include "bioString.h"
#include "bioDerivatives.h"
class bioExpression {
 public:
  bioExpression() ;
  virtual ~bioExpression() ;
  virtual void resetDerivatives() ;
  virtual bioString print(bioBoolean hp = false) const = PURE_VIRTUAL ;
  virtual void setParameters(std::vector<bioReal>* p) ;
  virtual void setFixedParameters(std::vector<bioReal>* p) ;
  virtual void setRowIndex(bioUInt* i) ;
  virtual void setIndividualIndex(bioUInt* i) ;
  virtual void setRandomVariableValuePtr(bioUInt rvId, bioReal* v) ;
  virtual void setDrawIndex(bioUInt* d) ;
  virtual void setData(std::vector< std::vector<bioReal> >* d) ;
  virtual void setMissingData(bioReal md) ;
  virtual void setDataMap(std::vector< std::vector<bioUInt> >* dm) ;
  virtual void setDraws(std::vector< std::vector< std::vector<bioReal> > >* d) ;
  virtual bioReal getValue() ;
  // Returns true is the expression contains at least one literal in
  // the list. Used to simplify the calculation of the derivatives
  virtual bioBoolean containsLiterals(std::vector<bioUInt> literalIds) const ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						       bioBoolean gradient,
						       bioBoolean hessian) = PURE_VIRTUAL ;
  virtual std::map<bioString,bioReal> getAllLiteralValues() ;
 protected:
  std::vector<bioReal>* parameters ;
  std::vector<bioReal>* fixedParameters ;
  bioDerivatives theDerivatives ;
  // Dimensons of the data
  // 1. number of rows
  // 2. number of variables
  std::vector< std::vector<bioReal> >* data;

  // Dimensions of the data map
  // 1. number of individuals
  // 2. two: the first and the last row of  each individual in the dataset
  std::vector< std::vector<bioUInt> >* dataMap;
  
  std::vector<bioExpression*> listOfChildren ;
  // Dimensions of the draws
  // 1. number of individuals
  // 2. number of draws
  // 3. number of draw variables
  std::vector< std::vector< std::vector<bioReal> > >* draws ;
  bioUInt sampleSize ;
  bioUInt numberOfDraws ;
  bioUInt numberOfDrawVariables ;
  bioUInt* rowIndex ;
  bioUInt* individualIndex ;
  bioReal missingData ;
};
#endif
