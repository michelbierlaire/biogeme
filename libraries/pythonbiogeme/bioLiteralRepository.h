//-*-c++-*------------------------------------------------------------
//
// File name : bioLiteralRepository.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Mon Apr 27 16:06:48 2009
//
//--------------------------------------------------------------------

#ifndef bioLiteralRepository_h
#define bioLiteralRepository_h

#include <map>
#include <set>
#include "bioLiteral.h"
#include "bioVariable.h"
#include "bioRandomVariable.h"
#include "bioFixedParameter.h"
#include "bioCompositeLiteral.h"
#include "patError.h"
#include "patVariables.h"
#include "patVectorIterator.h"

/*!
Class gathering all literals, keeping track of their ID,
guaranteeing their uniqueness. Literals can be accessed by name or ID.
Each literal has two IDs.
One unique ID across all literals. Their number start at 10000
The other ID is type specific, unique and consecutive for each type of literal. 
The following types of literals are considered:
Variables: explanatory variables of themodel, usually denoted by x.
Fixed parameters: usually denoted by beta. By fixed, it is meant that they are not random variables.
Random variables: used for mixing of models. 
Composite literals: used when a formula is decomposed. For instance, a formula
      f = log(x + y) / (1 + log(x + y))
can be written as
 z = log(x+y)
 f = z / (1 + z)
where z is the composite literal.



*/

class bioLiteralRepository {
  friend class bioPythonSingletonFactory ;
  friend ostream& operator<<(ostream &str, const bioLiteralRepository& x) ;  
public:
  static bioLiteralRepository* the() ;
  void reset() ;
  // Returns the unique ID and the type specific ID
  pair<patULong,patULong> addVariable(patString theName, patULong colId, patError*& err) ;
 
 // Returns true if the expression has indeed been added, as well as
 // the unique ID and the type specific ID
  pair<patBoolean,pair<patULong,patULong> > addUserExpression(patString theName, patError*& err) ;
   pair<patULong,patULong> addRandomVariable(patString theName, patError*& err) ;
   pair<patULong,patULong> addFixedParameter(patString theName, patReal val, patError*& err) ;
  pair<patULong,patULong> addFixedParameter(patString theName, patReal val, patReal lb, patReal up, patBoolean fixed, patString latexName, patError*& err) ;
  pair<patULong,patULong> addCompositeLiteral(patString name, patError*& err) ;
  // A default name is given by biogeme
  pair<patULong,patULong> addCompositeLiteral(patError*& err) ;
  patULong addFixedParameter(bioFixedParameter p) ;
  patULong getLiteralId(patString theName, patError*& err) const ;
  patULong getLiteralId(const bioLiteral* theLiteral, patError*& err) const ;
  patString getName(patULong theId, patError*& err) ;

  patULong getNumberOfVariables() const ;
  patULong getNumberOfRandomVariables() const ;
  patULong getNumberOfParameters() const ;
  patULong getNumberOfEstimatedParameters() ;
  patULong getNumberOfCompositeLiterals() ;
  // If all is patFALSE, only non fixed variables are returned
  patVariables getBetaValues(patBoolean all=patTRUE) const ;
  patReal getBetaValue(patString name, patError*& err) ;
  patVariables getLowerBounds() const ;
  patVariables getUpperBounds() const ;
  void setBetaValue(patString name, patReal value, patError*& err) ;
  void setBetaValues(patVariables b, patError*& err) ;
  // return the literal ids of the beta in the right order. If all is
  // patFALSE, only non fixed betas are returned
  vector<patULong> getBetaIds(patBoolean all, patError*& err) const ;
  // If all = patTRUE, the ID corresponds to the list of fixed parameters
  // If all = patFALSE, the ID ocrresponds to the list of parameters to be estimated.
  patString getBetaName(patULong betaId, patBoolean all,patError*& err) ;
  // Returns the ID in the fixed parameter vector
  pair<patULong,patULong>  getBetaParameter(patString name, patError*& err) ;
  // Returns the ID in the composite literal vector
  pair<patULong,patULong>  getCompositeLiteral(patULong literalId, patError*& err)  ;
  // Returns the ID in the composite literal vector
  pair<patULong,patULong>  getCompositeLiteral(patString name, patError*& err)  ;
  // Returns the ID in the literal and in the variable vector
  pair<patULong,patULong>  getVariable(patString name, patError*& err) const ;
  // Returns the column ID for a variable. This can be either a real
  // column, or a virtual one corresponding to user expressions.
  patULong getColumnIdOfVariable(patString name,patError*& err) const ;
  // Returns the ID in the random variable vector
  pair<patULong,patULong>  getRandomVariable(patString name, patError*& err) ;
  void setFlag(patULong id) ;
  void unsetFlag(patULong id) ;
  patBoolean isFlagSet(patULong id) const ;
  void resetAllFlags() ;
  patString printFixedParameters(patBoolean estimation=patFALSE) const ;

  const bioVariable* theVariable(patULong typeSpecificId) const ;
  const bioCompositeLiteral* theComposite(patULong typeSpecificId) const ;
  const bioFixedParameter* theParameter(patULong typeSpecificId) const  ;
  const bioRandomVariable* theRandomVariable(patULong typeSpecificId) const ;
  // Returns the unique literal id given the type-specific id
  patULong getLiteralId(patULong typeSpecificId, bioLiteral::bioLiteralType type) ;
  const bioLiteral* getLiteral(patULong id, patError*& err) const ;
//   bioLiteral* getLiteral(patString name, patError*& err) const ;

  patIterator<bioFixedParameter*>* getIteratorFixedParameters() ;
  patIterator<bioFixedParameter*>* getSortedIteratorFixedParameters() ;

  pair<patULong,bioLiteral::bioLiteralType> getIdAndType(patULong uniqueId,patError*& err) const ;

  patReal getCompositeValue(patULong litId, patULong threadId, patError*& err) ;
  void setCompositeValue(patReal v, patULong litId, patULong threadId, patError*& err) ;
  patReal getRandomVariableValue(patULong litId, patULong threadId, patError*& err) ;
  void setRandomVariableValue(patReal v, patULong litId, patULong threadId, patError*& err) ;

  // Should be called when all composite literals have been registered  

  void prepareMemory() ;

  int size();

  patULong getLastColumnId() const;
private:
  bioLiteralRepository() ;
  vector<bioVariable> listOfVariables ;
  vector<bioFixedParameter> listOfFixedParameters ;
  // The sorted list is used only for output and created by the
  // getSortedIteratorFixedParameters() method.
  vector<bioFixedParameter> sortedListOfFixedParameters ;
  vector<bioRandomVariable> listOfRandomVariables ;
  // Composite literals are intermediate variables designed to simplify expressions
  vector<bioCompositeLiteral> listOfCompositeLiterals ;
  map<patULong,patBoolean> flags ;
  map<patString,patULong> listOrganizedByNames ;
//   map<patString,patULong> idsOfVariables ;
//   map<patString,patULong> idsOfRandomVariables ;
//   map<patString,patULong> idsOfParameters ;
//   map<patString,patULong> idsOfCompositeLiterals ;

  // Given a unique id, store thr type ans type specific id of the literal
  map<patULong,pair<patULong,bioLiteral::bioLiteralType> > theIdAndTypes ;

  // First: index in the vector listOfFixedParameters
  // Second: index in the set of decision variables for the optimization problem
  map<patULong,patULong> parametersToBeEstimated ;

  vector< vector<patReal> > theCompositeValues ;
  // Used for integration
  vector< vector<patReal> > theRandomValues ;

  set<patString> userExpressions ;
};

#endif
