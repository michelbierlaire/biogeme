//-*-c++-*------------------------------------------------------------
//
// File name : bioRandomDraws.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Thu Jul 30 16:45:35 2009
//
//--------------------------------------------------------------------

#ifndef bioRandomDraws_h
#define bioRandomDraws_h

#include <map>
#include "patError.h"
#include "patString.h"
#include "patVariables.h"

class bioDrawIterator ;
class patUniform ;
class patRandomNumberGenerator ;
class bioSample ;
class bioExpression ; 
class bioRandomDraws {

  friend ostream& operator<<(ostream& str, const bioRandomDraws& x) ;
  friend class bioPythonSingletonFactory ;

 public:
  typedef enum {
    NORMAL,
    TRUNCNORMAL,
    UNIFORM,
    UNIFORMSYM,
    USERDEFINED,
    UNDEFINED
  } bioDrawsType ;
  
typedef struct bioDraw {

  friend ostream& operator<<(ostream& stream, const bioDraw& d) ;

  friend ostream& operator<<(ostream& stream, const bioDrawsType& d) ;

  bioDraw(patString name, bioRandomDraws::bioDrawsType type, patString id) ;
  patString theName ;
  bioDrawsType theType ;
  patString theId ;
} bioDraw ;



public:
  static bioRandomDraws* the() ;
  patULong addRandomVariable(patString name,
			     bioDrawsType type, 
			     patString index, 
			     bioExpression* theExpr, 
			     patString iteratorName,
			     patError*& err) ;
  void generateDraws(bioSample* theSample, patError*& err) ;
  //  vector< vector<patReal> >*  getRow(patULong rowId, patError*& err) ;
  patReal**  getRow(patULong rowId, patError*& err) ;
  patReal**  getUniformRow(patULong rowId, patError*& err) ;
  void setDraws(map<patString, bioDraw >* d, patError*& err) ;
  patULong getColId(patString name, patError*& err) ;
  patString getIndex(patULong colId, patError*& err) ;
  bioDrawsType getType(patULong colId, patError*& err) ;
  bioDrawIterator* createIterator(patError*& err) ;
  patBoolean hasRandomVariables() const ;
  void populateListOfIds(bioSample* aSample, patError*& err) ;
  patULong nbrOfDraws() const ;
  patULong addUserDraws(patString name,
			bioExpression* theUserExpression, 
			patString iteratorName,
			patError*& err) ;
  ~bioRandomDraws() ;
  patUniform* getUniformGenerator() ;
  patString getTypeName(bioDrawsType t) ;
 private:
  bioRandomDraws() ;
  void initDraws(patError*& err) ;
  // Names of the random variables
  vector<patString> names ;
  // Names of the column in the data file defining the
  // index. Typically, the id of an observation or of an individual.
  vector<patString> indexNames ;
  // Index within a data row of the above index.
  vector<patULong> indexIds ;
  // Type of each random variable (normal or uniform)
  vector<bioDrawsType> types ;
  // Expressions for user defined draws. 
  vector<bioExpression*> expressions ;
  // Iterators for user defined draws
  vector<patString> iteratorNames ;

  // For each random variables, list of the values thatthe index can
  // take. For instance, the list of observation ids, or the list of
  // individualids.
  vector<vector<patULongLong> > listOfIds ;
  // Database contains draws[drawNumber][beta][individualId]
  patULong betaDim ;
  patULong idDim ;
  patReal ***draws ;
  patReal ***uniformDraws ;
  //  vector<vector< vector<patReal> > > draws ;
  //  vector<vector< map<patULong,patReal> > > draws ;
  map<patString,patULong> idFromName ;
  // Just for reporting. Translate the type of draw into a string
  patUniform* rng ;
  patRandomNumberGenerator* theNormalGenerator ;
  patRandomNumberGenerator* theRectangularGenerator ;
  patRandomNumberGenerator* theZeroOneGenerator ;
  patULong drawDim ;
  patString iteratorName ;
  patBoolean drawsInitialized ;
};
#endif
