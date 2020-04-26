//-*-c++-*------------------------------------------------------------
//
// File name : bioExpressionRepository.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Sat Sep 12 15:56:25 2009
//
//--------------------------------------------------------------------

#ifndef bioExpressionRepository_h
#define bioExpressionRepository_h

#include "bioExpression.h"
#include <map>

class  bioExpressionRepository {

public:
  bioExpressionRepository(patULong threadId) ;
  patULong addExpression(bioExpression* exp) ;
  bioExpression* getExpression(patULong id) ;
  patString printList() ;
  
  bioExpression* getZero()  ;
  bioExpression* getOne()  ;
  patULong getThreadId() const;
  patString check(patError*& err) const ;
  bioExpressionRepository* getCopyForThread(patULong threadId, patError*& err) ;
  patULong getNbrOfExpressions() const ;

  bioExpression* getExpressionByPythonID(patString msHash);
  void addIDtoMap(patString hash, patULong id);
  map<patString,int> pyIDtorepID;
  void setReport(bioReporting* r) ;
private:
  vector<bioExpression*> theRepository ;
  patULong oneId ;
  patULong zeroId ;
  patULong theThreadId ;
  
} ;

#endif
