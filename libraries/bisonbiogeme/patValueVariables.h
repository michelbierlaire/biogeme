//-*-c++-*------------------------------------------------------------
//
// File name : patValueVariables.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Nov 23 14:43:48 2000
//
//--------------------------------------------------------------------

#ifndef patValueVariables_h
#define patValueVariables_h

#include <map>
#include "patError.h"
#include "patVariables.h"
#include "patIndividualData.h"

/**
 @doc This class, implemented as a singleton, is aimed at proving specific values for the variables appearing in the arithmetic expression
 @author Michel Bierlaire, EPFL (Thu Nov 23 14:43:48 2000)
 @see patArithNode
 */

class patValueVariables {

  friend class patBisonSingletonFactory ;
public:

/**
 */
  friend ostream& operator<<(ostream &str, const patValueVariables& x) ;

  /**
     @return pointer to the single instance of the class
   */
  static patValueVariables* the() ;

  /**
   */
  void setValue(patString variable, patReal value) ;
  
  /**
   */
  void setAttributes(vector<patObservationData::patAttributes>* y) ;

  /**
   */
  void setRandomDraws(patVariables* y) ;

  /**
   */
  patReal getValue(patString variable,patError*& err) ;

  /**
   */
  patReal getAttributeValue(unsigned long attrId,patError*& err) ;

  /**
   */
  patReal getRandomDrawValue(unsigned long attrId,patError*& err) ;

  /**
   */
  void setVariables(patVariables* y) ;

  /**
   */
  patReal getVariable(unsigned long index, patError*& err) ;

  /**
   */
  patBoolean areAttributesAvailable() ;
  
  /**
   */
  patBoolean areVariablesAvailable() ;

private:

  map<patString,patReal> values ; 
  patValueVariables() ;
  patVariables* x ;
  vector<patObservationData::patAttributes>* attributes;
  patVariables* randomDraws;
};


#endif
