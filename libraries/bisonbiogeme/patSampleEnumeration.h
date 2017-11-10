//-*-c++-*------------------------------------------------------------
//
// File name : patSampleEnumeration.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Jul 18 09:55:18 2002
//
//--------------------------------------------------------------------

#ifndef patSampleEnumeration_h
#define patSampleEnumeration_h

#include <list>
#include "patError.h"
#include "patString.h"
#include "patDiscreteParameter.h"
#include "patGenerateCombinations.h"

class patSample ;
class patProbaModel ;
class patUniform ;
class patDiscreteParameterProba ;
class patSampleEnuGetIndices ;

/**
   @doc This class implements the capability of simulating a model on the sample   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Thu Jul 18 09:55:18 2002) 
 */

class patSampleEnumeration {

public:
  /**
   */
  patSampleEnumeration(patString file, 
		       patPythonReal** arrayResult,
		       unsigned long resRow,
		       unsigned long resCol,
		       patSample* s,
		       patProbaModel* m,
		       patSampleEnuGetIndices* ei,
		       patUniform* rng) ;
  /**
   */
  ~patSampleEnumeration() ;
  /**
     @param aModel pointer to the probability model
     @param aSample pointer to the sample
   */
  void enumerate(patError*& err) ;

  /**
   */
  patBoolean includeUtilities() const ;

private :
  patString       sampleEnumerationFile ;
  list<patString> dataItemForSampleEnum ;
  patProbaModel*  model ;
  patSample*      sample ;
  patUniform*     randomNumbersGenerator ;


  patReal tmp ;
  patDiscreteParameterProba* theDiscreteParamModel ;

  patPythonReal** resultArray ;
  unsigned long resRow ;
  unsigned long resCol ;

  patSampleEnuGetIndices* theIndices ;
};

#endif
