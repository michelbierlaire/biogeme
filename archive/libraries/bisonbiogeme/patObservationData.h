//-*-c++-*------------------------------------------------------------
//
// File name : patObservationData.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Tue Jul 11 22:20:20 2000
//
//--------------------------------------------------------------------

#ifndef patObservationData_h
#define patObservationData_h

#include <map>
#include "patConst.h"
#include "patVariables.h" 
#include "patParameters.h"


/**

@doc Class for data related to one observation.   
   Each element of the attributes vector correspond to one alternative
   It is assumed here that each alternative has the same number of attributes
  
   It is also assumed that alternatives are numbered from zero to nalt-1
   The choice is specified according to this numbering.
  @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Tue Jul 11 22:20:20 2000)
 */
class patObservationData {

public:

  /**
    @doc Data associated with an attribute
   */
  struct patAttributes {

    /**
     */
    patAttributes() : name("__noname"),
		      value(patParameters::the()->getgevMissingValue()) {}

    /**
       Name of the attribute
     */
    patString name ;
    /**
       Value of the attribute
     */
    patReal value ;

    /**
     */
    void readAttrBinary(istream& aStream) ;
    
    /**
     */
    void writeAttrBinary(ostream& aStream) const ;
  };


  /**
   */
  void writeBinary(ostream& aStream) ;

  /**
   */
  void readBinary(istream& aStream) ;

  //  typedef map<unsigned long, attrib> patAttributes ;


  patObservationData() : 
    choice(0),
    aggWeight(1.0),
    isLast(patTRUE),
    weight(1.0),
    group(0),
    id(0),
    fileId(0) {} 

  patObservationData(unsigned long nAlt, unsigned long nAttr, 
		     unsigned long nRandomTerms, 
		     unsigned short nSnpTerms,
		     unsigned long nDraws) : 
    choice(0),
    aggWeight(1.0),
    isLast(patTRUE),
    weight(1.0),
    group(0),
    id(0),
    attributes(nAttr,patAttributes()), 
    availability(nAlt,patFALSE),
    draws(nDraws,patVariables(nRandomTerms)),
    unifDrawsForSnpPolynomial(nDraws,patVariables(nSnpTerms)),
    memory(4 * sizeof(unsigned long)+
	   2 * sizeof(patReal)+
	   sizeof(patBoolean)+
	   nAttr*sizeof(patAttributes) +
	   nAlt*sizeof(patBoolean) +
	   nDraws*nRandomTerms*sizeof(patReal)+
	   nDraws*nSnpTerms*sizeof(patReal))
{}


  unsigned long choice ;
  patReal aggWeight ;
  patBoolean isLast ;
  patReal weight ;
  unsigned long group ;
  unsigned long id ;
  unsigned long fileId ;
  vector<patAttributes > attributes ;
  vector<patBoolean> availability ;
  // Rows = draws, columns = random parameter
  vector<patVariables> draws ;
  // Rows = draws, columns = SNP terms
  vector<patVariables> unifDrawsForSnpPolynomial ;
  size_t memory ;
};

/**
 */
ostream& operator<<(ostream &str, const patObservationData& x) ;

/**
 */
ostream& operator<<(ostream &str, const patObservationData::patAttributes& x) ;



#endif
