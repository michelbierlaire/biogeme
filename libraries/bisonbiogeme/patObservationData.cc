//-*-c++-*------------------------------------------------------------
//
// File name : patObservationData.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri Dec  5 16:36:46 2003
//
//--------------------------------------------------------------------

#include <string.h>
#include "patObservationData.h"
#include "patDisplay.h"

void patObservationData::writeBinary(ostream& aStream) {
  aStream.write((char*)&choice,sizeof(choice)) ;
  aStream.write((char*)&weight,sizeof(weight)) ;
  aStream.write((char*)&aggWeight,sizeof(aggWeight)) ;
  aStream.write((char*)&isLast,sizeof(isLast)) ;
  aStream.write((char*)&group,sizeof(group)) ;
  aStream.write((char*)&id,sizeof(id)) ;
  aStream.write((char*)&fileId,sizeof(fileId)) ;
 
  unsigned long nAttr = attributes.size() ;

  aStream.write((char*)&nAttr,sizeof(nAttr)) ;

  for (vector<patAttributes >::const_iterator i = attributes.begin();
       i != attributes.end() ;
       ++i) {
    i->writeAttrBinary(aStream) ;
  }
  
  unsigned long nAvail = availability.size() ;
  aStream.write((char*)&nAvail,sizeof(nAvail)) ;

  for (vector<patBoolean>::const_iterator i = availability.begin() ;
       i != availability.end() ;
       ++i) {
    aStream.write((char*)&(*i),sizeof(*i)) ;
  }

  unsigned long nDraws = draws.size() ;
  aStream.write((char*)&nDraws,sizeof(nDraws)) ;

  for (vector<patVariables>::const_iterator j = draws.begin() ;
       j != draws.end() ;
       ++j) {
    
    unsigned long nCells = j->size() ;
    aStream.write((char*)&nCells,sizeof(nCells)) ;
    for (patVariables::const_iterator i = j->begin() ;
	 i != j->end() ;
	 ++i) {
      aStream.write((char*)&(*i),sizeof(*i)) ;
    }
  }
}

void patObservationData::readBinary(istream& aStream) {
  aStream.read((char *)&choice,sizeof(choice)) ;
  if (aStream.eof()) {
    return ;
  }
  aStream.read((char *)&weight,sizeof(weight)) ;
  aStream.read((char *)&aggWeight,sizeof(aggWeight)) ;
  aStream.read((char *)&isLast,sizeof(isLast)) ;
  aStream.read((char *)&group,sizeof(group)) ;
  aStream.read((char *)&id,sizeof(id)) ;
  aStream.read((char *)&fileId,sizeof(fileId)) ;
 
  unsigned long nAttr ; 
  aStream.read((char *)&nAttr,sizeof(nAttr)) ;
  if (aStream.eof()) {
    return ;
  }

  attributes.erase(attributes.begin(),attributes.end()) ;
  attributes.resize(nAttr) ;

  for (unsigned long i = 0 ;
       i < nAttr ;
       ++i) {
    attributes[i].readAttrBinary(aStream) ;
  }
 
  unsigned long nAvail ;
  aStream.read((char *)&nAvail,sizeof(nAvail)) ;
  availability.erase(availability.begin(),availability.end()) ;
  availability.resize(nAvail) ;

  for (unsigned long i = 0 ;
       i < nAvail ;
       ++i) {
    patBoolean theAvail ;
    aStream.read((char *)&(theAvail),sizeof(theAvail)) ;
    availability[i] = theAvail ;
  }

  unsigned long nDraws ;
  aStream.read((char *)&nDraws,sizeof(nDraws)) ;
  draws.erase(draws.begin(),draws.end()) ;
  draws.resize(nDraws) ;

  for (unsigned long j = 0 ;
       j < nDraws ;
       ++j) {
    
    unsigned long nCells ;
    aStream.read((char *)&nCells,sizeof(nCells)) ;
    draws[j].resize(nCells) ;
    
    for (unsigned long i = 0 ;
	 i < nCells ;
	 ++i) {
      patReal theCell ;
      aStream.read((char *)&theCell,sizeof(theCell)) ;
      draws[j][i] = theCell ;
    }
  }
}

ostream& operator<<(ostream &str, const patObservationData& x) {
  str << "Choice:\t" << x.choice << endl 
      << "Weight:\t" << x.weight << endl
      << "Agg. Weight:\t" << x.aggWeight << endl
      << "Is last agg. obs.:\t" << ((x.isLast)?"yes":"no") << endl
      << "Attributes:" << endl ;
  for (vector<patObservationData::patAttributes>::const_iterator i = 
	 x.attributes.begin() ;
       i != x.attributes.end() ;
       ++i) {
    str << '\t' << *i << endl ;
  }
  str << "Draws:" << endl ;
  for (vector<patVariables>::const_iterator i = x.draws.begin() ;
       i != x.draws.end() ;
       ++i) {
    str << *i << endl ;
  }

  return str ;
}

ostream& operator<<(ostream &str, const patObservationData::patAttributes& x) {
  str << x.name << "=" << x.value ;
  return str ;
}

void patObservationData::patAttributes::writeAttrBinary(ostream& aStream) const {
  unsigned long csize = name.size() ;
  char theName[csize+1] ;
  strcpy(theName,name.c_str()) ;
  theName[csize] = '\0' ;
  aStream.write((char*)&csize,sizeof(csize)) ;
  aStream.write((char*)&theName, sizeof(theName)) ;
  aStream.write((char*)&value,sizeof(value)) ;
}

void patObservationData::patAttributes::readAttrBinary(istream& aStream) {

  unsigned long csize ;
  aStream.read((char *)&csize,sizeof(csize)) ;
  char theName[csize+1] ;
  aStream.read(theName, sizeof(theName)) ;
  name = theName ;
  aStream.read((char *)&value,sizeof(value)) ;
}

