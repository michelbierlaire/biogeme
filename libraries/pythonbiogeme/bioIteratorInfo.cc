//-*-c++-*------------------------------------------------------------
//
// File name : bioIteratorInfo.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Jul 14 14:45:26 WEST 2009
//
//--------------------------------------------------------------------

#include <sstream>

#include "bioIteratorInfo.h"
#include "patString.h"
#include "patDisplay.h"
#include "patErrMiscError.h"
#include "patErrOutOfRange.h"
#include "bioIteratorInfoRepository.h"

bioIteratorInfo::bioIteratorInfo() : type(UNDEFINED), theParent(NULL) {

}

// Ctor for iterators at the file level or on another iterator (par) 
bioIteratorInfo::bioIteratorInfo(patString aFile,
				 patString par,
                                 patString child,
                                 patString anIndex, 
                                 bioIteratorType aType) :
  childIteratorName(child),
  datafile(aFile), 
  indexName(anIndex), 
  type(aType),
  theParent(par) {
}
  
  
// Ctor for draws iterator
bioIteratorInfo::bioIteratorInfo(patString aFile) :
  childIteratorName(),
  datafile(aFile),
  indexName(),
  type(DRAW),
  theParent(patString("")) {
}


patString bioIteratorInfo::getInfo() const {
  stringstream ss ;
  switch (type) {
    case META :
      ss << "Iter [" << indexName << "] " ;
      ss << "(" << datafile << ")" ;
      break ;
    case ROW :
      ss << "Iter [" << indexName << "] " ;
      ss << "(" << datafile << ")" ;
      break ;
    case DRAW :
      ss << "Iter [" << indexName << "] " ;
      ss << "(" << datafile << ")" ;
      break ;
    default :
      WARNING("Undefined type of iterator") ;
  }

  return patString(ss.str()) ;
}


ostream& operator<<(ostream &str, const bioIteratorInfo& x) {
  str << x.iteratorName << "[data: " << x.datafile << ", index: " << x.indexName << ", child: " << x.childIteratorName ;
  
  switch (x.type) {
  case META:
    str << ", META" ;
    break ;
  case ROW:
    str << ", ROW" ;
    break ;
 case DRAW:
   str << ", DRAW" ;
   break ;
 case UNDEFINED:
   str << ", UNDEFINED" ;
   break ;
  }
  str << " rows: " ;
  for (vector<patULong>::const_iterator i = x.mapIds.begin() ;
       i != x.mapIds.end() ;
       ++i) {
    if (i != x.mapIds.begin()) {
      str << ", " ;
    }
    str << *i ;
  }
  str << "]" ;
  return str ;
}



void bioIteratorInfo::computePointers(patULong sampleSize,patError*& err) {

  rowPointers.resize(sampleSize) ;
  fill(rowPointers.begin(),rowPointers.end(),patBadId);
  if (!mapIds.empty()) {
    for (patULong i = 0 ; i < mapIds.size()-1 ; ++i) {
      if (mapIds[i] >= rowPointers.size()) {
	WARNING(*this) ;
	err = new patErrOutOfRange<patULong>(mapIds[i],0,rowPointers.size()-1) ;
	WARNING(err->describe()) ;
	return ;
      }
      
      rowPointers[mapIds[i]] = mapIds[i+1] ; 
    }
  }

}

patBoolean bioIteratorInfo::isTopIterator() const {
  return ((type != DRAW) && theParent == patString("")) ;
}

patBoolean bioIteratorInfo::isRowIterator() const {
  return (type == ROW) ;
}

patBoolean bioIteratorInfo::isMetaIterator() const {
  return (type == META) ;
}
patBoolean bioIteratorInfo::isDrawIterator() const {
  return (type == DRAW) ;
}

patBoolean bioIteratorInfo::operator==(const bioIteratorInfo &other) const {
  if (iteratorName != other.iteratorName) {
    return patFALSE ;
  }
  if (datafile != other.datafile) {
    WARNING("BIZARRE: TWO ITERATORS WITH THE SAME NAME (" << iteratorName << ") ON DIFFERENT FILES " << datafile << " and " << other.datafile ) ;
    return patFALSE ;
  }
  if (type != other.type) {
    WARNING("BIZARRE: TWO ITERATORS WITH THE SAME NAME ("<<iteratorName<<"), ON THE SAME FILE ("<<datafile<<"), WITH DIFFERENT TYPES") ;
  }
  return patTRUE ;
}

patBoolean bioIteratorInfo::operator!=(const bioIteratorInfo &other) const {
  return !(*this == other) ;
}

bioIteratorType bioIteratorInfo::getType() const {
  return type ;
}

patULong bioIteratorInfo::getNumberOfIterations() const {
  return mapIds.size() ;
}

vector<patULong> bioIteratorInfo::getMapIds() const {
  return mapIds ;
}
vector<patULong> bioIteratorInfo::getRowPointers() const {
  return rowPointers ;
}

