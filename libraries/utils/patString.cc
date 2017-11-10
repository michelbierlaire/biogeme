//-*-c++-*------------------------------------------------------------
//
// File name : patString.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Nov 21 12:13:05 2001
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patDisplay.h"
#include "patString.h"
patString noSpace(const patString& aString) {
  patString result = aString ;
  for (unsigned long i = 0 ; i < result.size() ; ++i) {
    if (result[i] == ' ') {
      result[i] = '_' ;
    }
  }
  return result ;
}

patString* replaceAll(patString* theString, patString chain, patString with) {

  if (theString == NULL) {
    return NULL ;
  }

  patString::size_type ip = theString->find(chain) ;
 
  while (ip != string::npos ) {
    theString->replace(ip,chain.size(),with) ;
    ip = theString->find(chain) ;
  }
  return theString ;
}

patString fillWithBlanks(const patString& theStr, 
			 unsigned long n, short justifyLeft) {
  unsigned long currentLength = theStr.size() ;
  if (currentLength >= n) {
    return theStr ;
  }

  patString result ;
  patString blanks(n-currentLength,' ') ;
  if (justifyLeft) {
    result = theStr + blanks ; 
  }
  else {
    result = blanks + theStr ;
  }
  return result ;
}

vector<patString> split(patString s, char delimiter) {
  vector<patString> res ;
  size_t last = 0 ;
  size_t next = 0 ;
  while ((next = s.find(delimiter,last)) != patString::npos) {
    res.push_back(s.substr(last,next-last)) ;
    last = next + 1 ;
  }
  res.push_back(s.substr(last)) ;
  return res ;
}

patString extractFileNameFromPath(patString s) {
  size_t found;
  found=s.find_last_of("/\\");
  return (s.substr(found+1)) ;
  
}

patString extractDirectoryNameFromPath(patString s) {
  size_t found;
  found=s.find_last_of("/\\");
  return (s.substr(0,found)) ;
  
}

patString removeFileExtension(patString s) {
    int posDot = s.find_first_of('.') ;
    return s.substr(0,posDot) ;
}

patString getFileExtension(patString s) {
    int posDot = s.find_first_of('.') ;
    return s.substr(posDot+1) ;
}
