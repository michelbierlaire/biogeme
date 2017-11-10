//-*-c++-*------------------------------------------------------------
//
// File name : patString.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri Jun 18 13:40:01 1999
//
//--------------------------------------------------------------------

#ifndef patString_h
#define patString_h

#include <string>
#include <sstream>
#include <vector>

using namespace std ;

typedef string patString ;

patString noSpace(const patString& aString) ;

/**
   Replace all occurences of "chain" in "theString" with "with"
 */
patString* replaceAll(patString* theString, patString chain, patString with) ;

/**
   Ceate a new string of size n, where blanks are added before or after depending on the requested justification (1=left, 0=right).
If n is too snall, the string is returned as such without warning.
 */

patString fillWithBlanks(const patString& theStr, 
			 unsigned long n, 
			 short justifyLeft) ;


template <class T>
bool from_string(T& t, 
                 const std::string& s, 
                 std::ios_base& (*f)(std::ios_base&))
{
  std::istringstream iss(s);
  return !(iss >> f >> t).fail();
}

vector<patString> split(patString s, char delimiter) ;

patString extractFileNameFromPath(patString s) ;
patString extractDirectoryNameFromPath(patString s) ;
patString removeFileExtension(patString s) ;
patString getFileExtension(patString s) ;

#endif //patString_h



