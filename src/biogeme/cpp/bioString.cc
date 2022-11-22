//-*-c++-*------------------------------------------------------------
//
// File name : bioString.cc
// @date   Tue Apr 10 17:10:47 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioString.h"
#include <vector>
#include <sstream>
#include "bioExceptions.h"
#include "bioTypes.h"
#include "bioDebug.h"

bioString extractParentheses(char openParen,
			     char closeParen,
			     bioString str) {

  if (openParen == '"') {
    if (closeParen != '"') {
      std::stringstream errstr ;
      errstr << "Mismatch of quotation marks. Use \" insteaf of " << closeParen;
      throw bioExceptions(__FILE__,__LINE__,errstr.str()) ;
    }
  }
  else {
    // It may happen that the user has included some characters
    // considered as parenthesis. In this case, they are between two
    // quotation marks. Therefore, before extracting, we change to
    // blank all characters between quotes.
    bioBoolean inQuotes = false ;
    for (std::size_t i = 0 ; i < str.length() ; ++i) {
      if (str[i] == '"') {
	inQuotes = !inQuotes ;
      }
      else {
	if (inQuotes) {
	  str[i] = ' ' ;
	}
      }
    }
  }
  std::size_t firstParen = str.find(openParen) ;
  if (firstParen == bioString::npos) {
    throw bioExceptions(__FILE__,__LINE__,"Open parenthesis not found") ;
  }

  if (openParen == closeParen) {
    std::size_t lastParen = str.rfind(openParen) ;
    return str.substr(firstParen+1,lastParen-firstParen-1) ;

  }
  bioUInt level = 0 ;
  for (std::size_t i = firstParen+1 ; i < str.length() ; ++i) {
    if (str[i] == openParen) {
      ++level ;
    }
    else if (str[i] == closeParen) {
      if (level == 0) {
	return str.substr(firstParen+1,i-1-firstParen) ;
      }
      else {
	--level ;
      }
    }
  }
  throw bioExceptions(__FILE__,__LINE__,"Close parenthesis not found") ;
}



std::vector<bioString> split(const bioString& s, char delimiter) {
   std::vector<bioString> tokens;
   bioString token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}
