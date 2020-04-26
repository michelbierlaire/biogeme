//-*-c++-*------------------------------------------------------------
//
// File name : bioLiteral.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Apr 28 10:44:29 2009
//
//--------------------------------------------------------------------

#include "bioLiteral.h"

bioLiteral::bioLiteral(patString theName, patULong theId):name(theName), uniqueId(theId) {
}

patString bioLiteral::getName() const {
  return name ;
}

patULong bioLiteral::getId() const {
  return uniqueId ;
}

patBoolean bioLiteral::operator<(const bioLiteral& x) const {
  return (name < x.name) ;
}

ostream& operator<<(ostream &str, const bioLiteral& x)  {
  str << x.getName() << " (id=" << x.getId() << ")" ;
  return str ;
}
