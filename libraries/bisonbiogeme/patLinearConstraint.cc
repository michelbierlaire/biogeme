//-*-c++-*------------------------------------------------------------
//
// File name : patLinearConstraint.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri Oct 25 15:22:10 2002
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patLinearConstraint.h"
#include "patConst.h"

ostream& operator<<(ostream &str, const patLinearConstraint& x) {

  patBoolean firstTerm = patTRUE ;

  for (patConstraintEquation::const_iterator i = x.theEquation.begin() ;
       i != x.theEquation.end() ;
       ++i) {
    if (firstTerm) {
      firstTerm = patFALSE ;
      if (i->fact == 1.0) {
	str << i->param ;
      }
      else if (i->fact == -1.0) {
	str << "-" << i->param ;
      }
      else {
	str << i->fact << "*" << i->param ;
      }
    }
    else {
      if (i->fact == 1.0) {
	str << " + " << i->param ;
      }
      else if (i->fact == -1.0) {
	str << " - " << i->param ;
      }
      else {
	if (i->fact >= 0) {
	  str << " + " << i->fact ;
	}
	else {
	  str << " - " << -i->fact ;
	}
	str << "*" << i->param ;
      }
    }
  }
  switch (x.theType) {
  case patLinearConstraint::patEQUAL :
    str << " = " ;
    break ;
  case patLinearConstraint::patLESSEQUAL :
    str << " <= " ;
    break ;
  case patLinearConstraint::patGREATEQUAL :
    str << " >= " ;
    break ;
  default:
    str << " ?? " ;
  }
  str << x.theRHS << endl ;
  return(str) ;
}


patString patLinearConstraint::getFormForPython() {

  stringstream str ;
  patBoolean firstTerm = patTRUE ;

  for (patConstraintEquation::const_iterator i = theEquation.begin() ;
       i != theEquation.end() ;
       ++i) {
    if (firstTerm) {
      firstTerm = patFALSE ;
      if (i->fact == 1.0) {
	str << i->param ;
      }
      else if (i->fact == -1.0) {
	str << "-" << i->param ;
      }
      else {
	str << i->fact << "*" << i->param ;
      }
    }
    else {
      if (i->fact == 1.0) {
	str << " + " << i->param ;
      }
      else if (i->fact == -1.0) {
	str << " - " << i->param ;
      }
      else {
	if (i->fact >= 0) {
	  str << " + " << i->fact ;
	}
	else {
	  str << " - " << -i->fact ;
	}
	str << "*" << i->param ;
      }
    }
  }
  str << " - " << theRHS << "  #"  ;
  switch (theType) {
  case patLinearConstraint::patEQUAL :
    str << " = " ;
    break ;
  case patLinearConstraint::patLESSEQUAL :
    str << " <= " ;
    break ;
  case patLinearConstraint::patGREATEQUAL :
    str << " >= " ;
    break ;
  default:
    str << " ?? " ;
  }
  str << " 0 " << endl ;
  return(patString(str.str())) ;
}
