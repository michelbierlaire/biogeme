//-*-c++-*------------------------------------------------------------
//
// File name : bioConstraints.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Sun Dec 20 11:05:42 2009
//
//--------------------------------------------------------------------

#include "bioConstraints.h"

bioConstraints::~bioConstraints() {

}

vector<bioConstraintWrapper*>* bioConstraints::getConstraints() {
  return &constraints ;
}
