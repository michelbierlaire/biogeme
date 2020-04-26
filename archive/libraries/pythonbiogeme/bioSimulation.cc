//-*-c++-*------------------------------------------------------------
//
// File name : bioSimulation.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Sat Feb  6 15:17:28 2010
//
//--------------------------------------------------------------------

#include "bioSimulation.h"

vector<pair<patString,patReal> >* bioSimulation::getSimulation() {
  return &simulation ;
}

