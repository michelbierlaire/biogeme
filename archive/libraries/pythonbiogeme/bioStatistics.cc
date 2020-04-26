//-*-c++-*------------------------------------------------------------
//
// File name : bioStatistics.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Sat Dec 19 19:25:35 2009
//
//--------------------------------------------------------------------

#include "bioStatistics.h"

vector<pair<patString,patReal> >* bioStatistics::getStatistics() {
  return &statistics ;
}
