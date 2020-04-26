//-*-c++-*------------------------------------------------------------
//
// File name : patArithRandom.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Tue Dec 23 14:31:03 2003
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithRandom.h"
#include "patDisplay.h"

patArithRandom::patArithRandom(patArithNode* par) :
  patArithNode(par,NULL,NULL) ,
  isPanel(patFALSE),
  locationParameter("???"),
  scaleParameter("???") {
  
}

void patArithRandom::setLocationParameter(patString m) {
  locationParameter = m ;
}

void patArithRandom::setScaleParameter(patString s) {
  scaleParameter = s ;
}

patString patArithRandom::getLocationParameter() {
  return locationParameter ;
}

patString patArithRandom::getScaleParameter() {
  return scaleParameter ;
}

vector<patString>* patArithRandom::getLiterals(vector<patString>* listOfLiterals,
					       vector<patReal>* valuesOfLiterals,
					       patBoolean withRandom,
					       patError*& err) const {
    if (withRandom) {
      listOfLiterals->push_back(getOperatorName()) ;
      listOfLiterals->push_back(locationParameter) ;
      listOfLiterals->push_back(scaleParameter) ;
      if (valuesOfLiterals != NULL) {
	patReal val = getValue(err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
	valuesOfLiterals->push_back(val) ;
      }
    }
    return listOfLiterals ;

}

patString patArithRandom::getCompactName() const {
  stringstream str ;
  str << locationParameter << "_" << scaleParameter ;
  return patString(str.str()) ;
}
