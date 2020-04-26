//-*-c++-*------------------------------------------------------------
//
// File name : patUtility.cc
// Author :    Michel Bierlaire
// Date :      Wed Jan 10 14:06:04 2001
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patUtility.h"
#include "patModelSpec.h"


void patUtility::genericCppCode(ostream& cppFile, 
				patBoolean derivatives, 
				patError*& err) {
  unsigned long J = patModelSpec::the()->getNbrAlternatives() ; 
  if (derivatives) {
    cppFile << "    vector<vector<patReal> > DV(grad->size(),vector<patReal>("<< J <<",0.0)) ;" << endl ;
    cppFile << "    vector<patReal> probVec("<< J <<") ;" << endl ;
  }
  cppFile << "  patReal maxUtility = " << -patMaxReal <<" ;" << endl ;
  cppFile << "    vector<patReal> V("<<J<<") ;" << endl ;
  cppFile << "    vector<patReal> expV("<<J<<") ;" << endl ;
  cppFile << "  " << endl ;
  for (unsigned long j = 0 ; 
       j < J ;
       ++j) {
    cppFile << "    // Alternative " << j << endl ;
    cppFile << "    if (observation->availability[" << j << "]) {" << endl ;
      
    cppFile << "  V["<<j<<"] = " ;
    generateCppCode(cppFile,j,err) ;
   
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    cppFile << " ;" << endl ;
    if (derivatives) {
      for (unsigned long i = 0 ; i < patModelSpec::the()->getNbrNonFixedParameters() ; ++i) {
	cppFile << "      DV[" << i <<" ][" << j << "] = " ;
	generateCppDerivativeCode(cppFile,j,i,err) ;
	cppFile << ";" << endl ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
      }
    }

    cppFile << "    if (V["<<j<<"] > maxUtility) {" << endl ;
    cppFile << "	maxUtility = V["<<j<<"] ;" << endl ;
    cppFile << "      }//   if (V["<<j<<"] > maxUtility)" << endl ;
    cppFile << "    } //if (individual->availability[" << j << "])"<< endl ;
  }
  cppFile << "  for (unsigned long j = 0 ; j < "<< J <<" ; ++j) {" << endl ;
  cppFile << "    if (observation->availability[j]) {" << endl ;
  cppFile << "      V[j] -= maxUtility;" << endl ;
  cppFile << "  } //if (observation->availability[j]  " << endl ;
  cppFile << "  } //for ( j = 0 ; j < "<< J <<" ; ++j) " << endl ;
  return ;
}
