//-*-c++-*------------------------------------------------------------
//
// File name : pybiogeme.cc
// Author :    Michel Bierlaire
// Date :      Tue Apr 19 06:51:49 2011
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "bioMain.h"
#include "patFileNames.h"
//#include "bioMyObjects.h"
int main(int argc, char *argv[]) {
  patError* err(NULL) ;

  bioMain theMain ;
  
 //  myBioStatistics theStats ; 
//   theMain.setStatistics(&theStats) ;
//   myBioConstraints theContraints ; 
//   theMain.setConstraints(&theContraints) ;
  theMain.run(argc,argv,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return -1 ;
  }

}
