#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
//#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include "patType.h"
#include "patConst.h"
#include "patString.h" 
#include "patDisplay.h"

int main(int argc, char *argv[]) {

  if (((argc == 2) && patString(argv[1]) == patString("-h")) || argc <= 1) {
    cout << "Usage: " << argv[0] << " file1 file2 ... " << endl ;
    cout << "Each file must be of ASCII format and have the exact same number of rows. Row j of the output file will be the merging of row j of each files specificed." << endl ;
    exit(0 );
  }

  

  patString outputFile("biomergeOutput.lis") ;

  vector<patString> filesToMerge ;
  

  //  patULong bufferSize(100000) ;
  //  char buffer[bufferSize] ;
  patString buffer ;

  for (unsigned short i=1 ; i < argc ; ++i) {
    filesToMerge.push_back(patString(argv[i])) ;
  }
  unsigned short F = filesToMerge.size() ;
  vector<ifstream*> files(F) ;
  for (unsigned short i = 0 ; i < F ; ++i) {
    files[i] = new ifstream(filesToMerge[i].c_str()) ;
  }

  ofstream theOutput(outputFile.c_str()) ;

  patString dosCharacter("\r") ;
  unsigned long row(0) ;
    patBoolean allEof(patFALSE) ;
  while(!allEof) {
    ++row ;
    stringstream theRow ;
    for (unsigned short i = 0 ; i < F ; ++i) {
      if (!(*files[i]) || files[i]->eof()) {
	if (i == 0) {
	  allEof = patTRUE ;
	}
      }
      else {
	allEof = patFALSE ;
      }
    }
    if (!allEof) {
      for (unsigned short i = 0 ; i < F ; ++i) {
	if (!(*files[i]) || files[i]->eof()) {
	  theOutput << "---" << '\t' ;
	}
	else {
	  //	  files[i]->getline(buffer,bufferSize) ;
	  getline(*files[i],buffer) ;
	  replaceAll(&buffer,dosCharacter,"") ;
	  theOutput << buffer << '\t' ;
	}
      }
      theOutput << endl ;
    }
  }

  GENERAL_MESSAGE("File " << outputFile << " has been generated") ;
}
