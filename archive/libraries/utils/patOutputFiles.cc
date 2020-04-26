//-*-c++-*------------------------------------------------------------
//
// File name : patOutputFiles.cc
// Author :    Michel Bierlaire
// Date :      Sat Mar 11 10:13:27 2017
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "patOutputFiles.h"
#include "patSingletonFactory.h"
#include "patStlVectorIterator.h"

patOutputFiles::patOutputFiles() {

}

patOutputFiles::~patOutputFiles() {

}

patOutputFiles* patOutputFiles::the() {
  return patSingletonFactory::the()->patOutputFiles_the();
}

void patOutputFiles::addCriticalFile(patString fileName, patString description) {
  criticalFiles.push_back(pair<patString,patString>(fileName,description)) ;
}

void patOutputFiles::addUsefulFile(patString fileName, patString description) {
  usefulFiles.push_back(pair<patString,patString>(fileName,description)) ;
}
void patOutputFiles::addDebugFile(patString fileName, patString description) {
  debugFiles.push_back(pair<patString,patString>(fileName,description)) ;
}

patIterator<pair<patString,patString> >* patOutputFiles::createCriticalIterator() {
  patStlVectorIterator<pair<patString,patString> > *ptr =
    new patStlVectorIterator<pair<patString,patString> >(criticalFiles) ;
  return ptr ;
}

patIterator<pair<patString,patString> >* patOutputFiles::createUsefulIterator() {
  patStlVectorIterator<pair<patString,patString> > *ptr =
    new patStlVectorIterator<pair<patString,patString> >(usefulFiles) ;
  return ptr ;
}
patIterator<pair<patString,patString> >* patOutputFiles::createDebugIterator() {
  patStlVectorIterator<pair<patString,patString> > *ptr =
    new patStlVectorIterator<pair<patString,patString> >(debugFiles) ;
  return ptr ;
}

void patOutputFiles::display() {
  if (!debugFiles.empty()) {
    GENERAL_MESSAGE("--------------------------------------------------------");
    GENERAL_MESSAGE("Debug files") ;
    GENERAL_MESSAGE("--------------------------------------------------------");
    patIterator<pair<patString,patString> > *theDebugIter = createDebugIterator() ;  
    for (theDebugIter->first() ;
	 !theDebugIter->isDone() ;
	 theDebugIter->next()) {
      pair<patString,patString> f = theDebugIter->currentItem() ;
      GENERAL_MESSAGE("     " << f.first << ": " << f.second) ;
    }
    DELETE_PTR(theDebugIter) ;
  }
  if (!usefulFiles.empty()) {
    GENERAL_MESSAGE("--------------------------------------------------------");
    GENERAL_MESSAGE("Useful files") ;
    GENERAL_MESSAGE("--------------------------------------------------------");
    patIterator<pair<patString,patString> > *theUsefulIter =
      createUsefulIterator() ;  
    for (theUsefulIter->first() ;
	 !theUsefulIter->isDone() ;
	 theUsefulIter->next()) {
      pair<patString,patString> f = theUsefulIter->currentItem() ;
      GENERAL_MESSAGE("     " << f.first << ": " << f.second) ;
    }
    DELETE_PTR(theUsefulIter) ;
  }
  if (!criticalFiles.empty()) {
    GENERAL_MESSAGE("--------------------------------------------------------");
    GENERAL_MESSAGE("Important files") ;
    GENERAL_MESSAGE("--------------------------------------------------------");
    patIterator<pair<patString,patString> > *theCriticalIter =
      createCriticalIterator() ;  
    for (theCriticalIter->first() ;
	 !theCriticalIter->isDone() ;
	 theCriticalIter->next()) {
      pair<patString,patString> f = theCriticalIter->currentItem() ;
      GENERAL_MESSAGE("     " << f.first << ": " << f.second) ;
    }
    DELETE_PTR(theCriticalIter) ;
  }

}

void patOutputFiles::clearList() {
  if (!criticalFiles.empty()) {
    criticalFiles.erase(criticalFiles.begin(),criticalFiles.end()) ;
  }
  if (!usefulFiles.empty()) {
    usefulFiles.erase(usefulFiles.begin(),usefulFiles.end()) ;
  }
  if (!debugFiles.empty()) {
    debugFiles.erase(debugFiles.begin(),debugFiles.end()) ;
  }
}
