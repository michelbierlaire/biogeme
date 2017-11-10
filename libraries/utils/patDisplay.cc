//-*-c++-*------------------------------------------------------------
//
// File name : patDisplay.cc
// Author :    Michel Bierlaire
// Date :      Mon Apr  4 18:19:44 2016
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <cassert>
#include <iostream>
#include "patDisplay.h"
#include "patOutputFiles.h"
#include "patFileNames.h"
#include "patAbsTime.h"
#include "patVersion.h"
#include "patLogMessage.h"

patDisplay::patDisplay() : screenImportance(patImportance::patDEBUG), 
			   logImportance(patImportance::patDEBUG),
			   logFileName(patFileNames::the()->getLogFile().c_str()),
			   logMessage(NULL) {

  patAbsTime now ;
  now.setTimeOfDay() ;
  patOutputFiles::the()->addUsefulFile(logFileName,"Log of all outputs") ;
  logFile.open(logFileName.c_str(),std::ofstream::out) ;
  logFile << "This file has automatically been generated." << endl ;
  logFile << now.getTimeString(patTsfFULL) << endl ;
  logFile << patVersion::the()->getCopyright() << endl ;
  logFile << endl ;
  logFile << patVersion::the()->getVersionInfoDate() << endl ;
  logFile << patVersion::the()->getVersionInfoAuthor() << endl ;
  logFile << endl ;
  logFile << flush ;

}

patDisplay::patDisplay(const patDisplay& td) : screenImportance(patImportance::patDEBUG), 
			   logImportance(patImportance::patDEBUG),
			   logFile(patFileNames::the()->getLogFile().c_str()),
			   logMessage(NULL) {
  //  singleInstance = td.singleInstance ;
}

patDisplay& patDisplay::operator=(const patDisplay& td) {
  // if (this != &td) {
  //   singleInstance = td.singleInstance ;
  // }
  return *this ;
}

patDisplay::~patDisplay() {
  messages.erase(messages.begin(),messages.end()) ;
  
  logFile.close() ;
}

patDisplay& patDisplay::the() {
  // call_once(patDisplay::only_one,
  // 	   [] () {
  // 	     patDisplay::singleInstance.reset( new patDisplay() );
  // 	   });
  
  // return *patDisplay::singleInstance;
  static patDisplay singleInstance ;
  return singleInstance ;
}

void patDisplay::addMessage(const patImportance& aType,
			    const patString& text,
			    const patString& fileName,
			    const patString& lineNumber) {

  patMessage theMessage ;

  theMessage.theImportance = aType ;
  theMessage.text = text ;
  theMessage.fileName = fileName ;
  theMessage.lineNumber = lineNumber ;

  patAbsTime now ;
  now.setTimeOfDay() ;
  theMessage.theTime = now ;
  
  if (aType <= screenImportance) {
    if (screenImportance < patImportance::patDEBUG) {
      if (logMessage != NULL) {
	logMessage->addLogMessage(theMessage.shortVersion()) ;
      }
      else {
	cout << theMessage.shortVersion() << endl << flush  ;
      }
    }
    else {
      if (logMessage != NULL) {
	logMessage->addLogMessage(theMessage.fullVersion()) ;
      }
      else {
	cout << theMessage.fullVersion() << endl << flush  ;
      }
    }
  }
  
  if (aType <= logImportance) {
    if (logImportance < patImportance::patDEBUG) {
      logFile << theMessage.shortVersion() << endl << flush  ;
    }
    else {
      logFile << theMessage.fullVersion() << endl << flush  ;
    }
  }
  messages.push_back(theMessage) ;
}

void patDisplay::setScreenImportanceLevel(const patImportance& aType) {
  screenImportance = aType ;
}
  
void patDisplay::setLogImportanceLevel(const patImportance& aType) {
  logImportance = aType ;
}

void patDisplay::initProgressReport(const patString message,
			unsigned long upperBound) {
  

}

patBoolean patDisplay::updateProgressReport(unsigned long currentValue) {
  return patTRUE ;
}

void patDisplay::terminateProgressReport() {

}

void patDisplay::setLogMessage(patLogMessage* up) {
  logMessage = up ;
}

// std::once_flag patDisplay::only_one ;
// std::shared_ptr<patDisplay> patDisplay::singleInstance = nullptr ;
