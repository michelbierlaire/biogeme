//-*-c++-*------------------------------------------------------------
//
// File name : patMessage.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Mon Jun  2 23:12:21 2003
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <iostream>
#include <sstream>
#include "patMessage.h"

patMessage::patMessage() : theImportance(patImportance::patDEBUG) {


}

patMessage::patMessage(const patMessage& msg) : theImportance(msg.theImportance){
    theTime = msg.theTime ;
    text = msg.text ;
    lineNumber = msg.lineNumber ;
    fileName = msg.fileName ;

}

patString patMessage::shortVersion() {
  stringstream str ;
  switch (theImportance()) {
  case patImportance::patFATAL:
    str << "FATAL ERROR: " ;
    break;
  case patImportance::patERROR:
    str << "ERROR: " ;
    break ;
  case patImportance::patWARNING:
    str << "Warning: " ;
    break ;
  case patImportance::patDETAILED:
     break ;
  case patImportance::patDEBUG:
     break ;
  case patImportance::patGENERAL:
    break ;
  }
  str << " " << text  ;
  return patString(str.str()) ;
}


patString patMessage::fullVersion() {
  stringstream str ;
  switch (theImportance()) {
  case patImportance::patFATAL:
    str << "FATAL ERROR: " ;
    break;
  case patImportance::patERROR:
    str << "ERROR: " ;
    break ;
  case patImportance::patWARNING:
    str << "Warning: " ;
    break ;
  case patImportance::patDETAILED:
     break ;
  case patImportance::patDEBUG:
     break ;
  case patImportance::patGENERAL:
    break ;
  }
  str << "[" << theTime << "]" << fileName << ":" << lineNumber << " " <<  text  ;
  return patString(str.str()) ;
}
