//-*-c++-*------------------------------------------------------------
//
// File name : patExecScript.cc
// Author :    Michel Bierlaire
// Date :      Sat Apr  1 17:21:25 2017
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "patExecScript.h"

#include <string>
#include <iostream>
#include "patDisplay.h"

patExecScript::patExecScript(vector<patString> c) :
  command(c) {
  if (!command.empty()) {
    DEBUG_MESSAGE("Prepared to execute " << command[0]) ;
  }
}

void patExecScript::run() {
  patString c ;
  for (vector<patString>::iterator i = command.begin() ;
       i != command.end() ;
       ++i) {
    c += *i + " " ;
  }
#if defined(HAVE_SYS_WAIT_H)
  redi::ipstream proc(c, redi::pstreams::pstdout | redi::pstreams::pstderr);
  std::string line;
  // read child's stdout
  while (std::getline(proc.out(), line))
    output << line << endl;
  // read child's stderr
  while (std::getline(proc.err(), line))
    output << "stderr: " << line << endl;
#elif defined(HAVE_SYSTEM)
  int status = system(c.c_str()) ;
  output << "Command executed with status " << status ;
  return ;
#elif defined(HAVE_EXECLP)
  // I did not find a better way do that
  switch (command.size()) {
  case 0:
    output << "No command executed" ;
    return ;
  case 1:
    execlp(command[0].c_str(),command[0].c_str(),NULL) ;
    output<< "Finished to run '" << c << "'" ;
    return ;
  case 2:
    execlp(command[0].c_str(),command[0].c_str(),command[1].c_str(),NULL) ;
    output<< "Finished to run '" << c << "'" ;
    return ;
  case 3:
    execlp(command[0].c_str(),command[0].c_str(),command[1].c_str(),command[2].c_str(),NULL) ;
    output<< "Finished to run '" << c << "'" ;
    return ;
  case 4:
    execlp(command[0].c_str(),command[0].c_str(),command[1].c_str(),command[2].c_str(),command[3].c_str(),NULL) ;
    output<< "Finished to run '" << c << "'" ;
    return ;
  case 5:
    execlp(command[0].c_str(),command[0].c_str(),command[1].c_str(),command[2].c_str(),command[3].c_str(),command[4].c_str(),NULL) ;
    output<< "Finished to run '" << c << "'" ;
    return ;
  default:
    output << "Cannot execute commands with more than 4 arguments" ;
    return ;
  }
#else
  output << "This system does not allow to run a command from C++" << endl ;
  output << "You m ust use a terminal" << endl ;
#endif
  
}

patString patExecScript::getOutput() {
  return output.str() ;
}

patBoolean patExecScript::killAfterRun() const {
#if defined(HAVE_SYS_WAIT_H)
  return patFALSE ;
#elif defined(HAVE_SYSTEM)
  return patFALSE ;  
#elif defined(HAVE_EXECLP)
  return patTRUE ;
#else
  return patFALSE ;
#endif

}
