//-*-c++-*------------------------------------------------------------
//
// File name : patGetWorkingDir.cc
// Author :    Michel Bierlaire
// Date :      Sun Apr  2 15:22:53 2017
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include "patGetWorkingDir.h"

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_DIRECT_H
#include <direct.h>
#endif

patGetWorkingDir::patGetWorkingDir() {

}

patString patGetWorkingDir::operator()() {
#ifdef HAVE_GETCWD
  getcwd(buffer,FILENAME_MAX) ;
#elif HAVE__GETCWD
  _getcwd(buffer,FILENAME_MAX) ;
#else
  WARNING("The system where the software was compiled does not support getcwd") ;
  return patString() ;
	  
#endif
  return patString(buffer) ;
  
}

