//-*-c++-*------------------------------------------------------------
//
// File name : patFileNames.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Aug 15 21:50:46 2002
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include <cassert>
#include "patDisplay.h"
#include "patConst.h"
#include "patFileNames.h"
#include "patFileExists.h"
#include "patErrMiscError.h"
#include "patSingletonFactory.h"

patFileNames::~patFileNames() {

}
  
patFileNames::patFileNames() : modelName("default"), 
			       defaultName("default") {
  
}

patFileNames* patFileNames::the() {
  return patSingletonFactory::the()->patFileNames_the();
}

  
patString patFileNames::getModelName() {
  return modelName ;
}

void patFileNames::setModelName(const patString& name) {
  modelName = name ;
}

patString patFileNames::getInputFile(patString extension) {
  stringstream modstr ;
  modstr << modelName << extension ;
  patString inputFile = modstr.str() ;
  if (!patFileExists()(inputFile) && (modelName != defaultName)) {
    GENERAL_MESSAGE(inputFile << " does not exist") ;
    stringstream defstr ;
    defstr << defaultName << extension ;
    inputFile = defstr.str() ;
    GENERAL_MESSAGE("Trying " << inputFile << " instead") ;
  }
  return inputFile ;
}

patString patFileNames::getOutputFile(patString extension, patError*& err) {
  stringstream str ;
  short count = 0 ;
  str << modelName << extension ;
  patString outputFile = str.str() ;
  while (patFileExists()(outputFile)) {
    if (count == 9999) {
      stringstream errstr ;
      errstr << outputFile << " exists. Too many files for model " 
	     << modelName 
	     << ". Please purge your files." ;
      err = new patErrMiscError(errstr.str()) ;
      WARNING(err->describe()) ;
      return patString() ;
    }
    else {
      ++count ;
    }
    stringstream defstr ;
    defstr << modelName << '~' << count << extension ;
    outputFile = defstr.str() ;
  }
  return outputFile ;
}

patString patFileNames::getOutputFileNoBackup(patString extension) {
  stringstream str ;
  str << modelName << extension ;
  patString outputFile = str.str() ;
  return outputFile ;
}


patString patFileNames::getParFile() {
  return(getInputFile(".par")) ;
}
  
patString patFileNames::getModFile() {
  return(getInputFile(".mod")) ;
}

void patFileNames::addSamFile(patString name) {
  samFile.push_back(name) ;
}

void patFileNames::emptySamFiles() {
  samFile.erase(samFile.begin(),samFile.end()) ;
}

  
patString patFileNames::getSamFile(unsigned short nbr, patError*& err) {
  if (samFile.empty()) {
    samFile.push_back(patString("sample.dat")) ;
  }
  if (nbr >= samFile.size()) {
    stringstream str ;
    str << "Sample file #" << nbr << " does not exist. Only " <<  samFile.size() << " files, numbered from 0 to " << samFile.size()-1 ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patString();
  }
  return samFile[nbr] ;
}

patString patFileNames::getStanFile(unsigned short nbr, patError*& err) {
  if (samFile.empty()) {
    samFile.push_back(patString("sample.dstan")) ;
  }
  if (nbr >= samFile.size()) {
    stringstream str ;
    str << "Sample file #" << nbr << " does not exist. Only " <<  samFile.size() << " files, numbered from 0 to " << samFile.size()-1 ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patString();
  }
  
  patString dat = samFile[nbr] ;
  size_t position = dat.find(".") ;
  patString stan = (patString::npos == position) ?dat : dat.substr(0,position) ;
  return stan ;
}


  
patString patFileNames::getStaFile(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  patString f = getOutputFile(".sta",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  return(f) ;
}

patString patFileNames::getRepFile(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  patString f = getOutputFile(".rep",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  return(f) ;
}
  
patString patFileNames::getALogitFile(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  patString f = getOutputFile(".F12",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  return f ;
}
  
patString patFileNames::getHtmlFile(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  patString f = getOutputFile(".html",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  return f ;
}
  
patString patFileNames::getPythonSpecFile(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  patString f = getOutputFile(".py",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  return f ;
}
  
patString patFileNames::getLatexFile(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  patString f = getOutputFile(".tex",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  return f ;
}
  
patString patFileNames::getLogFile() {
  return("__biogeme.log") ;
}
  
patString patFileNames::getEnuFile(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  patString f = getOutputFile(".enu",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  return f ;
}
 
patString patFileNames::getZhengFosgerauLatex(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  stringstream str ;
  short count = 0 ;
  str << modelName << "_zheng.tex" ;
  patString zfLatex = str.str() ;
  while (patFileExists()(zfLatex)) {
    if (count == 99) {
      stringstream errstr ;
      errstr << zfLatex << " exists. Too many files for model " 
	     << modelName 
	     << ". Please purge your files." ;
      err = new patErrMiscError(errstr.str()) ;
      WARNING(err->describe()) ;
      return patString() ;
    }
    else {
      ++count ;
    }
    stringstream defstr ;
    defstr << modelName << "_zheng" << '~' << count << ".tex" ;
    zfLatex = defstr.str() ;
  }
  
  return zfLatex ;

}

patString patFileNames::getZhengFosgerau(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  stringstream str ;
  short count = 0 ;
  str << modelName << "_zheng.enu" ;
  patString zfDetails = str.str() ;
  while (patFileExists()(zfDetails)) {
    if (count == 99) {
      stringstream errstr ;
      errstr << zfDetails << " exists. Too many files for model " 
	     << modelName 
	     << ". Please purge your files." ;
      err = new patErrMiscError(errstr.str()) ;
      WARNING(err->describe()) ;
      return patString() ;
    }
    else {
      ++count ;
    }
    stringstream defstr ;
    defstr << modelName << "_zheng" << '~' << count << ".enu" ;
    zfDetails = defstr.str() ;
  }
  
  return zfDetails ;

}

patString patFileNames::getResFile(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  patString f = getOutputFile(".res",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  return f ;
}
  
patString patFileNames::getBckFile(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  patString f = getOutputFile(".bck",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  return f ;
}
  


patString patFileNames::getModelDebug() {
  return patString("model.debug") ;
}

patString patFileNames::getParamDebug() {
  return patString("parameters.out") ;
}

patString patFileNames::getSpecDebug() {
  return patString("__specFile.debug");
}

patString patFileNames::getGnuplotFile(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  patString f = getOutputFile(".gp",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  return f ;
}

void patFileNames::setNormalDrawsFile(patString fileName) {
  normalDrawFile = fileName ;
}

void patFileNames::setUnifDrawsFile(patString fileName) {
  unifDrawFile = fileName ;
}

patString patFileNames::getNormalDrawsFile(patError*& err) {
  return normalDrawFile ;
}

patString patFileNames::getUnifDrawsFile(patError*& err) {
  return unifDrawFile ;
}

patString patFileNames::getNormalDrawLogFile(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  patString f = getOutputFile(".ndraws",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  return f ;
}

patString patFileNames::getUnifDrawLogFile(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }

  patString f = getOutputFile(".udraws",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  return f ;
}


unsigned short patFileNames::getNbrSampleFiles() {
  if (samFile.empty()) {
    samFile.push_back(patString("sample.dat")) ;
  }
  return samFile.size() ;
}

patString patFileNames::getCcCode(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  patString f = getOutputFileNoBackup(".cc") ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  return f ;
}

patString patFileNames::getCcCode_f(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  stringstream str ;
  str << modelName << "_f.cc" ;
  return patString(str.str()) ;
}

patString patFileNames::getCcCode_s(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  stringstream str ;
  str << modelName << "_s.cc" ;
  return patString(str.str()) ;
}


patString patFileNames::getCcCode_g(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  stringstream str ;
  str << modelName << "_g.cc" ;
  return(patString(str.str())) ;
}

patString patFileNames::getPythonEstimated(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  patString f = getOutputFile("_param.py",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  return f ;
}

patString patFileNames::getMonteCarloLog(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  patString f = getOutputFile(".mclg",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  return f ;
}

