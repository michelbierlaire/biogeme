//-*-c++-*------------------------------------------------------------
//
// File name : solvOptParameters.h
// Date :      Fri Oct 22 14:52:31 2010
//
//--------------------------------------------------------------------

#ifndef solvoptParameters_h
#define solvoptParameters_h

class solvoptParameters {
 public:
  patReal errorArgument ; // patParameters::the()->getsolvoptErrorArgument() ;
  patReal errorFunction ; // patParameters::the()->getsolvoptErrorFunction() ;
  patReal maxIter ; //  patParameters::the()->getsolvoptMaxIter() ;
  patReal display ; //  patParameters::the()->getsolvoptDisplay() ;

  patString stopFileName ;

};

#endif 
