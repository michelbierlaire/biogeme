//-*-c++-*------------------------------------------------------------
//
// File name : patBiogemeRun.h
// Author :    Michel Bierlaire
// Date :      Mon Oct 15 15:37:23 2007
//
//--------------------------------------------------------------------

#ifndef patBiogemeRun_h
#define patBiogemeRun_h

  typedef  enum {
    //
    patNormalRun,
    // Biogeme does not compute anything, but generates a c++ code
    patGeneratingCode,
    // Biogeme uses the C++ generated code to run in parallel
    patParallelRun
  } patBiogemeRun ;

#endif
