//-*-c++-*------------------------------------------------------------
//
// File name : patUniform.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Thu Mar  6 16:04:50 2003
//
//--------------------------------------------------------------------

#ifndef patUniform_h
#define patUniform_h

#include "patConst.h"
#include "patType.h"

class patError ;

/**
   @doc Generic interface for uniformly distributed random numbers
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Thu Mar  6 16:04:50 2003)
 */

class patUniform {

 public:

  virtual ~patUniform() ;
  /**
     Return random numbers uniformly distributed between 0 and 1
   */
  virtual patReal getUniform(patError*& err) = PURE_VIRTUAL ;

  virtual patString getType() const = PURE_VIRTUAL ;
};
#endif
