//-*-c++-*------------------------------------------------------------
//
// File name : patConstraintNestIterator.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Sep 27 16:51:12 2001
//
//--------------------------------------------------------------------

#ifndef patConstraintNestIterator_h
#define patConstraintNestIterator_h

#include <vector>
#include "patIterator.h"

/**
   @doc Iterator on a list of pairs of parameters index, corresponding to
   pairs of nest coefficients to be constrained.
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Thu Sep 27 16:51:12 2001)
  */
class patConstraintNestIterator : 
  public patIterator<pair<unsigned long, unsigned long> > {
public:
  /**
   */
  patConstraintNestIterator(const vector<pair<unsigned long, unsigned long> >& v) ;
  /**
   */
  void first()  ;
  /**
   */
  void next()  ;
  /**
   */
  patBoolean isDone()  ;
  /**
   */
  pair<unsigned long, unsigned long> currentItem()  ;
private:
  vector<pair<unsigned long, unsigned long> > theVector ;
  vector<pair<unsigned long, unsigned long> >::const_iterator i ;
};
#endif
