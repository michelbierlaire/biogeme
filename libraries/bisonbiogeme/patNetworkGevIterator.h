//-*-c++-*------------------------------------------------------------
//
// File name : patNetworkGevIterator.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Sep 11 14:52:31 2002
//
//--------------------------------------------------------------------

#ifndef patNetworkGevIterator_h
#define patNetworkGevIterator_h

#include <vector>
#include "patIterator.h"

class patNetworkGevNode ;

/**
   @doc Iterator on the successors of a specific node from a Network GEV model
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Sep 11 14:52:31 2002)
*/

class patNetworkGevIterator: public patIterator<patNetworkGevNode*> {
public :
  /**
   */
  patNetworkGevIterator(vector<patNetworkGevNode*>* aList) ;
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
  patNetworkGevNode* currentItem()  ;

private:
  vector<patNetworkGevNode*>* theList ;
  vector<patNetworkGevNode*>::const_iterator theIterator ;
};

#endif
