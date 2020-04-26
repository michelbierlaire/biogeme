//-*-c++-*------------------------------------------------------------
//
// File name : patNetworkAlphaIterator.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Dec 12 15:41:08 2001
//
//--------------------------------------------------------------------

#ifndef patNetworkAlphaIterator_h
#define patNetworkAlphaIterator_h


#include <map>
#include "patIterator.h"
#include "patNetworkGevLinkParameter.h"

/**
   @doc Iterator on a map<patString,patNetworkAlphaParameter>
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Dec 12 15:41:51 2001)
 */
class patNetworkAlphaIterator: public patIterator<patBetaLikeParameter> {
public:
  patNetworkAlphaIterator(map<patString,patNetworkGevLinkParameter>* x) ;
  void first()  ;
  void next()  ;
  patBoolean isDone()  ;
  patBetaLikeParameter currentItem()  ;
private:
  map<patString,patNetworkGevLinkParameter>* theMap ;
  map<patString,patNetworkGevLinkParameter>::const_iterator i ;

};
#endif 
