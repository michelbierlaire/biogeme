//-*-c++-*------------------------------------------------------------
//
// File name : patNetworkGevLinkParameter.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed May 16 16:28:51 2001
//
//--------------------------------------------------------------------

#ifndef patNetworkGevLinkParameter_h
#define patNetworkGevLinkParameter_h

#include "patBetaLikeParameter.h"

/**
 */
struct patNetworkGevLinkParameter {
    
  /**
   */
  patBetaLikeParameter alpha ;
  /**
    Upstream node
   */
  patString aNode ;
  /**
     Downstream node
   */
  patString bNode ;
};



#endif
