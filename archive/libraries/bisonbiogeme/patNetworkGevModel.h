//-*-c++-*------------------------------------------------------------
//
// File name : patNetworkGevModel.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Sun Dec 16 21:52:48 2001
//
//--------------------------------------------------------------------

#ifndef patNetworkGevModel_h
#define patNetworkGevModel_h

#include <map>
#include "patBetaLikeParameter.h"
#include "patNetworkGevLinkParameter.h"
#include "patNetworkGevNode.h"

class patGEV ;
/**
   @doc This class manages the structure of the network GEV model. Note that
   it does not implement the interface of a GEV model. The associated GEV
   model is implemented by the root node which, like all other nodes,
   implements that interface.
   @author \URL[Michel  Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Sun Dec 16 21:52:48 2001)
 */

class patNetworkGevModel {
public:
  patNetworkGevModel(map<patString, patBetaLikeParameter>* nodes,
		     map<patString, patNetworkGevLinkParameter>* links) ;
  ~patNetworkGevModel() ;
  patGEV* getModel() const ; 
  unsigned long getNbrParameters() ;
  patNetworkGevNode* getRoot() ;

private:
  patNetworkGevNode* theRoot ;
  map<patString, patBetaLikeParameter>* networkGevNodes ;
  map<patString, patNetworkGevLinkParameter>* networkGevLinks ;
  map<unsigned long, patNetworkGevNode*> listOfNodes ;
};

#endif
