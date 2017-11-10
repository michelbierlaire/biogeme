//-*-c++-*------------------------------------------------------------
//
// File name : patNetworkGevModel.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Sun Dec 16 21:59:58 2001
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <assert.h>
#include "patNetworkGevModel.h"
#include "patModelSpec.h"
#include "patNetworkGevAlt.h"
#include "patNetworkGevNest.h"

patNetworkGevModel::
patNetworkGevModel(map<patString, patBetaLikeParameter>* nodes,
		   map<patString, patNetworkGevLinkParameter>* links) :
  theRoot(NULL),
  networkGevNodes(nodes),
  networkGevLinks(links) {
  assert(nodes != NULL) ;
  assert(links != NULL) ;
  
  // Create root node 
  theRoot = new patNetworkGevNest(patModelSpec::the()->rootNodeName) ;
  theRoot->setNbrParameters(getNbrParameters()) ;

  // Create nodes
  for (map<patString, patBetaLikeParameter>::iterator i =
	 networkGevNodes->begin() ;
       i != networkGevNodes->end() ;
       ++i) {

    DEBUG_MESSAGE("Create node " << i->first) ;
    unsigned long alt = patModelSpec::the()->getAltInternalId(i->first) ;
    DEBUG_MESSAGE("Alternative= " << alt) ;
    if (alt == patBadId) {
      //      DEBUG_MESSAGE("This is not an alternative") ;
      // Node is not an alternative
      patNetworkGevNode* ptr = new patNetworkGevNest(i->first,
						 i->second.id) ;
      listOfNodes.
	insert(pair<unsigned long,
	       patNetworkGevNode*>(i->second.id,
				   ptr)) ;
    }
    else {
      //      DEBUG_MESSAGE("This is an alternative") ;
      // Node is an alternative

      patNetworkGevNode* ptr = new patNetworkGevAlt(i->first,
						    i->second.id,
						    alt) ;  
      listOfNodes.insert(pair<unsigned long,
			 patNetworkGevNode*>(i->second.id,
					     ptr)) ;
    }
  }

  // Create links

  for (map<patString, patNetworkGevLinkParameter>::iterator i =
	 networkGevLinks->begin() ;
       i != networkGevLinks->end() ;
       ++i) {
    unsigned long aNode = patModelSpec::the()->getNetworkGevNodeId(i->second.aNode) ;
    unsigned long bNode = patModelSpec::the()->getNetworkGevNodeId(i->second.bNode) ;
    unsigned long alphaId = i->second.alpha.id ;
    map<unsigned long, patNetworkGevNode*>::iterator aFound =
      listOfNodes.find(aNode) ;
    map<unsigned long, patNetworkGevNode*>::iterator bFound =
      listOfNodes.find(bNode) ;
    if (bFound == listOfNodes.end()) {
      FATAL("Node " << i->second.bNode << " unknown") ;
    }
    if (aFound == listOfNodes.end()) {
      // Add to the root
      assert(theRoot != NULL) ;
      theRoot->addSuccessor(bFound->second,alphaId) ;
    } 
    else {
      aFound->second->addSuccessor(bFound->second,alphaId) ;
    }
  }

  DEBUG_MESSAGE("Network GEV model has been built") ;

//   theRoot->print(cout) ;
//   cout << endl ;
}

patNetworkGevModel::~patNetworkGevModel() {
  if (theRoot != NULL) {
    DELETE_PTR(theRoot) ;
  }
  for (map<unsigned long, patNetworkGevNode*>::iterator i = 
	 listOfNodes.begin() ;
       i != listOfNodes.end() ;
       ++i) {
    if (i->second != NULL) {
      DELETE_PTR(i->second) ;
    }
  }
}

patGEV* patNetworkGevModel::getModel() const {
  return theRoot ;
}

unsigned long patNetworkGevModel::getNbrParameters() {
  // There is one parameter per link, and one per node
  if (networkGevLinks == NULL || networkGevNodes == NULL) {
    return 0 ;
  }
  return networkGevNodes->size() + networkGevLinks->size()  ;
}

patNetworkGevNode* patNetworkGevModel::getRoot() {
  return theRoot ;
}
