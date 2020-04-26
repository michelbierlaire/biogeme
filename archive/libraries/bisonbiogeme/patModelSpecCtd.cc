//-*-c++-*------------------------------------------------------------
//
// File name : patModelSpec.cc
// Author :    Michel Bierlaire
// Date :      Tue Nov  7 16:27:33 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <algorithm>
#include <iomanip>
#include <set>
#include <list>
#include "patMath.h"
#include "patNL.h"
#include "patFileNames.h"
#include "patOutputFiles.h"
#include "patValueVariables.h"
#include "patLinearUtility.h"
#include "patGeneralizedUtility.h"
#include "patAdditiveUtility.h"
#include "patAttributeNamesIterator.h"
#include "patUsedAttributeNamesIterator.h"
#include "patDiscreteParameterIterator.h"
#include "patArithAttribute.h"
#include "patFileExists.h"
#include "patLoop.h"
#include "patSecondDerivatives.h"
#include "patPValue.h"
#include "patSampleEnuGetIndices.h"

// #ifndef patNO_MTL
// #include "patMtl.h"
// #endif

#include "patType.h"
#include "patParameters.h"
#include "patModelSpec.h"
#include "patSpecParser.hh"
#include "patErrMiscError.h"
#include "patErrOutOfRange.h"
#include "patErrNullPointer.h"
#include "patVersion.h"
#include "patBetaLikeIterator.h"
#include "patCorrelation.h"
#include "patCompareCorrelation.h"
#include "patNlNestIterator.h"
#include "patCnlAlphaIterator.h"
#include "patFullCnlAlphaIterator.h"
#include "patStlVectorIterator.h"
#include "patConstraintNestIterator.h"
#include "patSequenceIterator.h"
#include "patStlVectorIterator.h"
#include "patHybridMatrix.h"
#include "patNetworkAlphaIterator.h"
#include "patNetworkGevModel.h"
#include "patUtilFunction.h"

unsigned long patModelSpec::getIdFromScaleName(const patString& scaleName) const {
  
  istringstream istr(scaleName.c_str()) ;
  unsigned long outId ;

  char  tmp ;
  // "Scale" contains 5 letters.
  for (unsigned long i = 0 ; i < 5 ; ++i) {
    istr.get(tmp) ;
  }
  istr >> outId ;
  return outId ;
}

patString patModelSpec::getEstimatedParameterName(unsigned long index) const {

  if (index >= nonFixedParameters.size()) {
    stringstream str ;
    str << "Unknown" << index ;
    return(patString(str.str())) ;
  }

  patBetaLikeParameter* ptr = nonFixedParameters[index] ;
  if (ptr == NULL) {
    stringstream str ;
    str << "Unknown" << index ;
    return(patString(str.str())) ;
  }
  return ptr->name ;
}


patVariables* patModelSpec::gatherGradient(patVariables* grad,
					   patBoolean computeBHHH,
					   vector<patVariables>* bhhh,
					   trHessian* trueHessian,
					   patVariables* betaDerivatives,
					   patReal* scaleDerivative,
					   unsigned long scaleIndex,
					   patVariables* paramDerivative,
					   patReal* muDerivative,
					   patSecondDerivatives* secondDeriv,
					   patError*& err) {
  
  if (err != NULL) {
    WARNING(err->describe());
    return NULL ;
  }

  if (grad == NULL) {
    err = new patErrNullPointer("trVector") ;
    WARNING(err->describe()) ;
    return NULL ;
  }

  if (grad->size() !=  getNbrNonFixedParameters()) {
    stringstream str ;
    str << "Gradient's dimension (" << grad->size() 
	<< ") is incompatible with problem's dimension (" 
	<<  getNbrNonFixedParameters() << ")" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL ;
  }

  if (firstGatherGradient) {
    thisGradient.resize(grad->size(),0.0) ;
    keepBetaIndices.resize(grad->size()) ;
    firstGatherGradient = patFALSE ;
  }
  else {
    fill(thisGradient.begin(),thisGradient.end(),0.0) ; 
  }

  // beta values

  for (allBetaIter->first() ;
       !allBetaIter->isDone() ;
       allBetaIter->next()) {
    patBetaLikeParameter bb = allBetaIter->currentItem() ;
    
    if (!bb.isFixed && !bb.hasDiscreteDistribution) {
      if (bb.index >= grad->size()) {
	err = new patErrOutOfRange<unsigned long>(bb.index,
						  0,
						  grad->size()-1) ;
	WARNING(err->describe()) ;
	return NULL ;
      }
      if (bb.id >= betaDerivatives->size()) {
	err = new patErrOutOfRange<unsigned long>(bb.id,
						  0,
						  betaDerivatives->size()-1) ;
	WARNING(err->describe()) ;
	return NULL ;
	
      }
      (*grad)[bb.index] += thisGradient[bb.index] = -(*betaDerivatives)[bb.id] ;
      keepBetaIndices[bb.index] = bb.id ;
    }
  }


  // mu value

  patBetaLikeParameter mu = getMu(err) ;
  if (err != NULL) {
    WARNING(err->describe());
    return NULL ;
  }

  if (!mu.isFixed) {
    if (mu.index >= grad->size()) {
      err = new patErrOutOfRange<unsigned long>(mu.index,
						 0,
						 grad->size()-1) ;
      WARNING(err->describe()) ;
      return NULL ;
    }

    (*grad)[mu.index] += thisGradient[mu.index] = -(*muDerivative) ;
  }
  
  // scale parameters

//   patIterator<patBetaLikeParameter>* scaleIter = 
//     createScaleIterator() ;
//   if (scaleIter == NULL) {
//     err = new patErrNullPointer("patIterator<patBetaLikeParameter>") ;
//     WARNING(err->describe()) ;
//     return NULL ;
//   }
  
//   for (scaleIter->first() ;
//        !scaleIter->isDone() ;
//        scaleIter->next()) {
//     patBetaLikeParameter bb = scaleIter->currentItem() ;
    
//     if (!bb.isFixed) {
//       if (bb.index >= grad->size()) {
// 	err = new patErrOutOfRange<unsigned long>(bb.index,
// 						   0,
// 						   grad->size()-1) ;
// 	WARNING(err->describe()) ;
// 	return NULL ;
//       }
//       if (bb.id >= scaleDerivative->size()) {
// 	err = new patErrOutOfRange<unsigned long>(bb.id,
// 						   0,
// 						   scaleDerivative->size()-1) ;
// 	WARNING(err->describe()) ;
// 	return NULL ;
	
//       }
//       (*grad)[bb.index] += -(*scaleDerivative)[bb.id] ;
//     }
//   }


  if (scaleIndex != patBadId) {
    if (scaleIndex >= grad->size()) {
      err = new patErrOutOfRange<unsigned long>(scaleIndex,0,grad->size()-1) ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    
    (*grad)[scaleIndex] += thisGradient[scaleIndex] = - (*scaleDerivative) ;
  }

  // Model param

  patIterator<patBetaLikeParameter>* modelIter = 
    createAllModelIterator() ;
  if (modelIter == NULL) {
    err = new patErrNullPointer("patIterator<patBetaLikeParameter>") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  
  for (modelIter->first() ;
       !modelIter->isDone() ;
       modelIter->next()) {
    patBetaLikeParameter bb = modelIter->currentItem() ;
    
    if (!bb.isFixed) {
      if (bb.index >= grad->size()) {
	err = new patErrOutOfRange<unsigned long>(bb.index,
						   0,
						   grad->size()-1) ;
	WARNING(err->describe()) ;
	return NULL ;
      }
      if (bb.id >= paramDerivative->size()) {
	err = new patErrOutOfRange<unsigned long>(bb.id,
						   0,
						   paramDerivative->size()-1) ;
	WARNING(err->describe()) ;
	return NULL ;
	
      }
      (*grad)[bb.index] += thisGradient[bb.index] = -(*paramDerivative)[bb.id] ;
    }
  }

  if (computeBHHH) {
    for (unsigned long i = 0 ; i < thisGradient.size() ; ++i) {
      for (unsigned long j = i ; j < thisGradient.size() ; ++j) {
	// We compute only the upper triangular part

	(*bhhh)[i][j] += thisGradient[i] * thisGradient[j] ;
      }
    }
  }

  //  DEBUG_MESSAGE("GATHER GRADIENT") ;
  if (secondDeriv != NULL && trueHessian != NULL) {
    for (unsigned long i = 0 ; i < thisGradient.size() ; ++i) {
      for (unsigned long j = i ; j < thisGradient.size() ; ++j) {
	trueHessian->setElement(i,j,secondDeriv->secondDerivBetaBeta[keepBetaIndices[i]][keepBetaIndices[j]],err) ; 
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
      }
    }
    //    trueHessian->print(cout) ;
  }

  return grad ;
}

patVariables patModelSpec::getLowerBounds(patError*& err) const {
  patVariables result ;
  for (vector<patBetaLikeParameter*>::const_iterator iter = 
	 nonFixedParameters.begin() ;
       iter != nonFixedParameters.end() ;
       ++iter) {
    if (*iter == NULL) {
      err = new patErrNullPointer("patBetaLikeParameter") ;
      WARNING(err->describe()) ;
      return patVariables();
    }
    result.push_back((*iter)->lowerBound) ;
  }
  return result ;
}

patVariables patModelSpec::getUpperBounds(patError*& err) const {
  patVariables result ;
  for (vector<patBetaLikeParameter*>::const_iterator iter = 
	 nonFixedParameters.begin() ;
       iter != nonFixedParameters.end() ;
       ++iter) {
    if (*iter == NULL) {
      err = new patErrNullPointer("patBetaLikeParameter") ;
      WARNING(err->describe()) ;
      return patVariables();
    }
    result.push_back((*iter)->upperBound) ;
  }
  return result ;
}

unsigned long patModelSpec::getAltInternalId(const patString& name) {
  map<patString, patAlternative>::const_iterator found =
    utilities.find(name) ;
  if (found == utilities.end()) {
    return patBadId ;
  }
  return found->second.id ;
}

unsigned long patModelSpec::getAltUserId(const patString& name) {
  map<patString, patAlternative>::const_iterator found =
    utilities.find(name) ;
  if (found == utilities.end()) {
    return patBadId ;
  }
  return found->second.userId ;
}

void patModelSpec::copyParametersValue(patError*& err) {
  muValue = mu.estimated ;
  for (map<patString,patBetaLikeParameter>::const_iterator i = 
	 betaParam.begin() ;
       i != betaParam.end() ;
       ++i) {
    betaParameters[i->second.id] = i->second.estimated ;
    if (i->second.id >= betaParameters.size()) {
      err = new patErrOutOfRange<unsigned long>(0,
						 i->second.id,
						 betaParameters.size()-1) ;
      WARNING(err->describe()) ;
      return ;
    }
  }
  if (isNL()) {
    for (map<patString,patNlNestDefinition>::const_iterator i =
	   nlNestParam.begin() ;
	 i != nlNestParam.end() ;
	 ++i) {
      modelParameters[i->second.nestCoef.id] = i->second.nestCoef.estimated ;
    if (i->second.nestCoef.id >= modelParameters.size()) {
      err = new patErrOutOfRange<unsigned long>(0,
						 i->second.nestCoef.id ,
						 modelParameters.size()-1) ;
      WARNING(err->describe()) ;
      return ;
    }
    }
  }
  if (isCNL()) {
    for (map<patString,patBetaLikeParameter>::const_iterator i =
	   cnlNestParam.begin() ;
	 i != cnlNestParam.end() ;
	 ++i) {
      if (i->second.id >= modelParameters.size()) {
	err = new patErrOutOfRange<unsigned long>(0,
						   i->second.id ,
						   modelParameters.size()-1) ;
	WARNING(err->describe()) ;
	return ;
      }
      modelParameters[i->second.id] = i->second.estimated ;
    }
    for (map<patString,patCnlAlphaParameter>::const_iterator i =
	   cnlAlphaParam.begin() ;
	 i != cnlAlphaParam.end() ;
	 ++i) {
      if (i->second.alpha.id >= modelParameters.size()) {
	err = new patErrOutOfRange<unsigned long>(0,
						   i->second.alpha.id ,
						   modelParameters.size()-1) ;
	WARNING(err->describe()) ;
	return ;
      }
      modelParameters[i->second.alpha.id] = i->second.alpha.estimated ;
    }
  }
  if (isNetworkGEV()) {
    for (map<patString,patBetaLikeParameter>::const_iterator i =
	   networkGevNodes.begin() ;
	 i != networkGevNodes.end() ;
	 ++i) {
      if (i->second.id >= modelParameters.size()) {
	err = new patErrOutOfRange<unsigned long>(0,
						   i->second.id ,
						   modelParameters.size()-1) ;
	WARNING(err->describe()) ;
	return ;
      }
      modelParameters[i->second.id] = i->second.estimated ;
    }
    for (map<patString, patNetworkGevLinkParameter>::iterator i =
	   networkGevLinks.begin() ;
	 i != networkGevLinks.end() ;
	 ++i) {
      if (i->second.alpha.id >= modelParameters.size()) {
	err = new patErrOutOfRange<unsigned long>(i->second.alpha.id ,
						   0,
						   modelParameters.size()-1) ;
	WARNING(err->describe()) ;
	return ;
      }
      modelParameters[i->second.alpha.id] = i->second.alpha.estimated ;
    }
  }

  for (map<patString,patBetaLikeParameter>::const_iterator i =
	 scaleParam.begin() ;
       i != scaleParam.end() ;
       ++i) {
    scaleParameters[i->second.id] = i->second.estimated ;
    if (i->second.id >= scaleParameters.size()) {
      err = new patErrOutOfRange<unsigned long>(0,
						 i->second.id ,
						 scaleParameters.size()-1) ;
      WARNING(err->describe()) ;
      return ;
    }
  }

}

void patModelSpec::addRatio(const patString& num,
			    const patString& denom,
			    const patString& name) {
  ratios[name] = pair<patString, patString>(num,denom) ;
}

void patModelSpec::addConstraintNest(const patString& firstNest,
				     const patString& secondNest) {
  constraintNests.push_back(pair<patString, patString>(firstNest,secondNest)) ;
}

patIterator<patModelSpec::patConstantProductIndex>* patModelSpec::
createConstantProductIterator(patError*& err) {
  vector<patConstantProductIndex> indexList ;
  for (vector<patConstantProduct>::const_iterator item = 
	 listOfConstantProducts.begin() ;
       item != listOfConstantProducts.end() ;
       ++item) {
    patBoolean success;
    unsigned long p1 = getIndexFromName(item->param1,&success) ;
    if (success) {
      unsigned long p2 = getIndexFromName(item->param2,&success) ;
      if (success) {
	patConstantProductIndex cpi ;
	cpi.index1 = p1 ;
	cpi.index2 = p2 ;
	cpi.cte = item->cte ;
	indexList.push_back(cpi) ;
      }
      else {
	WARNING("Cannot identify parameter " << item->param2 << ". Constraint ignored") ;

      }
    }
    else {
      WARNING("Cannot identify parameter " << item->param1 << ". Constraint ignored") ;
    }
  }

  patIterator<patModelSpec::patConstantProductIndex>* iter = 
    new patStlVectorIterator<patModelSpec::patConstantProductIndex>(indexList) ;
  return (iter) ;


}

patIterator<pair<unsigned long, unsigned long> >* patModelSpec::
createConstraintNestIterator(patError*& err) {
  vector<pair<unsigned long, unsigned long> > indexList ;

  patBoolean found ;
  // Nested logit
  if (isNL()) {
    for (vector<pair<patString, patString> >::const_iterator i = 
	   constraintNests.begin() ;
	 i != constraintNests.end() ;
	 ++i) {
      patBetaLikeParameter param1 = getNlNest(i->first,&found) ;
      if (!found) {
	WARNING("Unknown parameter: " << i->first) ;
	return NULL ;
      }
      patBetaLikeParameter param2 = getNlNest(i->second,&found) ;
      if (!found) {
	WARNING("Unknown parameter: " << i->second) ;
	return NULL ;
      }
      indexList.push_back(pair<unsigned long, unsigned long>(param1.index,param2.index)) ;
    }
  }


  // Cross-Nested logit

  
  if (isCNL()) {
    for (vector<pair<patString, patString> >::const_iterator i = 
	   constraintNests.begin() ;
	 i != constraintNests.end() ;
	 ++i) {
      patBetaLikeParameter param1 = getCnlNest(i->first,&found) ;
      if (!found) {
	WARNING("Unknown parameter: " << i->first) ;
	return NULL ;
      }      
      patBetaLikeParameter param2 = getCnlNest(i->second,&found) ;
      if (!found) {
	WARNING("Unknown parameter: " << i->second) ;
	return NULL ;
      }
      if (param1.isFixed) {
	WARNING("Parameter " << i->first << " is fixed. Constraint "
		<< i->first << " = " << i->second << " ignored") ;
      }
      else if (param2.isFixed) {
	WARNING("Paramater " << i->second << " is fixed. Constraint "
		<< i->first << " = " << i->second << " ignored") ;
      }
      else {
	indexList.push_back(pair<unsigned long, unsigned long>(param1.index,
								 param2.index)) ;
      }
    }
  }

  // Network Gev Model

  if (isNetworkGEV()) {
    for (vector<pair<patString, patString> >::const_iterator i = 
	   constraintNests.begin() ;
	 i != constraintNests.end() ;
	 ++i) {
      patBetaLikeParameter param1 = getNetworkNode(i->first,&found) ;
      if (!found) {
	WARNING("Unknown parameter: " << i->first) ;
	return NULL ;
      }      
      patBetaLikeParameter param2 = getNetworkNode(i->second,&found) ;
      if (!found) {
	WARNING("Unkown parameter: " << i->second) ;
	return NULL ;
      }
      if (param1.isFixed) {
	WARNING("Paramater " << i->first << " is fixed. Constraint "
		<< i->first << " = " << i->second << " ignored") ;
      }
      else if (param2.isFixed) {
	WARNING("Paramater " << i->second << " is fixed. Constraint "
		<< i->first << " = " << i->second << " ignored") ;
      }
      else {
	indexList.push_back(pair<unsigned long, unsigned long>(param1.index,
								 param2.index)) ;
      }
    }
  }

  patIterator<pair<unsigned long, unsigned long> >* iter = 
    new patConstraintNestIterator(indexList) ;
  return (iter) ;
} 

void patModelSpec::addConstantProduct(const patString& firstParam,
				      const patString& secondParam,
				      patReal value) {
  listOfConstantProducts.push_back(patConstantProduct(firstParam,
						      secondParam,
						      value)) ;
}


unsigned long patModelSpec::getIndexFromName(const patString& name,
					      patBoolean* success) {
  // Try if it is a beta parameter
  patBoolean exists ;
  unsigned long index = getBetaIndex(name,&exists) ;
  if (exists) {
    *success = patTRUE ;
    return index ;
  }
  // Try if it is a nest name.
  patBoolean found ;
  if (isNL()) {
    patBetaLikeParameter param = getNlNest(name,&found) ;
    if (found) {
      *success = patTRUE ;
      return param.index ;
    }
  }
  if (isCNL()) {
    patBetaLikeParameter param = getCnlNest(name,&found) ;
    if (found) {
      *success = patTRUE ;
      return param.index ;
    }
  }
  if (isNetworkGEV()) {
    patBetaLikeParameter param = getNetworkNode(name,&found) ;
    if (found) {
      *success = patTRUE ;
      return param.index ;
    }
  }

  // Other possibilities will be implemented later on.
  *success = patFALSE ;
  return patBadId ;

}

patIterator<unsigned long>* 
patModelSpec::getAltIteratorForNonZeroAlphas(unsigned long nest) {

  if (nest >= nonZeroAlphasPerNest.size()) {
    WARNING("Unknown nest " << nest) ;
    return NULL ;
  }
  if (iterPerNest[nest] == NULL) {
    iterPerNest[nest] = 
      new patStlVectorIterator<unsigned long>(nonZeroAlphasPerNest[nest]) ;
  }
  return iterPerNest[nest] ;
}


patIterator<unsigned long>* 
patModelSpec::getNestIteratorForNonZeroAlphas(unsigned long alt) {
  if (alt >= nonZeroAlphasPerAlt.size()) {
    WARNING("Unknown alt " << alt) ;
    return NULL ;
  }
  if (iterPerAlt[alt] == NULL) {
    iterPerAlt[alt]  = 
      new patStlVectorIterator<unsigned long>(nonZeroAlphasPerAlt[alt]) ;
  }
  return iterPerAlt[alt] ;

}


void patModelSpec::addNetworkGevNode(const patString& name,
				     patReal defaultValue,
				     patReal lowerBound,
				     patReal upperBound,
				     patBoolean isFixed) {
  
  if (defaultValue < lowerBound) {
    WARNING("Default value for " << name << " set to " << lowerBound) ;
    defaultValue = lowerBound ;
  }
  if (defaultValue > upperBound) {
    WARNING("Default value for " << name << " set to " << upperBound) ;
    defaultValue = upperBound ;
  }
  patBetaLikeParameter node ;
  node.name = name ;
  node.defaultValue = defaultValue ;
  node.lowerBound = lowerBound ;
  node.upperBound = upperBound ;
  node.isFixed = isFixed ;
  node.estimated = defaultValue ;
  node.index = patBadId ;
  node.id = patBadId ;
  networkGevNodes[name] = node ;
}
		   
void patModelSpec::addNetworkGevLink(const patString& aNodeName,
				     const patString& bNodeName,
				     patReal defaultValue,
				     patReal lowerBound,
				     patReal upperBound,
				     patBoolean isFixed) {
  patString completeName = buildLinkName(aNodeName,bNodeName) ;
  if (defaultValue < lowerBound) {
    WARNING("Default value for " << completeName << " set to " << lowerBound) ;
    defaultValue = lowerBound ;
  }
  if (defaultValue > upperBound) {
    WARNING("Default value for " << completeName << " set to " << upperBound) ;
    defaultValue = upperBound ;
  }
  patBetaLikeParameter bp ;
  bp.name = completeName ;
  bp.defaultValue = defaultValue ;
  bp.lowerBound = lowerBound ;
  bp.upperBound = upperBound ;
  bp.isFixed = isFixed ;
  bp.estimated = defaultValue ;
  bp.index = patBadId ;
  bp.id = patBadId ;

  patNetworkGevLinkParameter lp ;
  lp.alpha = bp ;
  lp.aNode = aNodeName ;
  lp.bNode = bNodeName ;
  networkGevLinks[completeName] = lp ;
}
		   
unsigned long patModelSpec::getNetworkGevNodeId(const patString&  nodeName) {
  map<patString, patBetaLikeParameter>::iterator found =
    networkGevNodes.find(nodeName) ;
  if (found == networkGevNodes.end()) {
    //    WARNING("Node " << nodeName << " not known") ;
    return patBadId ;
  }
  return found->second.id ;
}


patGEV* patModelSpec::getNetworkGevModel() {
  if (theNetworkGevModel == NULL) {
    theNetworkGevModel = 
      new patNetworkGevModel(&networkGevNodes,&networkGevLinks) ;
  }
  return theNetworkGevModel->getModel() ;
}

patBoolean patModelSpec::isSampleWeighted() {
  return (weightExpr != NULL) ;
}

unsigned long patModelSpec::nbrSimulatedChoiceInSampleEnumeration() {
  return sampleEnumeration ;
}

void patModelSpec::writeGnuplotFile(patString fileName, 
				    patError*& err) {
  

  if (!gnuplotDefined) {
    return ;
  }

  ofstream gnuplotFile(fileName.c_str()) ;
  patAbsTime now ;
  now.setTimeOfDay() ;
  gnuplotFile << "# This file has automatically been generated." << endl ;
  gnuplotFile << "# " << now.getTimeString(patTsfFULL) << endl ;
  gnuplotFile << "# " << patVersion::the()->getCopyright() << endl ;
  gnuplotFile << "#" << endl ;
  gnuplotFile << "# " <<  patVersion::the()->getVersionInfoDate() << endl ;
  gnuplotFile << "# " <<  patVersion::the()->getVersionInfoAuthor() << endl ;
  gnuplotFile << endl ;
  gnuplotFile << "# This file has been designed to be read by:" << endl ;
  gnuplotFile << "#        G N U P L O T" << endl ;
  gnuplotFile << "#        Linux version 3.7" << endl ;
  gnuplotFile << "#        patchlevel 1" << endl ;
  gnuplotFile << "#        last modified Fri Oct 22 18:00:00 BST 1999" << endl ;
  gnuplotFile << "#" << endl ;
  gnuplotFile << "#########################################################" << endl ;
  gnuplotFile << "# We strongly recommend to read the gnuplot documentation." << endl ;
  gnuplotFile << "#########################################################" << endl ;
  gnuplotFile << "# Instructions:" << endl ;
  gnuplotFile << "#  1) Choose the attribute you want to vary" << endl ;
  gnuplotFile << "#  2) Set it to 'x' below. For example 'travelTime(x) = x'." << endl ;
  gnuplotFile << "#  3) Choose the range for that attribute by editing the command 'set xrange' below" 
	      << endl ;
  gnuplotFile << "#  4) Assign numeric values to all other attributes." << endl ;
  gnuplotFile << "#  5) Type 'gnuplot " << fileName << "'" << endl ;
  gnuplotFile << "#  6) If you want a Encapsulated Postscript output," << endl ;
  gnuplotFile << "#     uncommment the two following lines and, if needed, modify the filename."  << endl ;
  gnuplotFile << "#set terminal postscript eps monochrome" << endl ;
  gnuplotFile << "#set output \"biosim.eps\"" << endl ;
  gnuplotFile << "#" << endl ;
  gnuplotFile << "# Specify here the range for the variable under interest" << endl ;
  gnuplotFile << "set xrange[" << gnuplotMin << ":" << gnuplotMax << "]" << endl;
  gnuplotFile << "#Specify here the range for the Y-axis" << endl ;
  gnuplotFile << "set yrange[0:1]" << endl;
  gnuplotFile << "# Give a value for each data, except one which must be 'x'" << endl ;
  gnuplotFile << setprecision(7) << setiosflags(ios::scientific) ;  
  for (vector<patString>::iterator i = headers.begin() ;
       i != headers.end() ;
       ++i) {
    if (*i == gnuplotParam) {
      gnuplotFile << *i << "(x) = x" << endl ;  
    }
    else {
      gnuplotFile << *i << "(x) = 0.0" << endl ;  
    }
  }

  gnuplotFile << "##############################################" << endl ;
  gnuplotFile << "# You should not have to edit below this line" << endl ;
  gnuplotFile << "##############################################" << endl ;

  gnuplotFile << "#" << endl ;
  gnuplotFile << "# User defined expressions" << endl ;
  gnuplotFile << "#" << endl ;
  for (map<patString,patArithNode*>::iterator i = expressions.begin() ;
       i != expressions.end() ;
       ++i) {
    patString withSpace(i->first) ;
    withSpace += " " ;
    if (withSpace != i->second->getExpression(err)) {
      gnuplotFile << i->first << "(x)=" << i->second->getGnuplot(err) << endl;
    }
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
  gnuplotFile << "#" << endl ;
  gnuplotFile << "# Beta parameters" << endl ;
  gnuplotFile << "#" << endl ;
  patIterator<patBetaLikeParameter>* iter = createAllBetaIterator() ;
  for (iter->first() ;
       !iter->isDone() ;
       iter->next()) {
    gnuplotFile << iter->currentItem().name << "=" 
		<< iter->currentItem().estimated << endl ;
  }

  gnuplotFile << "#" << endl ;
  gnuplotFile << "# Mu" << endl ;
  gnuplotFile << "#" << endl ;
  gnuplotFile << "mu=" << mu.estimated << endl ;
  gnuplotFile << "#" << endl ;
  gnuplotFile << "# Model parameters" << endl ;
  gnuplotFile << "#" << endl ;
  patIterator<patBetaLikeParameter>* miter = createAllModelIterator() ;
  for (miter->first() ;
       !miter->isDone() ;
       miter->next()) {
    gnuplotFile << miter->currentItem().name << "=" 
		<< miter->currentItem().estimated << endl ;
  }
  gnuplotFile << "#" << endl ;
  gnuplotFile << "#" << endl ;
  gnuplotFile << "# Utility functions" << endl ;
  gnuplotFile << "#" << endl ;

  for (map<patString, patAlternative>::iterator i = utilities.begin() ;
       i != utilities.end() ;
       ++i) {
    gnuplotFile << "U" << i->second.userId << "(x)=" ;
    patUtilFunction* util = &(i->second.utilityFunction) ;
    for (list<patUtilTerm>::iterator ii = util->begin() ;
	 ii != util->end() ;
	 ++ii) {
      if (ii != util->begin()) {
	gnuplotFile << " + " ;
      }
      gnuplotFile << ii->beta << " * " << ii->x << "(x)" ;
    }
    map<unsigned long, patArithNode*>::iterator found = 
      nonLinearUtilities.find(i->second.userId) ;
    if (found != nonLinearUtilities.end()) {
      gnuplotFile << " + " << found->second->getGnuplot(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
    gnuplotFile << endl ;
  }
  unsigned long J = getNbrAlternatives() ;
  switch (modelType) {
  case patOLtype :
    gnuplotFile << "## Ordered logit not implement" << endl ;
  case patBPtype :
    gnuplotFile << "## Binary probit not implemented" << endl ;
    break ;
  case patMNLtype :
    gnuplotFile << "#" << endl ;
    gnuplotFile << "# Logit Model" << endl ; 
    gnuplotFile << "#" << endl ;
    gnuplotFile << "sumexp(x)=" ;
    for (unsigned long alt = 0 ; alt < J ; ++alt) {
      unsigned long userId = getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }
      if (alt != 0) {
	gnuplotFile << "+" ;
      }
      gnuplotFile << "exp(mu*U" << userId << "(x))" ;

    }
    gnuplotFile << endl ;
    for (unsigned long alt = 0 ; alt < J ; ++alt) {
      unsigned long userId = getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }
      gnuplotFile << "P" << userId << "(x)=" <<  "exp(mu*U" << userId 
		  << "(x)) / sumexp(x)" << endl ;
    }
    break ;
  case patNLtype :
    gnuplotFile << "#" << endl ;
    gnuplotFile << "# Nested Logit Model" << endl ; 
    gnuplotFile << "#" << endl ;
    for (unsigned long alt = 0 ; alt < J ; ++alt) {
      unsigned long userId = getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }
      gnuplotFile << "Y" << userId << "(x)=exp(U"
		  << userId
		  << "(x))" << endl ;
    }    
    for (map<patString,patNlNestDefinition>::iterator i = nlNestParam.begin() ;
	 i != nlNestParam.end() ;
	 ++i) {
      gnuplotFile << "logsum" << i->first << "(x)=(" ;
      for (list<long>::iterator alt = i->second.altInNest.begin() ;
	   alt != i->second.altInNest.end() ;
	   ++alt) {
	if (alt != i->second.altInNest.begin()) {
	  gnuplotFile << "+" ;
	}
	gnuplotFile << "(Y" << *alt << "(x)**" 
		    << i->second.nestCoef.name  <<")" ;
      }
      gnuplotFile << ")**(mu/" << i->second.nestCoef.name << " - 1.0)" << endl ;
    }
    for (map<patString,patNlNestDefinition>::iterator i = nlNestParam.begin() ;
	 i != nlNestParam.end() ;
	 ++i) {
      for (list<long>::iterator alt = i->second.altInNest.begin() ;
	   alt != i->second.altInNest.end() ;
	   ++alt) {
	gnuplotFile << "G" << *alt << "(x)= mu * (Y" << *alt << "(x)**(" 
		    << i->second.nestCoef.name << "-1.0)) * logsum"  
		    << i->first << "(x)" << endl ;
      }
    }
    gnuplotFile << "sumexp(x)=" ;
    for (unsigned long alt = 0 ; alt < J ; ++alt) {
      unsigned long userId = getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }
      if (alt != 0) {
	gnuplotFile << "+" ;
      }
      gnuplotFile << "exp(U" << userId
		  << "(x)+log(G" << userId << "(x)))" ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }
    }
    gnuplotFile << endl ;
    for (unsigned long alt = 0 ; alt < J ; ++alt) {
      unsigned long userId = getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }
      gnuplotFile << "P" << userId << "(x)= exp(U" 
		  << userId << "(x)+log(G" 
		  << userId << "(x))) / sumexp(x)" << endl ;
    }
    break ;
    
  case patCNLtype:
    gnuplotFile << "#" << endl ;
    gnuplotFile << "# Cross-Nested Logit Model" << endl ; 
    gnuplotFile << "#" << endl ;
    for (unsigned long alt = 0 ; alt < J ; ++alt) {
      unsigned long userId = getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }
      gnuplotFile << "Y" << userId << "(x)=exp(U"
		  << userId 
		  << "(x))" << endl ;
    }    
    for (map<patString,patBetaLikeParameter>::iterator i = 
	   cnlNestParam.begin() ;
	 i != cnlNestParam.end() ;
	 ++i) {
      patBoolean found ;
      patBetaLikeParameter mum = getCnlNest(i->first,&found) ; 
      if (!found) {
	stringstream str ;
	str << "Unknown CNL nest parameter: " << i->first ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe());
	return ;
      }
      gnuplotFile << "logsum" << i->first << "(x)=(" ;
      patBoolean firstAlpha = patTRUE ;
      for (map<patString,patCnlAlphaParameter>::iterator 
	       alphaIter = cnlAlphaParam.begin() ;
	     alphaIter != cnlAlphaParam.end() ;
	     ++alphaIter) {
	if (alphaIter->second.nestName == i->first) {
	  if (firstAlpha) {
	    firstAlpha = patFALSE ;
	  }
	  else {
	    gnuplotFile << "+" ;
	  }
	  patString altName = alphaIter->second.altName ;
	  patBoolean found ;
	  patBetaLikeParameter alpha = getCnlAlpha(alphaIter->second.nestName, 
						   alphaIter->second.altName,
						   &found) ;
	  if (!found) {
	    stringstream str ;
	    str << "UnknownCNl alpha parameter for nest " << alphaIter->second.nestName << " and alt. " << alphaIter->second.altName ;
	    err = new patErrMiscError(str.str()) ;
	    WARNING(err->describe());
	    return ;
	  }
	  
	  patBetaLikeParameter mum = cnlNestParam[alphaIter->second.nestName] ;
	  unsigned long altUserId = utilities[altName].userId ;
	  gnuplotFile << alpha.name << "*(Y" << altUserId
		      << "(x)**" << mum.name << ")" ;
	}
      }
      gnuplotFile << ")**((mu/" << mum.name << ")-1.0)" << endl ;

    }
    for (unsigned long alt = 0 ; alt < J ; ++alt) {
      unsigned long userId = getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }
      patString altName = getAltName(userId,err)  ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }
      
      gnuplotFile << "G" << userId << "(x)=mu*(" ;
      
      patBoolean first = patTRUE ;
      for (map<patString,patCnlAlphaParameter>::iterator 
	     alphaIter = cnlAlphaParam.begin() ;
	   alphaIter != cnlAlphaParam.end() ;
	   ++alphaIter) {
	if (alphaIter->second.altName == altName) {
	  patBoolean found ;
	  patBetaLikeParameter alpha = getCnlAlpha(alphaIter->second.nestName, 
						   alphaIter->second.altName,
						   &found) ;
	  if (!found) {
	    stringstream str ;
	    str << "UnknownCNl alpha parameter for nest " 
		<< alphaIter->second.nestName 
		<< " and alt. " 
		<< alphaIter->second.altName ;
	    err = new patErrMiscError(str.str()) ;
	    WARNING(err->describe());
	    return ;
	  }
	  if (first) {
	    first = patFALSE ;
	  }
	  else {
	    gnuplotFile << "+" ;
	  }
	  gnuplotFile << alpha.name << "*(Y" << userId << "(x)**(" 
		      << cnlNestParam[alphaIter->second.nestName].name 
		      << "-1.0) * logsum" 
		      << alphaIter->second.nestName << "(x))" ;
	}
	
      }    
      gnuplotFile << ")" << endl ;
    }

    gnuplotFile << "sumexp(x)=" ;
    for (unsigned long alt = 0 ; alt < J ; ++alt) {
      unsigned long userId = getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }
      if (alt != 0) {
	gnuplotFile << "+" ;
      }
      gnuplotFile << "exp(U" << userId
		  << "(x)+log(G" << userId << "(x)))" ;
    }
    gnuplotFile << endl ;
    for (unsigned long alt = 0 ; alt < J ; ++alt) {
      unsigned long userId = getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }
      gnuplotFile << "P" << userId << "(x)= exp(U" 
		  << userId << "(x)+log(G" 
		  << userId << "(x))) / sumexp(x)" << endl ;
    }
    
    
    break ;

  case patNetworkGEVtype:
    gnuplotFile << "#" << endl ;
    gnuplotFile << "# Network GEV Model" << endl ; 
    gnuplotFile << "#" << endl ;
    for (unsigned long alt = 0 ; alt < J ; ++alt) {
      unsigned long userId = getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }
      gnuplotFile << "Y" << userId << "(x)=exp(U"
		  << userId 
		  << "(x))" << endl ;
    }    
    nodesInGnuplotFile.erase(nodesInGnuplotFile.begin(),nodesInGnuplotFile.end()) ;
    gnuplotNetworkGevNode(theNetworkGevModel->getRoot(),gnuplotFile,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
    }
    gnuplotFile << "#" << endl ;
    gnuplotFile << "# Probabilities" << endl ;
    gnuplotFile << "#" << endl ;
    
    gnuplotFile << "sumexp(x)=" ;
    for (unsigned long alt = 0 ; alt < J ; ++alt) {
      unsigned long userId = getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }
      if (alt != 0) {
	gnuplotFile << "+" ;
      }
      gnuplotFile << "exp(U" << userId
		  << "(x)+log(DG" << rootNodeName << userId << "(x)))" ;
    }
    gnuplotFile << endl ;
    for (unsigned long alt = 0 ; alt < J ; ++alt) {
      unsigned long userId = getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }
      gnuplotFile << "P" << userId << "(x)= exp(U" 
		  << userId << "(x)+log(DG" << rootNodeName
		  << userId << "(x))) / sumexp(x)" << endl ;
    }
    break ;
  }
  gnuplotFile << "set xlabel \"" << gnuplotParam << "\"" << endl ;
  gnuplotFile << "plot " ;
  for (unsigned long alt = 0 ; alt < J ; ++alt) {
    unsigned long userId = getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }
    if (alt != 0) {
      gnuplotFile << ", "  ;
    }
    gnuplotFile << "P" << userId << "(x) title \"" 
		<< getAltName(userId,err) << "\"";
      if (err != NULL) {
	WARNING(err->describe());
	return ;
      }
  }


  gnuplotFile << endl ;
  gnuplotFile << "pause -1 \"Press a key\"" << endl ;
  gnuplotFile.close() ;
  patOutputFiles::the()->addUsefulFile(fileName,"Gnuplot file for plotting the model");
  
}

void patModelSpec::setColumns(unsigned long i) {  
  columnData.push_back(i) ;
}

//unsigned long patModelSpec::getColumns() {
//  return columnData ;
//}


ostream& patModelSpec::gnuplotNetworkGevNode(patNetworkGevNode* node, 
					     ostream& file, 
					     patError*& err) {
  if (node == NULL) {
    return file ;
  }
  std::set<patString>::const_iterator found = 
    nodesInGnuplotFile.find(node->getNodeName()) ;
  if (found != nodesInGnuplotFile.end()) {
    return file ;
  }
  unsigned long J = getNbrAlternatives() ;
  patIterator<patNetworkGevNode*>* theIter = node->getSuccessorsIterator() ;
  assert(theIter) ;
  for (theIter->first() ;
       !theIter->isDone() ;
       theIter->next()) {
    patNetworkGevNode* aSucc = theIter->currentItem() ;
    gnuplotNetworkGevNode(aSucc,file,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return file ;
    }
  }
  
  file << "#" << endl ;
  file << "# Node " << node->getNodeName() << endl ;
  file << "#" << endl ;
  if (node->isAlternative()) {
    std::set<unsigned long> alt = node->getRelevantAlternatives() ;
    //    unsigned long muIndex = node->getMuIndex() ;
    map<patString, patAlternative>::iterator found = 
      utilities.find(node->getNodeName()) ;
    if (found == utilities.end()) {
      stringstream str ;
      str << "Alternative " << node->getNodeName() << " unknown" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return file ;
    } 
    if (found->second.id != *alt.begin()) {
      stringstream str ;
      str << "NetworkGEV error. Alt " << node->getNodeName() 
	  << ". Id=" << found->second.id << " and relevant alt. is " 
	  << *alt.begin() ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return file ;
    }
    file << "G" << node->getNodeName()  << "(x)=Y" 
	 << found->second.userId << "(x)" << endl ;
    for (unsigned long alt = 0 ; alt < J ;  ++alt) {
      unsigned long userId = getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return file ;
      }
      if (userId == found->second.userId) {
	file << "DG" << node->getNodeName() << userId << "(x)=1.0" << endl ; 
      }
      else {
	file << "DG" << node->getNodeName() << userId << "(x)=0.0" << endl ; 
      }
    }
  }
  else {
    // Node is not an alternative
    file << "G" << node->getNodeName()  << "(x)=" ;
    patIterator<patNetworkGevNode*>* iter = node->getSuccessorsIterator() ;
    patBoolean first = patTRUE ;
    patString mui = (node->isRoot()) 
      ? "mu" 
      : node->getNodeName() ;
    for (iter->first() ;
	 !iter->isDone();
	 iter->next()) {
      if (first) {
	first = patFALSE ;
      }
      else {
	file << "+" ;
      }
      patNetworkGevNode* childNode = iter->currentItem() ;
      patString alpha = buildLinkName(node->getNodeName(),
				      childNode->getNodeName()) ;
      if (childNode->isAlternative()) {
	file << alpha << "*(G" << childNode->getNodeName() 
	     << "(x)**" << mui << ")" ;
      }
      else {
	file << alpha << "*(G" << childNode->getNodeName() << "(x)**(" 
	     << mui << "/" << childNode->getNodeName() <<"))" ;

      }
    }
    file << endl ;

    for (unsigned long alt = 0 ; alt < J ; ++alt) {
      unsigned long userId = getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return file ;
      }
      file << "DG" << node->getNodeName()  << userId << "(x)=" ;
      first = patTRUE ;
      for (iter->first() ;
	   !iter->isDone();
	   iter->next()) {
	if (first) {
	  first = patFALSE ;
	}
	else {
	  file << "+" ;
	}
	patNetworkGevNode* childNode = iter->currentItem() ;
	patString alpha = buildLinkName(node->getNodeName(),
					childNode->getNodeName()) ;
	if (childNode->isAlternative()) {
	  file << alpha << "*" << mui << "*(G" << childNode->getNodeName() 
	       << "(x)**(" << mui << "-1))*DG" << childNode->getNodeName() << userId << "(x)" ;
	}
	else {
	  file << alpha << "*(" << mui << "/" << childNode->getNodeName()  
	       << ")*(G" << childNode->getNodeName() 
	       << "(x)**((" << mui << "/" << childNode->getNodeName()  
	       << ")-1))*DG" << childNode->getNodeName() << userId << "(x)" ;
	}
      }
      file << endl ;
    }

  }

  nodesInGnuplotFile.insert(node->getNodeName()) ;
  return file ;
}

void patModelSpec::setListLinearConstraints(patListLinearConstraint* ptr) {
  if (ptr != NULL) {
    listOfLinearConstraints = *ptr ;
  }
}

patListProblemLinearConstraint patModelSpec::getLinearEqualityConstraints(patError*& err) {
  patListProblemLinearConstraint result ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patListProblemLinearConstraint() ;
  }
  for (patListLinearConstraint::iterator aConstraint = 
	 listOfLinearConstraints.begin() ;
       aConstraint != listOfLinearConstraints.end() ;
       ++aConstraint) {
    if (aConstraint->theType == patLinearConstraint::patEQUAL) {
      unsigned long n = getNbrNonFixedParameters() ;
      patProblemLinearConstraint 
	probCons(patVariables(n,0.0),
		 aConstraint->theRHS) ;
      
      // First check that all variables must be estimated
      patBoolean nonZeroCoef(patFALSE) ;
      for (patConstraintEquation::iterator aTerm = 
	     aConstraint->theEquation.begin() ;
	   aTerm != aConstraint->theEquation.end() ;
	   ++aTerm) {
	
	patBoolean found ;
	patBetaLikeParameter beta  = getParameter(aTerm->param,&found) ;
	if (!found) {
	  stringstream str ;
	  str << "Unknown parameter: " << aTerm->param ;
	  err = new patErrMiscError(str.str()) ;
	  return patListProblemLinearConstraint() ;
	}
	//	DEBUG_MESSAGE("--> found " << beta.name) ;
	if (beta.isFixed) {
	  probCons.second -= aTerm->fact * beta.defaultValue ;
	}
	else {
	  //DEBUG_MESSAGE("--> index = " << beta.index) ; 
	  if (aTerm->fact != 0.0) {
	    nonZeroCoef = patTRUE ;
	    probCons.first[beta.index] = aTerm->fact ;
	  }
	}
      }
      if (nonZeroCoef) {
	result.push_back(probCons) ;
      }
    }
  }
  return result ;
}

patListProblemLinearConstraint patModelSpec::getLinearInequalityConstraints(patError*& err) {
  patListProblemLinearConstraint result ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patListProblemLinearConstraint() ;
  }
  for (patListLinearConstraint::iterator aConstraint = 
	 listOfLinearConstraints.begin() ;
       aConstraint != listOfLinearConstraints.end() ;
       ++aConstraint) {
    patBoolean lessOrEqual(aConstraint->theType == patLinearConstraint::patLESSEQUAL) ;
    if (lessOrEqual ||
	aConstraint->theType == patLinearConstraint::patGREATEQUAL) {
      unsigned long n = getNbrNonFixedParameters() ;
      patProblemLinearConstraint 
	probCons(patVariables(n,0.0),
		 (lessOrEqual)
		 ?aConstraint->theRHS
		 :-aConstraint->theRHS) ;
      
      // First check that all variables must be estimated
      patBoolean nonZeroCoef(patFALSE) ;
      for (patConstraintEquation::iterator aTerm = aConstraint->theEquation.begin() ;
	   aTerm != aConstraint->theEquation.end() ;
	   ++aTerm) {
	patBoolean found ;
	patBetaLikeParameter beta  = getParameter(aTerm->param,&found) ;
	if (!found) {
	  stringstream str ;
	  str << "Unknown parameter: " << aTerm->param ;
	  err = new patErrMiscError(str.str()) ;
	  return patListProblemLinearConstraint() ;
	}
	if (beta.isFixed) {
	  if (lessOrEqual) {
	    probCons.second -= aTerm->fact * beta.defaultValue ;
	  }
	  else {
	    probCons.second += aTerm->fact * beta.defaultValue ;
	  }
	}
	else {
	  if (aTerm->fact != 0.0) {
	    nonZeroCoef = patTRUE ;
	    if (lessOrEqual) {
	      probCons.first[beta.index] = aTerm->fact ;
	    }
	    else {
	      probCons.first[beta.index] = -aTerm->fact ;
	    }
	  }
	}
      }
      if (nonZeroCoef) {
	result.push_back(probCons) ;
      }
    }
  }
  return result ;
}

patString patModelSpec::printEqConstraint(patProblemLinearConstraint aConstraint) {
  
  stringstream str ;
  patBoolean firstTerm(patTRUE) ;
  patReal LHS(0.0) ;
  for (patVariables::size_type i = 0 ;
       i < aConstraint.first.size() ;
       ++i) {
    if (aConstraint.first[i] != 0.0) {
      if (firstTerm) {
	firstTerm = patFALSE ;
      }
      else {
	str << " + " ;
      }
      LHS += aConstraint.first[i] * nonFixedParameters[i]->estimated ;
      str << aConstraint.first[i] << "*" << nonFixedParameters[i]->name ;
    }
  }
  str << " = " << aConstraint.second ;
  str << " [" << LHS << " = " << aConstraint.second << "]" ;
  patString res(str.str()) ;
  return res ;
}

patString patModelSpec::printIneqConstraint(patProblemLinearConstraint aConstraint) {
  
  stringstream str ;
  patBoolean firstTerm(patTRUE) ;
  patReal LHS(0.0) ;
  for (patVariables::size_type i = 0 ;
       i < aConstraint.first.size() ;
       ++i) {
    if (aConstraint.first[i] != 0.0) {
      if (firstTerm) {
	firstTerm = patFALSE ;
      }
      else {
	str << " + " ;
      }
      LHS += aConstraint.first[i] * nonFixedParameters[i]->estimated ;
      str << aConstraint.first[i] << "*" << nonFixedParameters[i]->name ;
    }
  }
  str << " <= " << aConstraint.second ;
  str << " [" << LHS << " <= " << aConstraint.second ;
  patString res(str.str()) ;
  return res ;
}

patBetaLikeParameter patModelSpec::getParameter(const patString& name,
						patBoolean* found) const {
  
  patBetaLikeParameter result ;

  // Check if it is a beta parameter
  
  result = getBeta(name,found) ;
  if (*found) {
    return result ;
  }

  // Check if it is a NL nest parameter 

  if (modelType == patNLtype) {
    result = getNlNest(name,found) ;
    if (*found) {
      return result ;
    }
  }

  // Check if it is a CNL nest parameter
  
  if (modelType == patCNLtype) {
    result = getCnlNest(name,found) ;
    if (*found) {
      return result ;
    }
  }
  
  // Check if it is a CNL alpha parameter
  
  if (modelType == patCNLtype) {
    pair<patString,patString> cnl = getCnlAlphaAltNest(name,
						       found)  ;
    if (*found) {
      patBoolean cnlFound  ;
      result = getCnlAlpha(cnl.second,cnl.first,&cnlFound) ;
      if (cnlFound) {
	return result ;
      }
    }
  }
  
  // Check if it is a Network GEV node parameter

  if (modelType == patNetworkGEVtype) {
    result = getNetworkNode(name,found) ;
    if (*found) {
      return result ;
    }
    
  }
  // Check if it is a Network GEV link parameter

  if (modelType == patNetworkGEVtype) {
    result = getNetworkLink(name,found) ;
    if (*found) {
      return result ;
    }
    
  }

  // Not found...

  *found = patFALSE ;
  return patBetaLikeParameter() ;
}



void patModelSpec::setListNonLinearEqualityConstraints(patListNonLinearConstraints* ptr) {
  equalityConstraints = ptr ;
}

void patModelSpec::setListNonLinearInequalityConstraints(patListNonLinearConstraints* ptr) {
  inequalityConstraints = ptr ;
}

patListNonLinearConstraints* patModelSpec::getNonLinearEqualityConstraints(patError*& err) {
  return equalityConstraints ;
}

patListNonLinearConstraints* patModelSpec::getNonLinearInequalityConstraints(patError*& err) {
  return inequalityConstraints ;
}

void patModelSpec::addNonLinearUtility(unsigned long id,
				       patArithNode* util) {
  
  nonLinearUtilities[id] = util ;
}

void patModelSpec::addDerivative(unsigned long id,
				 patString param,
				 patArithNode* util) {
  derivatives[id][param] = util ;
}



patArithNode* patModelSpec::getNonLinearUtilExpr(unsigned long altId,
						 patError*& err) {

  map<unsigned long, patArithNode*>::iterator found = 
    nonLinearUtilities.find(altId) ;
  if (found == nonLinearUtilities.end()) {
    return NULL ;
  }
  else {
    return found->second ;
  }
}

patArithNode* patModelSpec::getDerivative(unsigned long internalAltId,
					  unsigned long id,
					  patError*& err) {

  patBoolean found ;
  patBetaLikeParameter theParam = getParameterFromId(id,&found) ;
  if (!found) {
    stringstream str ;
    str << "No parameter with id " << id ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  unsigned long altId = getAltId(internalAltId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }

  patArithNode* result = getDerivative(altId,theParam.name,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return result ;
}


patArithNode* patModelSpec::getDerivative(unsigned long altId,
					  patString param,
					  patError*& err) {

  map<unsigned long, map<patString, patArithNode*> >::iterator 
    found =  derivatives.find(altId) ;

  
  if (found == derivatives.end()) {
    return NULL ;
  }
  
  map<patString, patArithNode*>::iterator refound = found->second.find(param) ;
  if (refound == found->second.end()) {
    return NULL ;
  }
  return refound->second ;


}




unsigned long patModelSpec::getAttributeId(const patString attrName) {

  map<patString, unsigned long>::iterator found = 
    attribIdPerName.find(attrName) ;
  if (found != attribIdPerName.end()) {
    return found->second ;
  }
  else {
    return patBadId ;
  }
}
  
patString patModelSpec::getAttributeName(unsigned long attrId) {
  if (attrId < attributeNames.size()) {
    return attributeNames[attrId] ;
  }
  else {
    return patString("UnknownAttribute") ;
  }
}

patUtility* patModelSpec::getFullUtilityFunction(patError*& err) {

  if (!nonLinearUtilities.empty()) {
    
    // Mixed Logit or nonlinear utilities

    patAdditiveUtility* theUtility = new patAdditiveUtility() ;
    theUtility->addUtility(new patLinearUtility(isMixedLogit())) ;
    theUtility->addUtility(new patGeneralizedUtility()) ;
    return theUtility ;
  }
  else {
    patUtility* ptr = new patLinearUtility(isMixedLogit()) ;
    return ptr ;
  }
}

void patModelSpec::setVariablesBetaWithinExpression(patArithNode* expression) {
  if (expression == NULL) {
    return ;
  }

  for (map<patString,patBetaLikeParameter>::iterator i = betaParam.begin() ;
       i != betaParam.end() ;
       ++i) {

    expression->setVariable(i->second.name,i->second.id) ;
  }
}

patArithRandom*  patModelSpec::addRandomExpression(patArithRandom* p) {
  if (p == NULL) {
    return NULL ;
  }
  patString m = p->getLocationParameter() ;
  patString s = p->getScaleParameter() ;
  patString r = buildRandomName(m,s) ;
  map<patString,pair<patRandomParameter,patArithRandom*> >::iterator found = randomParameters.find(r) ;
  if (found != randomParameters.end()) {
    assert(found->second.second != NULL) ;
    if (found->second.second->getDistribution() != p->getDistribution()) {
      WARNING("Random parameter " << r << " cannot have two different distributions") ;
      return NULL ;
    }
    return found->second.second;
  }
  patRandomParameter theParam ;
  theParam.name = r ;
  theParam.type = p->getDistribution() ;
  randomParameters[r].first = theParam ;
  randomParameters[r].second = p ;
  DEBUG_MESSAGE("Add random parameters: " << r) ;
  return NULL ;
  
}


void patModelSpec::buildCovariance(patError*& err) {

  // First, assign the covariance coefficients to the random variables.

  for (map<pair<patString,patString>,patBetaLikeParameter*>::iterator i = 
	 covarParameters.begin() ;
       i != covarParameters.end() ;
       ++i) {

    patString completeName = buildCovarName(i->first.first,i->first.second) ;
    
    map<patString,pair<patRandomParameter,patArithRandom*> >::iterator found1 =
      randomParameters.find(i->first.first) ;
    if (found1 == randomParameters.end()) {
      stringstream str ;
      str << "Unknown random parameter " <<  i->first.first ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
    map<patString,pair<patRandomParameter,patArithRandom*> >::iterator found2 =
      randomParameters.find(i->first.second) ;
    if (found2 == randomParameters.end()) {
      stringstream str ;
      str << "Unknown random parameter " <<  i->first.second ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }

    if (found1->second.first < found2->second.first) {
      
      found1->second.first.correlatedParameters.push_back(&found2->second.first) ;
      found1->second.second->addCovariance(completeName) ;
    }
    else if (found2->second.first < found1->second.first) {
      found2->second.first.correlatedParameters.push_back(&found1->second.first) ;
      found2->second.second->addCovariance(completeName) ;
    }
    else {
      stringstream str ;
      str << "Not allowed to define a variance in the covariance section for variable " << i->first.first ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;

    }
  }

  // Second, build the linear expression to compute the r.v. from the draws
  
  patUtilFunction linearExpression ;
  
  nbrDrawAttributesPerObservation = 0 ;
  
  for (map<patString,pair<patRandomParameter,patArithRandom*> >::iterator i =
	 randomParameters.begin() ;
       i != randomParameters.end() ;
       ++i) {

    DEBUG_MESSAGE("Name = " << i->first);
    DEBUG_MESSAGE("patRandomParameter : " << i->second.first.name) ;
    DEBUG_MESSAGE(" random expression : " << *(i->second.second)) ;
    
    patRandomParameter* beta = &(i->second.first) ;
    
    patUtilFunction linearExpression ;
    // term for the mean
    patUtilTerm theTerm ;
    patBoolean exists ;
    theTerm.beta = beta->location->name ;
    theTerm.betaIndex = getBetaId(theTerm.beta,err) ;
    theTerm.massAtZero = beta->massAtZero ;
    if (err != NULL) { 
      WARNING(err->describe()) ;
      return ;
    }
    theTerm.x = patParameters::the()->getgevOne() ;
    theTerm.xIndex = getUsedAttributeId(theTerm.x,&exists) ;
    if (!exists) {
      stringstream str ;
      str << "Attribute " << theTerm.x << " not defined" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
    
    
    
    linearExpression.push_back(theTerm) ;
    
    
    // term for the stdDev
    
    
    patString stdDevName = beta->scale->name ;
    unsigned long stdDevId = getBetaId(stdDevName,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    unsigned long stdDevDrawId =  nbrDrawAttributesPerObservation ;
    ++nbrDrawAttributesPerObservation ;
    typeOfDraws.push_back(beta->type) ;
    areDrawsPanel.push_back(beta->panel) ;
    massAtZero.push_back(beta->massAtZero) ;

    patUtilTerm stdDevTerm ;
    stdDevTerm.beta      = stdDevName ;
    stdDevTerm.betaIndex = stdDevId ;
    stdDevTerm.xIndex    = stdDevDrawId ;
    beta->index = stdDevDrawId ;
    
    linearExpression.push_back(stdDevTerm) ;
    
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    i->second.second->setLinearExpression(linearExpression) ;
  }
  
  // Prepare the correlation terms
  
  for (map<patString,pair<patRandomParameter,patArithRandom*> >::iterator i =
	 randomParameters.begin() ;
       i != randomParameters.end() ;
       ++i) {
    patRandomParameter* beta = &(i->second.first) ;
    if (beta->type != NORMAL_DIST && !beta->correlatedParameters.empty()) {
      stringstream str ;
      str << "Correlation can be estimated only for normally distributed parameters" << endl << *beta << " is not normally distributed" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
    for (vector<patRandomParameter*>::iterator j =
	   beta->correlatedParameters.begin() ; 
	 j != beta->correlatedParameters.end() ; 
	 ++j) {
      
      patUtilTerm corrTerm ;
      patBetaLikeParameter* covarParam = getCovarianceParameter(beta->name,
								(*j)->name) ;
      if (covarParam == NULL) {
	err = new patErrNullPointer("patBetaLikeParameter") ;
	WARNING(err->describe()) ;
	return ;
      }
      
      corrTerm.beta = covarParam->name ;
      corrTerm.betaIndex = covarParam->id ;
      corrTerm.xIndex = (*j)->index ; 
      
      i->second.second->addTermToLinearExpression(corrTerm) ;
      
    }
  }
  
  // Register the attributes
  
  for (map<patString,unsigned long>::iterator i = usedAttributes.begin() ;
       i != usedAttributes.end() ;
       ++i) {
    for (map<unsigned long, patArithNode*>::iterator expr =
	   nonLinearUtilities.begin()  ;
	 expr != nonLinearUtilities.end() ;
	 ++expr) {
      
      expr->second->setAttribute(i->first,i->second) ;
    }

    // In theory, this loop is not necessary, as all attributes in the
    // derivatives should be present in the utilities already..

    for (map<unsigned long, map<patString, patArithNode*> >::iterator iter =
	   derivatives.begin() ;
	 iter != derivatives.end() ;
	 ++iter) {
      for (map<patString, patArithNode*>::iterator expr =
	     iter->second.begin() ;
	   expr != iter->second.end() ;
	   ++expr) {
	expr->second->setAttribute(i->first,i->second) ;
      }
    }

    if (choiceExpr != NULL) {
      choiceExpr->setAttribute(i->first,i->second) ;
    }
    if (aggLastExpr != NULL) {
      aggLastExpr->setAttribute(i->first,i->second) ;
    }
    if (aggWeightExpr != NULL) {
      aggWeightExpr->setAttribute(i->first,i->second) ;
    }
    if (weightExpr != NULL) {
      weightExpr->setAttribute(i->first,i->second) ;
    }
    if (excludeExpr != NULL) {
      excludeExpr->setAttribute(i->first,i->second) ;
    }
    if (groupExpr != NULL) {
      groupExpr->setAttribute(i->first,i->second) ;
    }
  } 

  
  
  // register the variables
  for (map<unsigned long, patArithNode*>::iterator i = 
	 nonLinearUtilities.begin() ;
       i != nonLinearUtilities.end() ;
       ++i) {
    setVariablesBetaWithinExpression(i->second) ;
   
  }

  for (map<unsigned long, map<patString, patArithNode*> >::iterator iter =
	 derivatives.begin() ;
       iter != derivatives.end() ;
       ++iter) {
    for (map<patString, patArithNode*>::iterator expr =
	   iter->second.begin() ;
	 expr != iter->second.end() ;
	 ++expr) {
      setVariablesBetaWithinExpression(expr->second) ;
    }
  }
}
patBetaLikeParameter* patModelSpec::getCovarianceParameter(const patString& rv1,
							   const patString& rv2) {
  if (rv1 == rv2) {
    // It is a variance
    for (map<patString,pair<patRandomParameter,patArithRandom*> >::iterator i =
	   randomParameters.begin() ;
	 i != randomParameters.end() ;
	 ++i) {
      if (i->second.first.name == rv1) {
	return (i->second.first.scale) ;
      }
    }
    return NULL ;
  }
  else {

    pair<patString,patString> p1(rv1,rv2) ;
    map<pair<patString,patString>,patBetaLikeParameter*>::iterator found =
      covarParameters.find(p1) ;
    if (found != covarParameters.end()) {
      return found->second ;
    }
    pair<patString,patString> p2(rv2,rv1) ;
    found = covarParameters.find(p2) ;
    if (found != covarParameters.end()) {
      return found->second ;
    }
    return NULL ;
  }
}


patIterator<patString>* patModelSpec::createAttributeNamesIterator() {
  patIterator<patString>* ptr =
    new patAttributeNamesIterator(&attributeNames) ;
  return ptr ;
}

patIterator<pair<patString,unsigned long> >* patModelSpec::createUsedAttributeNamesIterator() {
  patIterator<pair<patString,unsigned long> >* ptr =
    new patUsedAttributeNamesIterator(&usedAttributes) ;
  return ptr ;
}

unsigned long patModelSpec::getNumberOfDraws() {
  return numberOfDraws ;
}

unsigned long patModelSpec::getAlgoNumberOfDraws() {
  return patMin(algoNumberOfDraws,numberOfDraws)  ;
}

void patModelSpec::setNumberOfDraws(unsigned long d) {
  algoNumberOfDraws = d ;
  numberOfDraws = d ;
}

void patModelSpec::setAlgoNumberOfDraws(unsigned long d) {
  algoNumberOfDraws = d ;
}


void patModelSpec::computeVarCovarOfRandomParameters(patError*& err) {

  // This routine computes the variance-covariance matrix as if all
  // random parameters are normally distributed. This is valid as
  // unifromly distrivuted random parameters are supposed to be
  // uncorrelated and, therefore, appear only on the diagonal. 
  // When displayed, those value will be adjusted. 
  // Indeed the variance of U[b-s,b+s] is s^2/3 and not s^2 as for the normal.

  // Number the random parameters

  vector<patRandomParameter*> vectorOfRandomParameters ;
  map<patString,unsigned long> numberFromRandomParameterName ;

  for (map<patString,pair<patRandomParameter,patArithRandom*> >::iterator i =
	 randomParameters.begin() ;
       i != randomParameters.end() ;
       ++i) {

    numberFromRandomParameterName[i->first] = vectorOfRandomParameters.size() ;
    vectorOfRandomParameters.push_back(&(i->second.first)) ;
  }

  unsigned long N = vectorOfRandomParameters.size() ;

  // Number of elements of var-covar matrix

  vector<pair<unsigned long,unsigned long> > vectorOfVarCovarEntries ;
  map<pair<unsigned long,unsigned long>,unsigned long> indexOfVarCovarEntries ;

  for (map<patString,pair<patRandomParameter,patArithRandom*> >::iterator i =
	 randomParameters.begin() ;
       i != randomParameters.end() ;
       ++i) {
    for (map<patString,pair<patRandomParameter,patArithRandom*> >::iterator j =
	   randomParameters.begin() ;
	 j != randomParameters.end() ;
	 ++j) {
      unsigned long index_i = numberFromRandomParameterName[i->first] ;
      unsigned long index_j = numberFromRandomParameterName[j->first] ;
      if (index_i <= index_j) {
	pair<unsigned long,unsigned long> thePair = pair<unsigned long,unsigned long>(index_i,index_j) ;
	indexOfVarCovarEntries[thePair] = vectorOfVarCovarEntries.size() ;
	vectorOfVarCovarEntries.push_back(thePair) ;
      }
    }
  }

  unsigned long K = vectorOfVarCovarEntries.size() ;

  // Number the estimated parameters associated with random coefficients

  vector<patBetaLikeParameter* > vectorOfEstimated ;
  map<patString,unsigned long> indexOfEstimated ;
  
  // Variances

  for (map<patString,pair<patRandomParameter,patArithRandom*> >::iterator i =
	 randomParameters.begin() ;
       i != randomParameters.end() ;
       ++i) {
    patBetaLikeParameter* variance = i->second.first.scale ;
    if (!variance->isFixed) {

      indexOfEstimated[variance->name] = vectorOfEstimated.size() ;
      vectorOfEstimated.push_back(variance) ;
    }
  }
  

  // Covariances

  for (map<pair<patString,patString>,patBetaLikeParameter*>::iterator i =
	 covarParameters.begin() ;
       i != covarParameters.end() ;
       ++i) {
    patBetaLikeParameter* correlation = i->second ;
    if (!correlation->isFixed) {
      indexOfEstimated[correlation->name] = vectorOfEstimated.size() ;
      vectorOfEstimated.push_back(correlation) ;
    }
  }

  unsigned long n = vectorOfEstimated.size() ;

  if (n == 0) {
    // All coefficients are fixed
    return ;
  }

  // Build gamma

  patMyMatrix gamma(N,N) ;

  for (map<patString,pair<patRandomParameter,patArithRandom*> >::iterator i =
	 randomParameters.begin() ;
       i != randomParameters.end() ;
       ++i) {
    
    unsigned long index = numberFromRandomParameterName[i->first] ;
    patReal stdDev = i->second.first.scale->estimated ;
    gamma[index][index] = stdDev ;

    for (vector<patRandomParameter*>::iterator j = 
	   i->second.first.correlatedParameters.begin() ;
	 j != i->second.first.correlatedParameters.end() ;
	 ++j) {
      unsigned long secondIndex = numberFromRandomParameterName[(*j)->name] ;
      patBetaLikeParameter* correl = getCovarianceParameter(i->first,(*j)->name) ;
      if (index >= secondIndex) {
	gamma[index][secondIndex] = correl->estimated ;
      }
      else {
	gamma[secondIndex][index] = correl->estimated ;
      }

    }

  }

//   DEBUG_MESSAGE("GAMMA") ;

//   print_all_matrix(gamma) ;

  patMyMatrix R(N,N) ;

  multABTransp(gamma,gamma,R,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
//   print_all_matrix(R) ;

  patMyMatrix G(K,n,0.0) ;


  for (unsigned long row = 0 ; row < N ; ++row) {
    
    patString rowParam = vectorOfRandomParameters[row]->name ;

    for (unsigned long col = 0 ; col <= row ; ++col) {

      patString colParam = vectorOfRandomParameters[col]->name ;
      // Look for index 
      pair<unsigned long, unsigned long> thePair(col,row) ;
      // Kindex is the row of the G matrix
      unsigned long Kindex = indexOfVarCovarEntries[thePair] ;

      for (unsigned long m = 0 ; m <= col ; ++ m) {
	
	patString mParam = vectorOfRandomParameters[m]->name ;

	patBetaLikeParameter* row_m = getCovarianceParameter(rowParam,mParam) ;
	if (row_m != NULL) {
	  if (!row_m->isFixed) {
	    // G(Kindex,param(row,m)) = gamma(col,m)
	    G[Kindex][indexOfEstimated[row_m->name]] += gamma[col][m] ; 
	  }
	}
	patBetaLikeParameter* col_m = getCovarianceParameter(colParam,mParam) ;
	if (col_m != NULL) {
	  if (!col_m->isFixed) {
	    // G(Kindex,param(col,m)) = gamma(row,m)
	    G[Kindex][indexOfEstimated[col_m->name]] += gamma[row][m] ; 
	  }
	}
      }
      
    }
  }

//   DEBUG_MESSAGE("G") ;

//   print_all_matrix(G) ;

  patMyMatrix V(n,n) ;

  for (unsigned long row = 0 ; row < n ; ++row) {
    for (unsigned long col = 0 ; col <= row ; ++ col) {

      unsigned long index_varCovar_row = vectorOfEstimated[row]->index;
      unsigned long index_varCovar_col = vectorOfEstimated[col]->index;
      
      V[row][col] = (*estimationResults.varCovarMatrix)[index_varCovar_row][index_varCovar_col] ;
      V[col][row] = (*estimationResults.varCovarMatrix)[index_varCovar_row][index_varCovar_col] ;
      
    }
  }

//   DEBUG_MESSAGE("V") ;

//   print_all_matrix(V) ;

  patMyMatrix tmp(K,n) ;

  mult(G,V,tmp,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  patMyMatrix Omega(K,K) ;

  multABTransp(tmp,G,Omega,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

//   DEBUG_MESSAGE("Omega") ;

//   print_all_matrix(Omega) ;

  // Fill in the results

  for (map<patString,pair<patRandomParameter,patArithRandom*> >::iterator i =
	 randomParameters.begin() ;
       i != randomParameters.end() ;
       ++i) {
    unsigned long Nindex = numberFromRandomParameterName[i->first] ;
    patReal var = R[Nindex][Nindex] ;
    patReal stdErr ;
    patBetaLikeParameter* stdDev = i->second.first.scale ;
    if (stdDev->isFixed) {
      relevantVariance[i->first] = patFALSE ;
    }
    else {
      relevantVariance[i->first] = patTRUE ;
      pair<unsigned long,unsigned long> thePair(Nindex,Nindex) ;
      unsigned long Kindex = indexOfVarCovarEntries[thePair] ;
      stdErr = sqrt(Omega[Kindex][Kindex]) ;
    }
    varianceRandomCoef[i->first] = pair<patReal,patReal>(var,stdErr) ;
  }

  for (map<pair<patString,patString>,patBetaLikeParameter*>::iterator i =
	 covarParameters.begin() ;
       i != covarParameters.end() ;
       ++i) {
    unsigned long NindexRow = numberFromRandomParameterName[i->first.first] ;
    unsigned long NindexCol = numberFromRandomParameterName[i->first.second] ;
    patReal covar = R[NindexRow][NindexCol] ;
    patReal stdErr ;
    if (i->second->isFixed) {
      relevantCovariance[i->first] = patFALSE ;
    }
    else {
      relevantCovariance[i->first] = patTRUE ;
      unsigned long Kindex ;
      if (NindexRow <= NindexCol) {
	pair<unsigned long,unsigned long> thePair(NindexRow,NindexCol) ;
	Kindex = indexOfVarCovarEntries[thePair] ;
      }
      else {
	pair<unsigned long,unsigned long> thePair(NindexCol,NindexRow) ;
	Kindex = indexOfVarCovarEntries[thePair] ;
      }
      stdErr = sqrt(Omega[Kindex][Kindex]) ;
    }

    covarianceRandomCoef[i->first] = pair<patReal,patReal>(covar,stdErr) ;
    
  }

}

void patModelSpec::readSummaryParameters(patError*& err) {
  patString fileName = patParameters::the()->getgevSummaryParameters() ;
  if (!patFileExists()(fileName)) return ;
  ifstream theFile(fileName.c_str()) ;
  while (!theFile.eof()) {
    patString name ;
    theFile >> name ;
    if (!name.empty()) {
      summaryParameters.push_back(name) ;
    }
  }
  theFile.close() ;
}

void patModelSpec::addExpressionLoop(const patString& name, 
				     patArithNode* expr,
				     patLoop* theLoop,
				     patError*& err) {


  
  for (long i = theLoop->lower ;
       i <= theLoop->upper ;
       i += theLoop->step) {

    stringstream freeVarStr ;
    freeVarStr << i ;
    patString freeVar(freeVarStr.str()) ;
    patString curName = name ;
    patArithNode* curExpr = expr->getDeepCopy(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    replaceAll (&curName,theLoop->variable,freeVar) ;
    curExpr->replaceInLiterals(theLoop->variable,freeVar) ;
    addExpression(curName,curExpr,err);
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
}

unsigned long patModelSpec::getUsedAttributeId(const patString attrName,
					      patBoolean* found) {
  assert(found != NULL) ;
  map<patString,unsigned long>::iterator f = usedAttributes.find(attrName) ;
  if (f == usedAttributes.end()) {
    *found = patFALSE ;
    return patBadId ;
  }
  *found = patTRUE ;
  return f->second ;
}

unsigned long patModelSpec::getIdSnpAttribute(patBoolean* found) {
  return getRandomAttributeId(snpBaseParameter,found) ;
}

patBoolean patModelSpec::applySnpTransform() {
  return !listOfSnpTerms.empty() ;
}

unsigned long patModelSpec::getRandomAttributeId(const patString attrName,
						patBoolean* found) {
  assert(found != NULL) ;

  map<patString,pair<patRandomParameter,patArithRandom*> >::iterator f = 
    randomParameters.find(attrName) ;
  if (f == randomParameters.end()) {
    *found = patFALSE ;
    return patBadId ;
  }
  *found = patTRUE ;
  return f->second.first.index ;
}

void patModelSpec::createExpressionForOne() {
  patString oneName = patParameters::the()->getgevOne() ;
  unsigned long id = getAttributeId(oneName) ;
  if (id == patBadId) {
    patArithConstant* oneExpr = new patArithConstant(NULL) ;
    oneExpr->setValue(1.0) ;
    patError* err(NULL) ;
    addExpression(oneName,oneExpr,err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return ;
    }
  }
  
}

unsigned long patModelSpec::getNbrDrawAttributesPerObservation() {
  return nbrDrawAttributesPerObservation ;
}


patBetaLikeParameter patModelSpec::getParameterFromId(unsigned long id,
						      patBoolean* found) {
  patIterator<patBetaLikeParameter>* iter = createAllParametersIterator() ;
  for (iter->first() ; !iter->isDone() ; iter->next()) {
    patBetaLikeParameter beta = iter->currentItem() ;
    if (beta.id == id) {
      *found = patTRUE ;
      return beta ;
    }
  }
  
  *found = patFALSE ;
  return patBetaLikeParameter() ;

  
}
patBetaLikeParameter patModelSpec::getParameterFromIndex(unsigned long index,
							 patBoolean* found) {
  patIterator<patBetaLikeParameter>* iter = createAllParametersIterator() ;
  for (iter->first() ; !iter->isDone() ; iter->next()) {
    patBetaLikeParameter beta = iter->currentItem() ;
    if (beta.index == index) {
      *found = patTRUE ;
      return beta ;
    }
  }
  
  *found = patFALSE ;
  return patBetaLikeParameter() ;

}


/**
   The non-random term first, followed by the random ones.
 */				     
void  patModelSpec::buildLinearRandomUtilities(patError*& err) {
  for (map<patString, patAlternative>::iterator i = utilities.begin() ;
       i != utilities.end() ;
       ++i) {
    patAlternative* alt = &(i->second) ;
    patUtilFunction newUtility ;
    patUtilFunction randomPart ;
    for (patUtilFunction::iterator term = alt->utilityFunction.begin() ;
	 term != alt->utilityFunction.end() ;
	 ++term) {
      if (term->randomParameter != NULL) {
	patUtilFunction* rndExpr = 
	  term->randomParameter->getLinearExpression() ;
	for (patUtilFunction::iterator rndTerm = rndExpr->begin() ;
	     rndTerm != rndExpr->end() ;
	     ++rndTerm) {
	  patUtilTerm newTerm ;
	  if (rndTerm == rndExpr->begin()) {
	    // The first term is the mean. Does not involve draws
	    newTerm.beta = rndTerm->beta ;
	    newTerm.betaIndex = rndTerm->betaIndex ;
	    newTerm.x = term->x ;
	    newTerm.xIndex = term->xIndex ;
	    newTerm.random = patFALSE ;
	    newTerm.rndIndex = patBadId ;
	    newTerm.massAtZero = term->massAtZero ;
	    newUtility.push_back(newTerm) ;
	  }
	  else {
	    newTerm.beta = rndTerm->beta ;
	    newTerm.betaIndex = rndTerm->betaIndex ;
	    newTerm.x = term->x ;
	    newTerm.xIndex = term->xIndex ;
	    newTerm.rndIndex = rndTerm->xIndex ;
	    newTerm.random = patTRUE ;
	    newTerm.massAtZero = term->massAtZero ;
	    randomPart.push_back(newTerm) ;
	  }
	}
      }
      else {
	term->random = patFALSE ;
	newUtility.push_back(*term) ;
      }
    }
    for (patUtilFunction::iterator term = randomPart.begin() ;
	 term != randomPart.end() ;
	 ++term) {
      newUtility.push_back(*term) ;
    }
    alt->utilityFunction = newUtility ;
  }
}

void patModelSpec::identifyScales(patVariables* attributeScales,
				  patError*& err) {

  DEBUG_MESSAGE("Identify scales");
  if (attributeScales == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return ;
  }

  automaticParametersScales.resize(getNbrOrigBeta(),1.0) ;
  automaticAttributesScales.resize(getNbrUsedAttributes(),1.0) ;

  if (nonLinearUtilities.size() > 0) {
    err = new patErrMiscError("No automatic scaling is possible for nonlinear utilities") ;
    WARNING(err->describe()) ;
    return ;
  }

  patBoolean found;
  fill(automaticParametersScales.begin(), automaticParametersScales.end(), 1.0) ;
  // First pass

  for (map<patString, patAlternative>::iterator i = utilities.begin() ;
       i != utilities.end() ;
       ++i) {

    for (patUtilFunction::iterator term = i->second.utilityFunction.begin() ;
	 term != i->second.utilityFunction.end() ;
	 ++term) {
      if (term->xIndex > attributeScales->size()) {
	err = new patErrOutOfRange<unsigned long>(term->xIndex,0,attributeScales->size()-1) ;
	WARNING(err->describe()) ;
	return ;
      }
      patBetaLikeParameter beta = getParameter(term->beta,&found) ;
      if (!found) {
	stringstream str ;
	str << "Unknown parameter " << term->beta ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return ;
      }
      if (beta.id >= automaticParametersScales.size()) {
	err = new patErrOutOfRange<unsigned long>(beta.id,
						 0,
						 automaticParametersScales.size()-1) ;
	
	WARNING(err->describe()) ;
	return ;
      }
	automaticParametersScales[beta.id] = 
	  patMax((*attributeScales)[term->xIndex],
		 automaticParametersScales[beta.id]) ;
    }
  }
  // Second pass

  for (map<patString, patAlternative>::iterator i = utilities.begin() ;
       i != utilities.end() ;
       ++i) {

    for (patUtilFunction::iterator term = i->second.utilityFunction.begin() ;
	 term != i->second.utilityFunction.end() ;
	 ++term) {
      if (term->xIndex > attributeScales->size()) {
	err = new patErrOutOfRange<unsigned long>(term->xIndex,0,attributeScales->size()-1) ;
	WARNING(err->describe()) ;
	return ;
      }
      patBetaLikeParameter beta = getParameter(term->beta,&found) ;
      if (!found) {
	stringstream str ;
	str << "Unknown parameter " << term->beta ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return ;
      }
      if (beta.id >= automaticParametersScales.size()) {
	err = new patErrOutOfRange<unsigned long>(beta.id,
						 0,
						 automaticParametersScales.size()-1) ;
	
	WARNING(err->describe()) ;
	return ;
      }
      automaticAttributesScales[term->xIndex] =
	automaticParametersScales[beta.id] ;
    }
  }
}

void patModelSpec::scaleBetaParameters() {
  // Scale the initial value of betas
  
  if (automaticScaling) {
    for (map<patString,patBetaLikeParameter>::iterator i = betaParam.begin() ;
	 i != betaParam.end() ;
	 ++i) {
      DEBUG_MESSAGE("Scale " << i->second.name 
		    << " by " << automaticParametersScales[i->second.id]) ;
      i->second.defaultValue *= automaticParametersScales[i->second.id] ;
      i->second.lowerBound *= automaticParametersScales[i->second.id] ;
      i->second.upperBound *= automaticParametersScales[i->second.id] ;
      i->second.estimated *= automaticParametersScales[i->second.id] ;
    }
  }
}

void patModelSpec::unscaleBetaParameters() {
  // Scale the initial value of betas
  
  if (automaticScaling) {
    for (map<patString,patBetaLikeParameter>::iterator i = betaParam.begin() ;
	 i != betaParam.end() ;
	 ++i) {
      DEBUG_MESSAGE("Scale " << i->second.name << " by " << automaticParametersScales[i->second.id]) ;
      i->second.defaultValue /= automaticParametersScales[i->second.id] ;
      i->second.lowerBound /= automaticParametersScales[i->second.id] ;
      i->second.upperBound /= automaticParametersScales[i->second.id] ;
      i->second.estimated /= automaticParametersScales[i->second.id] ;
    }
  }
}

void patModelSpec::unscaleMatrix(patMyMatrix* matrix, patError*& err) {
  if (matrix == NULL) {
    err = new patErrNullPointer("patMyMatrix") ;
  }

  // WARNING: Scales affect only the part of the matrix assocaited to
  // the parameters. The rest of the matrix should also be
  // affected. It need some more work...
  
  for (map<patString,patBetaLikeParameter>::iterator i = betaParam.begin() ;
       i != betaParam.end() ;
       ++i) {
    if (!i->second.isFixed) {
      for (map<patString,patBetaLikeParameter>::iterator j = betaParam.begin() ;
	   j != betaParam.end() ;
	   ++j) {
	if (!j->second.isFixed) {
	  (*matrix)[i->second.index][j->second.index] /= 
	    automaticParametersScales[i->second.id] * 
	    automaticParametersScales[j->second.id] ;
	}
      }
    }
  }
}

patVariables* patModelSpec::getAttributesScale() {
  return &automaticAttributesScales ;
}

patDistribType patModelSpec::getDistributionOfDraw(unsigned long i, 
						   patError*& err) {
  if (i >= typeOfDraws.size()) {
    err = new patErrOutOfRange<unsigned long>(i,0, typeOfDraws.size()-1) ;
    WARNING(err->describe()) ;
    return UNDEFINED_DIST ;
  }
  return typeOfDraws[i] ;
}

patBoolean patModelSpec::isDrawPanel(unsigned long i, patError*& err) {
  if (i >= areDrawsPanel.size()) {
    err = new patErrOutOfRange<unsigned long>(i,0, areDrawsPanel.size()-1) ;
    WARNING(err->describe()) ;
    return patFALSE ;
  }
  return areDrawsPanel[i] ;
  
}

patReal patModelSpec::getMassAtZero(unsigned long i, patError*& err) {
  if (i >= massAtZero.size()) {
    err = new patErrOutOfRange<unsigned long>(i,0, massAtZero.size()-1) ;
    WARNING(err->describe()) ;
    return patFALSE ;
  }
  return massAtZero[i] ;
}


void patModelSpec::setPanelVariables(list<patString>* panelVar) {
  if (panelVar == NULL) {
    WARNING("No random parameter capturing panel effect have been selected") ;
    return ;
  }
  panelVariables = *panelVar ;
}

patBoolean patModelSpec::isPanelData() const {
  return (panelExpr != NULL) ;
}


patBoolean patModelSpec::checkAllBetaUsed(patError*& err) {

 map<patString,patBoolean> checkBetas ;

  // Loop on linear utility functions

  for (map<patString, patAlternative>::const_iterator i = utilities.begin() ;
       i != utilities.end();
       ++i) {
    for (patUtilFunction::const_iterator term = i->second.utilityFunction.begin() ;
	 term != i->second.utilityFunction.end() ;
	 ++term) {
      checkBetas[term->beta] = patTRUE ;
    }
  }
  
  // Loop on nonlinear utility functions

  vector<patString>  literals ;

  for (map<unsigned long, patArithNode*>::const_iterator i = 
	 nonLinearUtilities.begin() ;
       i != nonLinearUtilities.end() ;
       ++i) {
    i->second->getLiterals(&literals,NULL,patTRUE,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE ;
    }
  }

  //  DEBUG_MESSAGE("Literals identified in utilities") ;
  for (vector<patString>::iterator lit = literals.begin() ;
       lit != literals.end() ;
       ++lit) {
    //    DEBUG_MESSAGE(*lit) ;
    checkBetas[*lit] = patTRUE ;
  }

  for (map<pair<patString,patString>,patBetaLikeParameter*>::const_iterator 
	 covPar = covarParameters.begin() ;
       covPar != covarParameters.end() ;
       ++covPar) {
    DEBUG_MESSAGE(covPar->second->name) ;
    checkBetas[covPar->second->name] = patTRUE ;
  }

  for (vector<patDiscreteParameter>::const_iterator i = discreteParameters.begin() ;
       i != discreteParameters.end() ;
       ++i) {

    map<patString,patBoolean>::iterator found = checkBetas.find(i->name) ;
    if (found != checkBetas.end()) {
      for (vector<patDiscreteTerm>::const_iterator t =
	     i->listOfTerms.begin() ;
	   t != i->listOfTerms.end() ;
	   ++t) {
	checkBetas[t->massPoint->name] = patTRUE ;
	checkBetas[t->probability->name] = patTRUE ;
      }
    }
  }

  for (map<unsigned long, patString>::const_iterator i = 
	 selectionBiasParameters.begin() ;
       i != selectionBiasParameters.end() ;
       ++i) {
    checkBetas[i->second] = patTRUE ;
  }

  for (map<unsigned long, patString>::const_iterator i =
	 ordinalLogitThresholds.begin() ;
       i != ordinalLogitThresholds.end() ;
       ++i) {
    checkBetas[i->second] = patTRUE ;
  }

  for (list<pair<unsigned short,patString> >::const_iterator i =
	 listOfSnpTerms.begin() ;
       i != listOfSnpTerms.end() ;
       ++i) {
    checkBetas[i->second] = patTRUE ;
  }
  
  literals.erase(literals.begin(),literals.end());

  if (equalityConstraints != NULL) {
    for (patListNonLinearConstraints::iterator i = equalityConstraints->begin() ;
	 i != equalityConstraints->end() ;
	 ++i) {
      
      i->getLiterals(&literals,NULL,patTRUE,err) ;
      
    }
    
    for (vector<patString>::iterator lit = literals.begin() ;
	 lit != literals.end() ;
	 ++lit) {
      //    DEBUG_MESSAGE(*lit) ;
      checkBetas[*lit] = patTRUE ;
    }
  }

  if (generalizedExtremeValueParameter != NULL) {
    checkBetas[generalizedExtremeValueParameter->name] = patTRUE ;
  }


  list<patString> unusedBetas ;
  patIterator<patBetaLikeParameter>* betaIter = createBetaIterator() ;
  for (betaIter->first() ;
       !betaIter->isDone() ;
       betaIter->next()) {
    if (!checkBetas[betaIter->currentItem().name] && !betaIter->currentItem().isFixed) {
      unusedBetas.push_back(betaIter->currentItem().name) ;
    }
  }


  if (!unusedBetas.empty()) {
    stringstream str ;
    str << "Unused parameter(s): " ;
    for (list<patString>::iterator i = unusedBetas.begin() ;
	 i != unusedBetas.end() ;
	 ++i) {
      if (i != unusedBetas.begin()) {
	str << "," ;
      }
      str << *i ;
    }
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patFALSE;
  }
  return patTRUE ;
}

void patModelSpec::addDiscreteParameter(const patString& paramName,
					const vector<patThreeStrings>& listOfTerms,
					patError*& err) {

  // Add a fake beta parameters which will play a role in the
  // computation of the utility


  patDiscreteParameter theParam ;
  theParam.name = paramName ;
  patBoolean isFixed = patTRUE ;

  for (vector<patThreeStrings>::const_iterator i =
	 listOfTerms.begin() ;
       i != listOfTerms.end() ;
       ++i) {
    patDiscreteTerm theTerm ;
    if (i->s2.empty()) {
      // Deterministic parameter
      patString nameBeta = i->s1 ;
      map<patString,patBetaLikeParameter>::iterator found = betaParam.find(nameBeta) ;
      if (found != betaParam.end()) {
	theTerm.random = patFALSE;
	theTerm.massPoint = &(found->second) ;
        theTerm.massPointRandom = NULL ;
	if (!theTerm.massPoint->isFixed) {
	  isFixed = patFALSE ;
	}
      }
      else {
	stringstream str ;
	str << "Parameters " << nameBeta << " used in the definition of " << paramName << " has not been defined" ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return ;
      }
    }
    else { // Random parameter
      err = new patErrMiscError("Biogeme does not allow discrete distributions to involve random parameters") ;
      WARNING(err->describe()) ;
      return ;

      patString nameLocation = i->s1 ;
      patString nameScale = i->s2 ;
      patString nameRanParam = buildRandomName(nameLocation,nameScale) ;
      map<patString,pair<patRandomParameter,patArithRandom*> >::iterator found =
	randomParameters.find(nameRanParam) ;
      if (found != randomParameters.end()) {
	theTerm.random = patTRUE ;
	theTerm.massPoint = NULL ;
	theTerm.massPointRandom = found->second.second ;
      }
      else {
	stringstream str ;
	str << "Random parameter " << nameRanParam << " used in the definition of " << paramName << " has not been defined" ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return ;

      }
    }
      
    patString nameWeight = i->s3 ;
    map<patString,patBetaLikeParameter>::iterator found = betaParam.find(nameWeight) ;
    if (found != betaParam.end()) {
      theTerm.probability = &(found->second) ;
    }
    else {
      stringstream str ;
      str << "Parameter " << nameWeight << " used in the definition of " << paramName << " has not been defined" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
    theParam.listOfTerms.push_back(theTerm) ;
  }

  patBetaLikeParameter b ;
  b.name = paramName ;
  b.hasDiscreteDistribution = patTRUE ;
  b.index = patBadId ;
  b.id = discreteParameters.size() ;
  b.isFixed = isFixed ;
  betaParam[paramName] = b ;


  discreteParameters.push_back(theParam) ;
  DEBUG_MESSAGE("Added: " << paramName) ;
  DEBUG_MESSAGE("Number = " << discreteParameters.size()) ;

}

patIterator<patDiscreteParameter*>* patModelSpec::getDiscreteParametersIterator() {
  patIterator<patDiscreteParameter*>* ptr = new patDiscreteParameterIterator(&discreteParameters) ;
  return ptr ;
}

void patModelSpec::setDiscreteParameterValue(patString param,
					     patBetaLikeParameter* beta,
					     patError*& err) {

  if (beta == NULL){
    err = new patErrNullPointer("patBetaLikeParameter") ;
    WARNING(err->describe()) ;
    return ;
  }

  map<patString,patBetaLikeParameter>::iterator found =  
    betaParam.find(param) ;
  if (found == betaParam.end()) {
    stringstream str ;
    str << "Beta parameter " << param << " not found" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return  ;
  }

//    DEBUG_MESSAGE("DISCRETE PARAMETER " << found->second) ;
//    DEBUG_MESSAGE("BETA PARAMETER " << *beta) ;
  
//   DEBUG_MESSAGE("Assign beta("<< beta->id << ") to beta( " << found->second.id << ")") ;
  betaParameters[found->second.id] = betaParameters[beta->id] ;
  

  found->second.estimated = beta->estimated ;
  found->second.isFixed = beta->isFixed ;


}


patBoolean patModelSpec::containsDiscreteParameters() const {
  return !discreteParameters.empty() ;
}

patString patModelSpec::getDiscreteParamName(unsigned long i, patError*& err) {
  if (i >= discreteParameters.size()) {
    err = new patErrOutOfRange<unsigned long>(i,0,discreteParameters.size()-1) ;
    WARNING(err->describe());
    return patString() ;
  }
  return discreteParameters[i].name ;
}

void patModelSpec::generateFileForDenis(patString fileName) {
  ofstream denis(fileName.c_str()) ;
  patIterator<patBetaLikeParameter>* iter = createAllParametersIterator() ;

  patBoolean theFirst = patTRUE ;
  for (iter->first() ;
       !iter->isDone() ;
       iter->next()) {
    patBetaLikeParameter beta = iter->currentItem() ;
    if (theFirst) {
      theFirst = patFALSE ;
    }
    else {
      denis << '\t' ;
    }
    denis << beta.name ;
  }
  denis << endl ;
  theFirst = patTRUE ;
  for (iter->first() ;
       !iter->isDone() ;
       iter->next()) {
    patBetaLikeParameter beta = iter->currentItem() ;
    if (theFirst) {
      theFirst = patFALSE ;
    }
    else {
      denis << '\t' ;
    }
    denis << beta.estimated ;
  }
  denis << endl ;
  denis.close() ;
  patOutputFiles::the()->addDebugFile(fileName,"List of estimated values of the parameters");
}


list<patString> patModelSpec::getMultiLineModelDescription() {
  return modelDescription ;
}

vector<patString> patModelSpec::getHeaders() {
  return headers ;
}

void patModelSpec::addSelectionBiasParameter(const unsigned long alt,
					     const patString& paramName) {
  selectionBiasParameters[alt] = paramName ;
}

patBoolean patModelSpec::correctForSelectionBias() {
  if (selectionBiasParameters.empty()) {
    return patFALSE ;
  }
  if (isMNL()) {
    return patFALSE ;
  }
  if (isMixedLogit()) {
    return patFALSE ;
  }
  return patTRUE ;
}

void patModelSpec::setSnpBaseParameter(const patString& n) {
  snpBaseParameter = n ;
}

void patModelSpec::addSnpTerm(unsigned short i, const patString& n) {
  listOfSnpTerms.push_back(pair<unsigned short,patString>(i,n)) ;
}

unsigned short patModelSpec::numberOfSnpTerms() {
  return listOfSnpTerms.size() ;
}

unsigned short patModelSpec::orderOfSnpTerm(unsigned short i, patError*& err) {
  if (i >= ordersOfSnpTerms.size()) {
    err = new patErrOutOfRange<unsigned short>(i,0,ordersOfSnpTerms.size()-1) ;
    WARNING(err->describe()) ;
    return patShortBadId ;
  }
  return ordersOfSnpTerms[i] ;
  
}

vector<unsigned long> patModelSpec::getIdOfSnpBetaParameters() {
  return idOfSnpBetaParameters ;
}

patBoolean patModelSpec::isIndividualSpecific(const patString& rndParam) {
  DEBUG_MESSAGE("Check if " << rndParam << " is individual specific") ;
  list<patString>::iterator found ;
  found = find(panelVariables.begin(),panelVariables.end(),rndParam) ;
  if (found != panelVariables.end()) {
    DEBUG_MESSAGE(*found << " is panel") ;
    return patTRUE ;
  }
  else {
    DEBUG_MESSAGE(rndParam << " is not panel") ;
    return patFALSE ;
  }
}


patBoolean patModelSpec::utilDerivativesAvailableFromUser() {
  return !derivatives.empty() ;
}

void patModelSpec::setGnuplot(patString x, patReal l, patReal u) {
  gnuplotDefined = patTRUE ;
  gnuplotParam = x ;
  gnuplotMin = l ;
  gnuplotMax = u ;
}


patBoolean patModelSpec::isSimpleMnlModel() {
  if (!isMNL()) {
    //    DEBUG_MESSAGE("Not a MNL model") ;
    return patFALSE ;
  }
  if (isMixedLogit()) {
    //    DEBUG_MESSAGE("Mixed logit") ;
    return patFALSE ;
  }
  if (isPanelData()) {
    //    DEBUG_MESSAGE("Panel data") ;
    return patFALSE ;
  }
  if (applySnpTransform()) {
    //    DEBUG_MESSAGE("SNP transforms") ;
    return patFALSE ;
  }
  if (isAggregateObserved()) {
    //    DEBUG_MESSAGE("Aggregate observations") ;
    return patFALSE ;
  }
  if (!isMuFixed()) {
    //    DEBUG_MESSAGE("Mu is estimated") ;
    return patFALSE ;
  }
  patError* localErr = NULL ;
  patUtility* util = getFullUtilityFunction(localErr) ;
  if (util == NULL) {
    //    DEBUG_MESSAGE("Full utility is not available") ;
    return patFALSE ;
  }
  if (!util->isLinear()) {
    //    DEBUG_MESSAGE("Utility is not linear") ;
    return patFALSE ;
  }
  if (estimateGroupScales()) {
    //    DEBUG_MESSAGE("Scales are estimated") ;
    return patFALSE ;
  }
  return patTRUE ;
}

patBoolean patModelSpec::estimateGroupScales() {
  static patBoolean first = patTRUE ;
  if (first) {
    estimGroupScale = patFALSE ;
    for (map<patString,patBetaLikeParameter>::iterator i = scaleParam.begin() ;
	 i != scaleParam.end() ;
	 ++i) {
      if (!i->second.isFixed) {
	estimGroupScale = patTRUE ;
	return patTRUE ;
      }
    }
    first = patFALSE ;
    return patFALSE ;
  }
  else {
    return estimGroupScale ;
  }
}

patBoolean patModelSpec::groupScalesAreAllOne() {
  for (map<patString,patBetaLikeParameter>::iterator i = scaleParam.begin() ;
       i != scaleParam.end() ;
       ++i) {
    if (!i->second.isFixed) {
      return patFALSE ;
    }
    if (!(i->second.defaultValue == 1.0)) {
      return patFALSE ;
    }
  }
  return patTRUE ;
}


void patModelSpec::addLatexName(const patString& coeffName,
				const patString& latexName) {
  latexNames[coeffName] = latexName ;
  
}

patString patModelSpec::getLatexName(const patBetaLikeParameter* betaParam) {
  if (betaParam == NULL) {
    return patString() ;
  }
  map<patString,patString>::const_iterator found = latexNames.find(betaParam->name) ;
  if (found == latexNames.end()) {
    patString name = betaParam->name ;
    replaceAll(&name,patString("_"),patString("TO+BE+REPLACED")) ;
    replaceAll(&name,patString("TO+BE+REPLACED"),patString("\\_")) ;
    replaceAll(&name,patString("$"),patString("TO+BE+REPLACED")) ;
    replaceAll(&name,patString("TO+BE+REPLACED"),patString("\\$")) ;
    return(name) ;
  }
  else {
    return found->second ;
  }
}

patString patModelSpec::generateLatexRow(patBetaLikeParameter beta,
					 const patVariables& stdErr,
					 const patVariables& robustStdErr,
					 patReal ttestBasis) {
  stringstream latexFile ;
  latexFile << getLatexName(&(beta)) << " & " ;
  if (!beta.hasDiscreteDistribution) {
    if (beta.isFixed) {
       latexFile << "\\multicolumn{6}{l}{fixed} \\\\" << endl  ;
     }
     else {
       patString anumber = theNumber.formatParameters(beta.estimated) ;
       replaceAll(&anumber,patString("."),patString("&")) ;
       latexFile << anumber << " & "  ;
       if (estimationResults.isRobustVarCovarAvailable) {
	 patReal rttest = (beta.estimated - ttestBasis)  / robustStdErr[beta.index] ; 
	 patString anumber = theNumber.formatParameters(robustStdErr[beta.index]) ;
	 replaceAll(&anumber,patString("."),patString("&")) ;
	 latexFile << anumber << " & " ;
	 anumber = theNumber.formatTTests(rttest) ;
	 replaceAll(&anumber,patString("."),patString("&")) ;
	 latexFile << anumber ;
	 if (ttestBasis != 0.0) {
	   latexFile << "\\footnotemark[1]" ;
	 }
	 if (patParameters::the()->getgevPrintPValue()) {
	   latexFile << " & " ;
	   patError* err = NULL;
	   patReal pvalue = patPValue(patAbs(rttest),err) ;
	   if (err != NULL) {
	     WARNING(err->describe()) ;
	     return patString("Error in generating output") ;
	   }
	   anumber= theNumber.formatTTests(pvalue) ;
	   replaceAll(&anumber,patString("."),patString("&")) ;
	   latexFile << anumber ;
	  }
	 
	 latexFile << " \\\\" << endl ;
       } 
       else if (estimationResults.isVarCovarAvailable) {
	 patReal ttest = beta.estimated  / stdErr[beta.index] ; 
	 patString anumber = theNumber.formatParameters(stdErr[beta.index]) ;
	 replaceAll(&anumber,patString("."),patString("&")) ;
	 latexFile << anumber << " & "  ;
	 anumber =  theNumber.formatTTests(ttest) ;
	 replaceAll(&anumber,patString("."),patString("&")) ;
	 latexFile << anumber ;
	 if (ttestBasis != 0.0) {
	   latexFile << "\\footnotemark[1]" ;
	 }
	 if (patParameters::the()->getgevPrintPValue()) {
	   latexFile << " & " ;
	   patError* err = NULL ;
	   patReal pvalue = patPValue(patAbs(ttest),err) ;
	   if (err != NULL) {
	     WARNING(err->describe()) ;
	     return patString("Error in generating output") ;
	   }
	   anumber= theNumber.formatTTests(pvalue) ;
	   replaceAll(&anumber,patString("."),patString("&")) ;
	   latexFile << anumber  ;
	  }
	 latexFile << " \\\\" << endl ;
       }
       else {
	 latexFile << "\\multicolumn{6}{l}{var-covar unavailable} \\\\"<< endl ;
       }
     }
   }
  else {
    latexFile << "\\multicolumn{6}{l}{distributed} \\\\" << endl ;
  }
  return patString(latexFile.str()) ;
}

unsigned long patModelSpec::getOrdinalLogitLeftAlternative() {
  return ordinalLogitLeftAlternative ;
}

map<unsigned long, patBetaLikeParameter*>* patModelSpec::getOrdinalLogitThresholds() {
  return &ordinalLogitBetaThresholds ;
}

patULong patModelSpec::getOrdinalLogitNumberOfIntervals() {
  return ordinalLogitBetaThresholds.size() + 1 ;
}

patBoolean patModelSpec::isMuFixed() const {
  return mu.isFixed ;
}

void patModelSpec::saveBackup() {
  if (patParameters::the()->getgevSaveIntermediateResults()) {
    patError* err = NULL ;
    patString file = patFileNames::the()->getBckFile(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      file = patString("___defaultBckFile.bck") ;
    }
    patModelSpec::the()->writeSpecFile(file,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
    }
    GENERAL_MESSAGE("*** SAVE INTERMEDIATE RESULTS ON FILE " << file) ;
  }
}

void patModelSpec::setRegressionObservation(const patString& name) {
  regressionObservation = name ;
  gianlucaObservationId = patBadId ;
  useModelForGianluca = patTRUE ;
}

void patModelSpec::setStartingTime(const patString& name) {
  fixationStartingTime = name ;
  startingTimeId = patBadId ;
  useModelForGianluca = patTRUE ;
}


void patModelSpec::addAcqRegressionModel(const patUtilFunction* f) {
  assert(f != NULL) ;
  acquisitionModel = *f ;
  useModelForGianluca = patTRUE ;

  for (patUtilFunction::const_iterator i = f->begin() ;
       i != f->end() ;
       ++i) {
    usedAttributes[i->x] = patBadId ;
  }

}

void patModelSpec::addValRegressionModel(const patUtilFunction* f) {
  assert(f != NULL) ;  
  validationModel = *f ;
  useModelForGianluca = patTRUE ;
  for (patUtilFunction::const_iterator i = f->begin() ;
       i != f->end() ;
       ++i) {
    usedAttributes[i->x] = patBadId ;
  }
}


void patModelSpec::setDurationParameter(const patString& p1) {
  durationNameParam = p1 ;
}

void patModelSpec::setAcqSigma(const patString& a) {
  sigmaAcqName = a ;
}

void patModelSpec::setValSigma(const patString& a) {
  sigmaValName = a ;
}

patBoolean patModelSpec::isGianluca() const {
  return useModelForGianluca ;
}

void patModelSpec::addZhengFosgerau(patOneZhengFosgerau zf, patError*& err) {
  if (!zf.isProbability()) {
    if (err != NULL) {
      WARNING(err->describe()) ;
    }
  }
  zhengFosgerau.push_back(zf) ;
}

unsigned short patModelSpec::numberOfZhengFosgerau(unsigned short* nbrProba)  {
  if (nbrProba == NULL) {
    return zhengFosgerau.size() ;
  }
  if (zhengFosgerau.size() == 0) {
    *nbrProba = 0 ;
    return 0 ;
  }
  if (numberOfProbaInZf == 0) {
    patULong expId = 0 ;
    for (vector<patOneZhengFosgerau>::iterator i = zhengFosgerau.begin() ;
	 i != zhengFosgerau.end() ;
	 ++i) {
      if (i->isProbability()) {
	++numberOfProbaInZf ;
      }
      else {
	i->expressionIndex = expId ;
	++expId ;
      }
    }
  }
  *nbrProba = numberOfProbaInZf ;
  return zhengFosgerau.size() ;
}

void patModelSpec::computeZhengFosgerau(patPythonReal** arrayResult,
					unsigned long resRow,
					unsigned long resCol,
					patSampleEnuGetIndices* ei,
					patError*& err) {

  if (ei == NULL) {
    err = new patErrNullPointer("patSampleEnuGetIndices") ;
    WARNING(err->describe());
    return ;
  }


  
//   DEBUG_MESSAGE("COMPUTE ZHENG-FOSGERAU:" << zhengFosgerau.size()) ;

//   DEBUG_MESSAGE("Data: " << resRow << " x " << resCol) ;


//   unsigned long nAlt = getNbrAlternatives() ;
//   for (list<patOneZhengFosgerau>::iterator t = zhengFosgerau.begin() ;
//        t != zhengFosgerau.end() ;
//        ++t) {
//     DEBUG_MESSAGE(t->describeVariable()) ;
//     if (t->isProbability()) {
//       patString altName = t->getAltName() ;
//       unsigned long altId = getAltInternalId(altName) ;
//       unsigned long indexT = ei->getIndexProba(altId,err) ;
//       if (err != NULL) {
// 	WARNING(err->describe());
// 	return ;
//       }
//       patVariables variableToTest ;
//       patVariables normalizedVariableToTest ;
//       patVariables probabilities ;
//       vector<patBoolean> trim(resRow,patFALSE) ;
//       // Normalize t
//       patReal largest ;
//       patReal smallest ;
//       for (patULong row = 0 ; row < resRow ; ++row) {
// 	patReal current = arrayResult[row][indexT] ;
// 	if (row == 0) {
// 	  largest = smallest = current ;
// 	}
// 	else {
// 	  if (current > largest) {
// 	    largest = current ;
// 	  }
// 	  if (current < smallest) {
// 	    smallest = current ;
// 	  }
// 	}
//       }
//       // Populate the vectors
      
//       patReal range = largest - smallest ;

//       DEBUG_MESSAGE("Range = " << range) ;
//       if (range >= patEPSILON) {
// 	patReal thisBw = t->bandwidth * range / sqrt(patReal(resRow)) ;
// 	DEBUG_MESSAGE(t->bandwidth << "*" <<  range << "/ sqrt(" << patReal(resRow) << ") = " << thisBw ) ;
// 	t->resetTrimCounter() ;
// 	for (patULong row = 0 ; row < resRow ; ++row) {
// 	  patReal current = (arrayResult[row][indexT] - smallest) / range ;
// 	  if (!t->trim(current)) {
// 	    normalizedVariableToTest.push_back(current) ;
// 	    variableToTest.push_back(arrayResult[row][indexT]) ;
// 	  }
// 	  else {
// 	    trim[row] = patTRUE ;
// 	  }
// 	}
// 	GENERAL_MESSAGE("Trimming: " << t->describeTrimming()) ;
// 	DEBUG_MESSAGE("Size of t:" << variableToTest.size()) ;
	
	
// 	for(unsigned long alt = 0 ; alt < nAlt ; ++alt) {
// 	  unsigned long indexResid = ei->getIndexResid(alt,err) ;
// 	  unsigned long indexProba = ei->getIndexProba(alt,err) ;
// 	  if (err != NULL) {
// 	    WARNING(err->describe());
// 	    return ;
// 	  }
// 	  // Compute Zheng test for current t and alternative alt
// 	  patVariables residual ;
// 	  patVariables probabilities ;
// 	  // Populate the vectors
// 	  for (patULong row = 0 ; row < resRow ; ++row) {
// 	    patReal current = (arrayResult[row][indexT] - smallest) / range ;
// 	    if (!trim[row]) {
// 	      residual.push_back(arrayResult[row][indexResid]) ;
// 	      probabilities.push_back(arrayResult[row][indexProba]) ;
// 	    }
// 	  }
// 	  //	  DEBUG_MESSAGE("residual=" << residual) ;
// 	  patZhengTest theTest(&variableToTest,&residual,thisBw,err) ;
// 	  if (err != NULL) {
// 	    WARNING(err->describe()) ;
// 	    return ;
// 	  }
	  
// 	  patReal aTest = theTest.compute(err) ;
// 	  DEBUG_MESSAGE("test = " << t->theTest) ;
// 	  if (err != NULL) {
// 	    WARNING(err->describe()) ;
// 	    return ;
// 	  }
// 	  t->setTest(alt,aTest) ;
	  

// 	  patNonParamRegression npReg(&residual, 
// 				      &normalizedVariableToTest, 
// 				      &probabilities,
// 				      thisBw,
// 				      err) ;

// 	  if (err != NULL) {
// 	    WARNING(err->describe()) ;
// 	    return ;
// 	  }
	  
// 	  npReg.compute(err) ;
// 	  if (err != NULL) {
// 	    WARNING(err->describe()) ;
// 	    return ;
// 	  }

// 	  patPsTricks thePst(&normalizedVariableToTest,
// 			     npReg->getMainPlot(),
// 			     npReg->getUpperPlot(),
// 			     npReg->getLowerPlot(),
// 			     err) ;
	  
// 	  if (err != NULL) {
// 	    WARNING(err->describe()) ;
// 	    return ;
// 	  }

// 	  patString pstricksCode = thePst->getCode(err) ;
// 	  if (err != NULL) {
// 	    WARNING(err->describe()) ;
// 	    return ;
// 	  }

// 	  patString latexCode = t->generateLatexCode(pstricksCode) ;
	  
// // 	  stringstream str ;
// // 	  str << "npreg" << altName << "_" << alt << ".enu" ;
// // 	  npReg.saveOnFile(patString(str.str())) ;

// 	}
//      }
//     }
//     else {
//       WARNING("Zheng test not yet implemented for anything else than probabilties") ;
      
//     }

//  }
}

patBoolean patModelSpec::includeUtilitiesInSimulation() const {

  if (isMixedLogit()) {
    return patFALSE ;
  }

  if (containsDiscreteParameters()) {
    return patFALSE ;
  }

  return patTRUE ;
}

vector<patOneZhengFosgerau>* patModelSpec::getZhengFosgerauVariables() {
  return &zhengFosgerau ;
}

patBetaLikeParameter* patModelSpec::getMGEVParameter() {
  return generalizedExtremeValueParameter ;
}



#ifdef GIANLUCA

patVariables* patModelSpec::gatherGianlucaGradient(patVariables* grad,
						   patVariables* betaDerivatives,
						   patError*& err) {
  
  if (err != NULL) {
    WARNING(err->describe());
    return NULL ;
  }

  if (grad == NULL) {
    err = new patErrNullPointer("trVector") ;
    WARNING(err->describe()) ;
    return NULL ;
  }

  if (grad->size() !=  getNbrNonFixedParameters()) {
    stringstream str ;
    str << "Gradient's dimension (" << grad->size() 
	<< ") is incompatible with problem's dimension (" 
	<<  getNbrNonFixedParameters() << ")" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL ;
  }

  if (firstGatherGradient) {
    thisGradient.resize(grad->size()) ;
    keepBetaIndices.resize(grad->size()) ;
    firstGatherGradient = patFALSE ;
  }

  // beta values
  
  for (allBetaIter->first() ;
       !allBetaIter->isDone() ;
       allBetaIter->next()) {
    patBetaLikeParameter bb = allBetaIter->currentItem() ;
    
    if (!bb.isFixed) {
      if (bb.index >= grad->size()) {
	err = new patErrOutOfRange<unsigned long>(bb.index,
						  0,
						  grad->size()-1) ;
	WARNING(err->describe()) ;
	return NULL ;
      }
      if (bb.id >= betaDerivatives->size()) {
	err = new patErrOutOfRange<unsigned long>(bb.id,
						  0,
						  betaDerivatives->size()-1) ;
	WARNING(err->describe()) ;
	return NULL ;
	
      }
      (*grad)[bb.index] += thisGradient[bb.index] = -(*betaDerivatives)[bb.id] ;
      keepBetaIndices[bb.index] = bb.id ;
    }
  }
  return grad ;
}

#endif


