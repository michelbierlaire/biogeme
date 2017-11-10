//-*-c++-*------------------------------------------------------------
//
// File name : bioRandomDraws.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Thu Jul 30 16:55:28 2009
//
//--------------------------------------------------------------------

#include <sstream>
#include "patMath.h"
#include "bioSample.h"
#include "patOutputFiles.h"
#include "patFileSize.h"
#include "bioLiteralRepository.h"
#include "bioRandomDraws.h"
#include "bioParameters.h"
#include "patErrMiscError.h"
#include "patErrOutOfRange.h"
#include "patErrNullPointer.h"
#include "bioDrawIterator.h"
#include "bioRowIterator.h"
#include "patUnixUniform.h"
#include "patHalton.h"
#include "patHessTrain.h"
#include "patNormalWichura.h"
#include "patCenteredUniform.h"
#include "patZeroOneUniform.h"
#include "bioExpression.h"
#include "bioPythonSingletonFactory.h"

bioRandomDraws::bioRandomDraws() : draws(NULL), uniformDraws(NULL), rng(NULL),theNormalGenerator(NULL), theRectangularGenerator(NULL), theZeroOneGenerator(NULL), drawDim(0), drawsInitialized(patFALSE) {

}

void bioRandomDraws::initDraws(patError*& err) {
  patString rvType = bioParameters::the()->getValueString("RandomDistribution",err) ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  DEBUG_MESSAGE("++++++++++++ " << rvType << " ++++++++++++++++++") ;
  patULong seed = bioParameters::the()->getValueInt("Seed",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  if (rvType == "HALTON") {
    DETAILED_MESSAGE("Prepare Halton draws for " << types.size() << " random parameters") ;
    patULong maxPrime = bioParameters::the()->getValueInt("maxPrimeNumbers",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
    rng = new patHalton(types.size(),
			maxPrime,
			drawDim,
			err) ;
    if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
    }
  }
  else if (rvType == "MLHS") {
    
    patUnixUniform* arng = new patUnixUniform(seed) ;
    rng = new patHessTrain(drawDim,arng) ;
  }
  else if (rvType == "PSEUDO") {
    rng = new patUnixUniform(seed) ;
  }
  else {
    err = new patErrMiscError("Unknown value of the parameter RandomDistribution. Valid entries are \"HALTON\", \"PSEUDO\" and \"MLHS\"") ;
    WARNING(err->describe()) ;
    return ;
  }
  drawsInitialized = patTRUE ;
}

bioRandomDraws::~bioRandomDraws() {
  if (draws != NULL) {
    for (patULong r = 0 ; r < drawDim ; ++r) {
      for (patULong rv = 0 ; rv < betaDim ; ++rv) {
	delete [] draws[r][rv] ;
      }
      delete [] draws[r] ;
    }
    delete [] draws ;
  }

  if (uniformDraws != NULL) {
    for (patULong r = 0 ; r < drawDim ; ++r) {
      for (patULong rv = 0 ; rv < betaDim ; ++rv) {
	delete [] uniformDraws[r][rv] ;
      }
      delete [] uniformDraws[r] ;
    }
    delete [] uniformDraws ;
  }
}
bioRandomDraws* bioRandomDraws::the() {
  return bioPythonSingletonFactory::the()->bioRandomDraws_the() ;
}

patULong bioRandomDraws::addRandomVariable(patString name, 
					   bioRandomDraws::bioDrawsType type,
					   patString index,
					   bioExpression* theExpr,
					   patString iteratorName,
					   patError*& err) {
  if (type == UNDEFINED) {
    err = new patErrMiscError("Cannot create a variable with an undefined type") ;
    WARNING(err->describe()) ;
    return patBadId ;
  }
  map<patString,patULong>::iterator found = idFromName.find(name) ;
  if (found != idFromName.end()) {
    if (types[found->second] != type) {
      stringstream str;
      str << "Variable " << name << " has two different types: " 
	  << getTypeName(types[found->second]) << " and " << getTypeName(type) ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patBadId;
    }
    return found->second;
  }
  patULong id = names.size() ;
  names.push_back(name) ;
  indexNames.push_back(index) ;
  patULong indexId = bioLiteralRepository::the()->getColumnIdOfVariable(index, err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patBadId ;
  }
  indexIds.push_back(indexId) ;
  listOfIds.push_back(vector<patULongLong>()) ;
  types.push_back(type) ;
  expressions.push_back(theExpr) ;
  iteratorNames.push_back(iteratorName) ;

  idFromName[name] = id ;
  return id ;
}


void bioRandomDraws::generateDraws(bioSample* theSample, patError*& err) {


  drawDim = bioParameters::the()->getValueInt("NbrOfDraws",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  if (!drawsInitialized) {
    initDraws(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }

  if (!hasRandomVariables()) {
    return ;
  }
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  int dump = bioParameters::the()->getValueInt("dumpDrawsOnFile",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  // Largest ID
  patULongLong largestId(0) ;
  for (patULong rv = 0 ; rv < types.size() ; ++rv) {
    for (vector<patULongLong>::iterator ids = listOfIds[rv].begin() ;
	 ids != listOfIds[rv].end();
	 ++ids) {
      if (*ids > largestId) {
	largestId = *ids ;
      }
    }
  }

  betaDim = types.size() ;
  idDim = largestId+1 ;
  draws = new patReal**[drawDim] ;
  for (patULong r = 0 ; r < drawDim ; ++r) {
    draws[r] = new patReal*[betaDim] ;
    for (patULong rv = 0 ; rv < betaDim ; ++rv) {
      draws[r][rv] = new patReal[idDim] ;
    }
  }

  long udraws = bioParameters::the()->getValueInt("saveUniformDraws",err) ; 
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  if (udraws != 0) {
    uniformDraws = new patReal**[drawDim] ;
    for (patULong r = 0 ; r < drawDim ; ++r) {
      uniformDraws[r] = new patReal*[betaDim] ;
      for (patULong rv = 0 ; rv < betaDim ; ++rv) {
	uniformDraws[r][rv] = new patReal[idDim] ;
      }
    }
  }


  DETAILED_MESSAGE("Generate " << drawDim << " draws for " << betaDim << " random parameters") ;
  DETAILED_MESSAGE("Uniform draws: " << rng->getType()) ;
  if (udraws != 0) {
    DETAILED_MESSAGE("Size reserved in memory: 2 * " << patFileSize(betaDim * drawDim * idDim * sizeof(patReal)) << " = " << idDim << " * " << betaDim << " * " << drawDim << " * " << patFileSize(sizeof(patReal)) ) ;

  }
  else {
    DETAILED_MESSAGE("Size reserved in memory: " << patFileSize(betaDim * drawDim * idDim * sizeof(patReal)) << " = " << idDim << " * " << betaDim << " * " << drawDim << " * " << patFileSize(sizeof(patReal)) ) ;
  }
  DETAILED_MESSAGE("Size of draws memory        : " << sizeof(draws)) ;
  if (udraws != 0) {
    DETAILED_MESSAGE("Size of uniform draws memory: " << sizeof(uniformDraws)) ;
  }
  // Init random number generators

  patNormalWichura* theNormal = new patNormalWichura() ;
  theNormal->setUniform(rng) ;
  theNormalGenerator = theNormal ;
  
  patZeroOneUniform* theZeroOne = new patZeroOneUniform(dump) ; 
  theZeroOne->setUniform(rng) ;
  theZeroOneGenerator = theZeroOne ;

  patCenteredUniform* theRect = new patCenteredUniform(dump) ;
  theRect->setUniform(rng) ;
  theRectangularGenerator = theRect ;

  patReal truncation = bioParameters::the()->getValueReal("NormalTruncation",
							  err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  for (patULong rv = 0 ; rv < types.size() ; ++rv) {
	switch (types[rv]) {
	case NORMAL:
	  {
	    for (vector<patULongLong>::iterator ids = listOfIds[rv].begin() ;
		 ids != listOfIds[rv].end();
		 ++ids) {
	      //	      DEBUG_MESSAGE("GENERATE DRAWS FOR ID " << *ids) ;
	      for (patULong r = 0 ; r < drawDim ; ++r) {
		//	  DEBUG_MESSAGE("Normal draws[" <<r << "][" << rv << "][" << *ids << "]") ;
		pair<patReal,patReal> theDraw = theNormalGenerator->getNextValue(err) ;
		draws[r][rv][*ids] = theDraw.first ;
		if (err != NULL) {
		  WARNING(err->describe()) ;
		  return ;
		}
		if (udraws != 0) {
		  uniformDraws[r][rv][*ids] = theDraw.second ;
		  if (err != NULL) {
		    WARNING(err->describe()) ;
		    return ;
		  }
		}
	      }
	    }
	    break ;
	  }
	case TRUNCNORMAL:
	  {
	    for (vector<patULongLong>::iterator ids = listOfIds[rv].begin() ;
		 ids != listOfIds[rv].end();
		 ++ids) {
	      for (patULong r = 0 ; r < drawDim ; ++r) {
		//	  DEBUG_MESSAGE("Normal draws[" <<r << "][" << rv << "][" << *ids << "]") ;
		patBoolean reject = patTRUE ;
		pair<patReal,patReal> z ;
		while (reject) {
		  z = theNormalGenerator->getNextValue(err) ;
		  if (err != NULL) {
		    WARNING(err->describe()) ;
		    return ;
		  }
		  if (patAbs(z.first) <= truncation) {
		    reject = patFALSE ;
		  }
		}
		draws[r][rv][*ids] = z.first ;
		if (udraws != 0) {
		  uniformDraws[r][rv][*ids] = z.second ;
		}
	      }
	    }
	    break ;
	  }
	case UNIFORM:
	  {
	    for (vector<patULongLong>::iterator ids = listOfIds[rv].begin() ;
		 ids != listOfIds[rv].end();
		 ++ids) {
	      for (patULong r = 0 ; r < drawDim ; ++r) {
		//	  DEBUG_MESSAGE("Normal draws[" <<r << "][" << rv << "][" << *ids << "]") ;
		//	  DEBUG_MESSAGE("Uniform draws[" <<r << "][" << rv << "][" << *ids << "]") ;
		pair<patReal,patReal> z = theZeroOneGenerator->getNextValue(err) ;
		draws[r][rv][*ids] = z.first ;
		if (udraws != 0) {
		  // Note: in principle, z.first is equal to z.second this case
		  uniformDraws[r][rv][*ids] = z.second ;
		}
	      }
	    }
	    break ;
	  }
	case UNIFORMSYM:
	  {
	    for (vector<patULongLong>::iterator ids = listOfIds[rv].begin() ;
		 ids != listOfIds[rv].end();
		 ++ids) {
	      for (patULong r = 0 ; r < drawDim ; ++r) {
		//	  DEBUG_MESSAGE("Uniform symmetric draws[" <<r << "][" << rv << "][" << *ids << "]") ;
		pair<patReal,patReal> z = theRectangularGenerator->getNextValue(err) ;
		draws[r][rv][*ids] = z.first ;
		if (udraws != 0) {
		  uniformDraws[r][rv][*ids] = z.second ;
		}
	      }
	    }
	    break ;
	  }
	case USERDEFINED:
	  {
	    if (theSample == NULL) {
	      err = new patErrNullPointer("bioSample") ;
	      WARNING(err->describe()) ;
	      return ;
	    }
	    bioIteratorSpan theSpan(iteratorNames[rv],0) ;
	    bioRowIterator* theIterator = 
	      theSample->createIdIterator(theSpan, 
					  theSpan, 
					  indexNames[rv], 
					  err)  ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	    
	    patULong theIndexId = indexIds[rv] ;
	    for (theIterator->first() ;
		 !theIterator->isDone() ;
		 theIterator->next()) {
	      // for (vector<patULongLong>::iterator ids = listOfIds[rv].begin() ;
	      // 	 ids != listOfIds[rv].end();
	      // 	 ++ids) {
	      const patVariables* x = theIterator->currentItem() ;
	      if (x == NULL) {
		err = new patErrNullPointer("patVariables") ;
		WARNING(err->describe()) ;
		return ;
	      }
	      if (theIndexId >= x->size()) {
		err = new patErrOutOfRange<patULong>(theIndexId,0,x->size()-1) ;
		WARNING(err->describe()) ;
		return ;
	      }
	      //	      DEBUG_MESSAGE("GENERATE "<<drawDim<<" DRAWS FOR ID " << indexNames[rv] << "("<<indexIds[rv]<<"): " << (*x)[theIndexId]) ;
	      for (patULong r = 0 ; r < drawDim ; ++r) {
		if (rv >= expressions.size()) {
		  err = new patErrOutOfRange<patULong>(rv,0,expressions.size()-1) ;
		  WARNING(err->describe()) ;
		  return ;
		}
		bioExpression* theExpr = expressions[rv] ;
		
		if (theExpr == NULL) {
		  err = new patErrNullPointer("bioExpression") ;
		  WARNING(err->describe()) ;
		  return ;
		}
		
		theExpr->setVariables(x) ;
		
		if (uniformDraws != NULL) {
		  pair<patReal**, patReal**> d(draws[r],uniformDraws[r]) ;
		  DEBUG_MESSAGE("Set draws") ;
		  theExpr->setDraws(d) ;
		}
		else {
		  pair<patReal**, patReal**> d(draws[r],NULL) ;
		  theExpr->setDraws(d) ;
		}
		patReal v =  theExpr->getValue(patFALSE,patLapForceCompute, err) ;
		if (err != NULL) {
		  WARNING(err->describe()) ;
		  return ;
		}

		draws[r][rv][patULong((*x)[theIndexId])] = v ;
	      }
	    }
	    break ;
	  }
	case UNDEFINED:
	default:
	  {
	    err = new patErrMiscError("Undefined type for random variable") ;
	    WARNING(err->describe()) ;
	    return ;
	  }
	}
  }


  if (dump != 0) {
    patString dumpFileName("draws.lis") ;
    ofstream df(dumpFileName) ;
    for (patULong rv = 0 ; rv < types.size() ; ++rv) {
      for (vector<patULongLong>::iterator ids = listOfIds[rv].begin() ;
	   ids != listOfIds[rv].end();
	   ++ids) {
	df << names[rv] << "[" << *ids << "]" << '\t' ;
      }
    }
    df << endl ;
  
  
    for (patULong r = 0 ; r < drawDim ; ++r) {
      for (patULong rv = 0 ; rv < types.size() ; ++rv) {
	for (vector<patULongLong>::iterator ids = listOfIds[rv].begin() ;
	     ids != listOfIds[rv].end();
	     ++ids) {
	  df << draws[r][rv][*ids] << '\t' ;
	}
      }
      df << endl ;
    }
    df.close() ;
    patOutputFiles::the()->addDebugFile(dumpFileName,"List of normal draws used for simulation.");
    
    if (uniformDraws != NULL) {
      patString dumpFileName("uniformDraws.lis") ;
      ofstream df(dumpFileName) ;
      for (patULong rv = 0 ; rv < types.size() ; ++rv) {
	for (vector<patULongLong>::iterator ids = listOfIds[rv].begin() ;
	     ids != listOfIds[rv].end();
	     ++ids) {
	  df << names[rv] << "[" << *ids << "]" << '\t' ;
	}
      }
      df << endl ;
      
      
      for (patULong r = 0 ; r < drawDim ; ++r) {
	for (patULong rv = 0 ; rv < types.size() ; ++rv) {
	  for (vector<patULongLong>::iterator ids = listOfIds[rv].begin() ;
	       ids != listOfIds[rv].end();
	       ++ids) {
	    df << uniformDraws[r][rv][*ids] << '\t' ;
	  }
	  df << endl ;
	}
      }
      df.close() ;
      patOutputFiles::the()->addDebugFile(dumpFileName,"List of uniform draws used for simulation.");
    }
  }
}

//vector< vector<patReal> >* bioRandomDraws::getRow(patULong rowId, patError*& err) {
patReal** bioRandomDraws::getRow(patULong rowId, patError*& err) {
  if (rowId >= drawDim) {
    err = new patErrOutOfRange<patULong>(rowId,0,drawDim-1) ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  return draws[rowId] ;
}

patReal** bioRandomDraws::getUniformRow(patULong rowId, patError*& err) {
  if (rowId >= drawDim) {
    err = new patErrOutOfRange<patULong>(rowId,0,drawDim-1) ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  return uniformDraws[rowId] ;
}

patULong bioRandomDraws::getColId(patString name, patError*& err) {
  map<patString,patULong>::iterator found = idFromName.find(name) ;
  if (found != idFromName.end()) {
    return found->second ;
  }
  else {
    stringstream str ;
    str << "Unknown draws: " << name << ". Known draws" ;
    for (map<patString,patULong>::iterator i = idFromName.begin() ;
	 i != idFromName.end() ;
	 ++i) {
      str << i->first << " " ;
    }
    
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patBadId ;
  }
}

patString bioRandomDraws::getIndex(patULong colId, patError*& err) {
  if (colId >= indexNames.size()) {
    err = new patErrOutOfRange<patULong>(colId,0,indexNames.size()-1) ;
    WARNING(err->describe()) ;
    return patString("") ;
  }
  return indexNames[colId];
}

bioRandomDraws::bioDrawsType bioRandomDraws::getType(patULong colId, 
						     patError*& err) {
  if (colId >= types.size()) {
    err = new patErrOutOfRange<patULong>(colId,0,types.size()-1) ;
    WARNING(err->describe()) ;
    return UNDEFINED ;
  }
  return types[colId];
}



bioDrawIterator* bioRandomDraws::createIterator(patError*& err) {
  if (drawDim == 0) {
    err = new patErrMiscError("Cannot create a draw iterator as the number of drwas is 0") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  //  bioDrawIterator* ptr = new bioDrawIterator(&draws) ;
  bioDrawIterator* ptr = new bioDrawIterator(draws,uniformDraws,drawDim) ;
  return ptr ;
}

patBoolean bioRandomDraws::hasRandomVariables() const {
  return !names.empty() ;
}

void bioRandomDraws::populateListOfIds(bioSample* aSample, patError*& err) {
  for (patULong i = 0 ; i < indexNames.size() ; ++i) {
    vector<patULongLong> theList = aSample->getListOfIds(indexNames[i],err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    listOfIds[i] = theList ;
  }
}

patULong bioRandomDraws::nbrOfDraws() const {
  //  return draws.size() ;
  return drawDim ;
}

ostream& operator<<(ostream& str, const bioRandomDraws& x) {
  
  str << "Number of draws:   " << x.drawDim << endl ;
  if (x.drawDim > 0) {
    str << "Number of variables: " << x.betaDim << endl ;
    if (x.betaDim > 0) {
      str << "Number of obs/ind:     " << x.idDim << endl ;
      
      for (patULong obs = 0 ; obs < x.idDim ; ++obs) {
	str << "Observation " << obs << endl ;
	str << "*****************" << endl ;
	for (patULong var = 0 ; var < x.betaDim ;++var) {
	  str << "Draws for " << x.names[var] << ": " ;
	  for (patULong d = 0 ; d <  x.drawDim ; ++d) {
	    str << x.draws[d][var][obs] << " " ;
	    if (x.uniformDraws != NULL) {
	      str << "(" << x.uniformDraws[d][var][obs] << ") " ;
	    }
	  }
	  str << endl ;
	}
	str << endl ;
      }
    }
  }
  return str ;
}

patUniform* bioRandomDraws::getUniformGenerator() {
  return rng ;
}


patULong bioRandomDraws::addUserDraws(patString name,
				      bioExpression* theUserExpression, 
				      patString iteratorName,
				      patError*& err) {

  
  vector<patULong> listOfDraws = theUserExpression->getListOfDraws(err)  ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patBadId ;
  }
  patString e = theUserExpression->getExpression(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patBadId ;
  }

  if (listOfDraws.empty()) {
    stringstream str ;
    str << "Expression for user defined draws does not refer to other draws: " << e ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patBadId ;
  }
  stringstream str ;
  str <<  "Draws " << name << " calculated based on " ;
  patBoolean first(patTRUE) ;
  patULong theIndexId(patBadId) ;
  patString iName ;
  for (vector<patULong>::iterator i = listOfDraws.begin() ;
       i != listOfDraws.end() ;
       ++i) {
    if (theIndexId == patBadId) {
      theIndexId = indexIds[*i] ;
      iName = indexNames[*i] ;
    }
    else {
      if (theIndexId != indexIds[*i]) {
	stringstream str ;
	str << "The draws in the expression correspond to different indices: " << iName << " and " << indexNames[*i] ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return patBadId ;
      }
    }
    if (first) {
      first = patFALSE ;
    }
    else {
      str << ", " ;
    }
    str << names[*i]  ;
  }
  GENERAL_MESSAGE(str.str()) ;

  patULong theId = addRandomVariable(name,
				     USERDEFINED,
				     iName,
				     theUserExpression,
				     iteratorName,
				     err) ;
  if (err != NULL) {
    WARNING(err->describe());
    return patBadId ;
  }
	
  return theId ;
}



patString bioRandomDraws::getTypeName(bioDrawsType t) {
  switch (t) {
  case NORMAL:
    return patString("Normal") ;
  case TRUNCNORMAL:
    return patString("Truncated normal") ;
  case UNIFORM:
    return patString("Uniform") ;
  case UNIFORMSYM: 
    return patString("Symmetric uniform") ;
  case USERDEFINED: 
    return patString("User defined") ;
  case UNDEFINED:
    return patString("Undefined") ;

  }
}

void bioRandomDraws::setDraws(map<patString, bioDraw >* d, patError*& err) {

  for (map<patString, bioDraw >::iterator iter = d->begin() ;
       iter != d->end() ;
       ++iter) {
    addRandomVariable(iter->first, 
		      iter->second.theType, 
		      iter->second.theId,
		      NULL,
		      patString(""),
		      err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }

}


bioRandomDraws::bioDraw::bioDraw(patString name,
			bioRandomDraws::bioDrawsType type, 
			patString id) :  theName(name),
					 theType(type),
					 theId(id) {
  
}

ostream& operator<<(ostream& stream, const bioRandomDraws::bioDraw& d) {
  stream << "Draw " << d.theName << " (" << d.theType << "," << d.theId << ")" ;
  return stream ;
}

ostream& operator<<(ostream& stream, const bioRandomDraws::bioDrawsType& d) {
  switch(d) {
  case bioRandomDraws::NORMAL:
    stream << "Normal" ;
    return stream ;
  case bioRandomDraws::TRUNCNORMAL:
    stream << "Truncated normal" ;
    return stream ;
  case bioRandomDraws::UNIFORM:
    stream << "Uniform(0,1)" ;
    return stream ;
  case bioRandomDraws::UNIFORMSYM:
    stream << "Uniform(-1,1)" ;
    return stream ;
  case bioRandomDraws::USERDEFINED:
    stream << "User defined" ;
    return stream ;
  case bioRandomDraws::UNDEFINED:
    stream << "Undefined" ;
    return stream ;
  }
}
