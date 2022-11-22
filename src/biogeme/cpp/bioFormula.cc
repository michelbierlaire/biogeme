//-*-c++-*------------------------------------------------------------
//
// File name : bioFormula.cc
// @date   Mon Apr 23 13:56:24 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioFormula.h"

#include "bioDebug.h"

#include <vector>
#include <map>
#include <sstream>
#include "bioMemoryManagement.h"
#include "bioTypes.h"
#include "bioString.h"
#include "bioExceptions.h"

#include "bioExprFreeParameter.h"
#include "bioExprFixedParameter.h"
#include "bioExprVariable.h"
#include "bioExprAnd.h"
#include "bioExprOr.h"
#include "bioExprEqual.h"
#include "bioExprNotEqual.h"
#include "bioExprLess.h"
#include "bioExprLessOrEqual.h"
#include "bioExprGreater.h"
#include "bioExprGreaterOrEqual.h"
#include "bioExprElem.h"
#include "bioExprPlus.h"
#include "bioExprMinus.h"
#include "bioExprTimes.h"
#include "bioExprDivide.h"
#include "bioExprPower.h"
#include "bioExprUnaryMinus.h"
#include "bioExprExp.h"
#include "bioExprLog.h"
#include "bioExprLogzero.h"
#include "bioExprMultSum.h"
#include "bioExprLogLogit.h"
#include "bioExprLogLogitFullChoiceSet.h"
#include "bioExprLinearUtility.h"
#include "bioExprNumeric.h"
#include "bioExprDerive.h"
#include "bioExprDraws.h"
#include "bioExprMontecarlo.h"
#include "bioExprNormalCdf.h"
#include "bioExprPanelTrajectory.h"
#include "bioExprRandomVariable.h"
#include "bioExprIntegrate.h"
#include "bioExprMin.h"
#include "bioExprMax.h"

bioFormula::bioFormula(): theFormula(NULL) {

}

void bioFormula::setExpression(std::vector<bioString> expressionsStrings) {
  // Process the formulas
  for (std::vector<bioString>::iterator i = expressionsStrings.begin() ;
       i != expressionsStrings.end() ;
       ++i) {
    // As the formula is the last in the list, it will
    // be correct at the end of the loop. The other assignments do not
    // matter.
    theFormula = processFormula(*i) ;
  }
}

void bioFormula::resetExpression() {
  theFormula = NULL ;
}

bioBoolean bioFormula::isDefined() const {
  return theFormula != NULL ;
}

bioFormula::~bioFormula() {
  
}

bioExpression* bioFormula::processFormula(bioString f) {
  bioExpression* theExpression(NULL) ;
  bioString typeOfExpression = extractParentheses('<','>',f) ;
  bioString id = extractParentheses('{','}',f) ;
  std::map<bioString,bioExpression*>::iterator found = expressions.find(id) ;
  if (found != expressions.end()) {
    // The expression has already been processed
    return found->second ;
  }
  if (typeOfExpression == "Beta") {
    bioString name = extractParentheses('"','"',f) ;
    bioUInt status = std::stoi(extractParentheses('[',']',f)) ;
    std::vector<bioString> items = split(f,',') ;
    bioUInt uniqueId = std::stoi(items[1]) ;
    bioUInt parameterId = std::stoi(items[2]) ;
    if (status == 0) {
      theExpression = bioMemoryManagement::the()->get_bioExprFreeParameter(uniqueId,
									   parameterId,
									   name) ;
    }
    else {
      theExpression = bioMemoryManagement::the()->get_bioExprFixedParameter(uniqueId,
									    parameterId,
									    name) ;
    }
    expressions[id] = theExpression ;
    literals[id] = theExpression ;
    return theExpression ;
  }
  if (typeOfExpression == "Variable" ||
      typeOfExpression == "DefineVariable") {
    bioString name = extractParentheses('"','"',f) ;
    std::vector<bioString> items = split(f,',') ;
    bioUInt uniqueId = std::stoi(items[1]) ;
    bioUInt variableId = std::stoi(items[2]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprVariable(uniqueId,
								    variableId,
								    name) ;
    expressions[id] = theExpression ;
    literals[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "bioDraws") {
    bioString name = extractParentheses('"','"',f) ;
    std::vector<bioString> items = split(f,',') ;
    bioUInt uniqueId = std::stoi(items[1]) ;
    bioUInt drawId = std::stoi(items[2]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprDraws(uniqueId,
								 drawId,
								 name) ;
    expressions[id] = theExpression ;
    literals[id] = theExpression ;
    return theExpression ;
    
  }
  else if (typeOfExpression == "RandomVariable") {
    bioString name = extractParentheses('"','"',f) ;
    std::vector<bioString> items = split(f,',') ;
    bioUInt uniqueId = std::stoi(items[1]) ;
    bioUInt rvId = std::stoi(items[2]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprRandomVariable(uniqueId,
									  rvId,
									  name) ;
    expressions[id] = theExpression ;
    literals[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "Numeric") {
    std::vector<bioString> items = split(f,',') ;
    bioReal v = std::stof(items[1]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprNumeric(v) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "Plus") {
    bioString strChildren = extractParentheses('(',')',f) ;
    bioUInt children = std::stoi(strChildren) ;
    if (children != 2) {
      std::stringstream str ;
      str << "Incorrect number of children for Plus: " << children ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator fl = expressions.find(items[1]) ;
    std::map<bioString,bioExpression*>::iterator fr = expressions.find(items[2]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprPlus(fl->second,fr->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "Minus") {
    bioString strChildren = extractParentheses('(',')',f) ;
    bioUInt children = std::stoi(strChildren) ;
    if (children != 2) {
      std::stringstream str ;
      str << "Incorrect number of children for Plus: " << children ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator fl = expressions.find(items[1]) ;
    std::map<bioString,bioExpression*>::iterator fr = expressions.find(items[2]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprMinus(fl->second,fr->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "Times") {
    bioString strChildren = extractParentheses('(',')',f) ;
    bioUInt children = std::stoi(strChildren) ;
    if (children != 2) {
      std::stringstream str ;
      str << "Incorrect number of children for Plus: " << children ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator fl = expressions.find(items[1]) ;
    std::map<bioString,bioExpression*>::iterator fr = expressions.find(items[2]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprTimes(fl->second,fr->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "Divide") {
    bioString strChildren = extractParentheses('(',')',f) ;
    bioUInt children = std::stoi(strChildren) ;
    if (children != 2) {
      std::stringstream str ;
      str << "Incorrect number of children for Plus: " << children ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator fl = expressions.find(items[1]) ;
    std::map<bioString,bioExpression*>::iterator fr = expressions.find(items[2]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprDivide(fl->second,fr->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "Power") {
    bioString strChildren = extractParentheses('(',')',f) ;
    bioUInt children = std::stoi(strChildren) ;
    if (children != 2) {
      std::stringstream str ;
      str << "Incorrect number of children for Plus: " << children ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator fl = expressions.find(items[1]) ;
    std::map<bioString,bioExpression*>::iterator fr = expressions.find(items[2]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprPower(fl->second,fr->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "And") {
    bioString strChildren = extractParentheses('(',')',f) ;
    bioUInt children = std::stoi(strChildren) ;
    if (children != 2) {
      std::stringstream str ;
      str << "Incorrect number of children for And: " << children ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator fl = expressions.find(items[1]) ;
    std::map<bioString,bioExpression*>::iterator fr = expressions.find(items[2]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprAnd(fl->second,fr->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "Or") {
    bioString strChildren = extractParentheses('(',')',f) ;
    bioUInt children = std::stoi(strChildren) ;
    if (children != 2) {
      std::stringstream str ;
      str << "Incorrect number of children for Or: " << children ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator fl = expressions.find(items[1]) ;
    std::map<bioString,bioExpression*>::iterator fr = expressions.find(items[2]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprOr(fl->second, fr->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "Equal") {
    bioString strChildren = extractParentheses('(',')',f) ;
    bioUInt children = std::stoi(strChildren) ;
    if (children != 2) {
      std::stringstream str ;
      str << "Incorrect number of children for Equal: " << children ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator fl = expressions.find(items[1]) ;
    std::map<bioString,bioExpression*>::iterator fr = expressions.find(items[2]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprEqual(fl->second, fr->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "NotEqual") {
    bioString strChildren = extractParentheses('(',')',f) ;
    bioUInt children = std::stoi(strChildren) ;
    if (children != 2) {
      std::stringstream str ;
      str << "Incorrect number of children for NotEqual: " << children ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator fl = expressions.find(items[1]) ;
    std::map<bioString,bioExpression*>::iterator fr = expressions.find(items[2]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprNotEqual(fl->second, fr->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "Less") {
    bioString strChildren = extractParentheses('(',')',f) ;
    bioUInt children = std::stoi(strChildren) ;
    if (children != 2) {
      std::stringstream str ;
      str << "Incorrect number of children for Less: " << children ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator fl = expressions.find(items[1]) ;
    std::map<bioString,bioExpression*>::iterator fr = expressions.find(items[2]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprLess(fl->second,fr->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "LessOrEqual") {
    bioString strChildren = extractParentheses('(',')',f) ;
    bioUInt children = std::stoi(strChildren) ;
    if (children != 2) {
      std::stringstream str ;
      str << "Incorrect number of children for LessOrEqual: " << children ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator fl = expressions.find(items[1]) ;
    std::map<bioString,bioExpression*>::iterator fr = expressions.find(items[2]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprLessOrEqual(fl->second,fr->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "Greater") {
    bioString strChildren = extractParentheses('(',')',f) ;
    bioUInt children = std::stoi(strChildren) ;
    if (children != 2) {
      std::stringstream str ;
      str << "Incorrect number of children for Greater: " << children ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator fl = expressions.find(items[1]) ;
    std::map<bioString,bioExpression*>::iterator fr = expressions.find(items[2]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprGreater(fl->second,fr->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "GreaterOrEqual") {
    bioString strChildren = extractParentheses('(',')',f) ;
    bioUInt children = std::stoi(strChildren) ;
    if (children != 2) {
      std::stringstream str ;
      str << "Incorrect number of children for GreaterOrEqual: " << children ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator fl = expressions.find(items[1]) ;
    std::map<bioString,bioExpression*>::iterator fr = expressions.find(items[2]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprGreaterOrEqual(fl->second,fr->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "bioMin") {
    bioString strChildren = extractParentheses('(',')',f) ;
    bioUInt children = std::stoi(strChildren) ;
    if (children != 2) {
      std::stringstream str ;
      str << "Incorrect number of children for bioMin: " << children ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator fl = expressions.find(items[1]) ;
    std::map<bioString,bioExpression*>::iterator fr = expressions.find(items[2]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprMin(fl->second,fr->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "bioMax") {
    bioString strChildren = extractParentheses('(',')',f) ;
    bioUInt children = std::stoi(strChildren) ;
    if (children != 2) {
      std::stringstream str ;
      str << "Incorrect number of children for bioMax: " << children ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator fl = expressions.find(items[1]) ;
    std::map<bioString,bioExpression*>::iterator fr = expressions.find(items[2]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprMax(fl->second,fr->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "UnaryMinus") {
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator e = expressions.find(items[1]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprUnaryMinus(e->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "MonteCarlo") {
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator e = expressions.find(items[1]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprMontecarlo(e->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "bioNormalCdf") {
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator e = expressions.find(items[1]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprNormalCdf(e->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "PanelLikelihoodTrajectory") {
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator e = expressions.find(items[1]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprPanelTrajectory(e->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "exp") {
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator e = expressions.find(items[1]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprExp(e->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "Derive") {
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator e = expressions.find(items[1]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprDerive(e->second,
								  bioUInt(std::stoi(items[2]))) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "Integrate") {
    std::vector<bioString> items = split(f,',') ;
    
    std::map<bioString,bioExpression*>::iterator e = expressions.find(items[1]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprIntegrate(e->second,bioUInt(std::stoi(items[2]))) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "log") {
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator e = expressions.find(items[1]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprLog(e->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "logzero") {
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator e = expressions.find(items[1]) ;
    theExpression = bioMemoryManagement::the()->get_bioExprLogzero(e->second) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "bioLinearUtility") {
    bioString strNbrTerms = extractParentheses('(',')',f) ;
    bioUInt nbrTerms = std::stoi(strNbrTerms) ;
    std::vector<bioString> terms = split(f,',') ;
    std::vector<bioLinearTerm> listOfTerms ;
    for (bioUInt i = 0 ; i < nbrTerms ; ++i) {
      bioLinearTerm aTerm ;
      aTerm.theBeta = expressions.find(terms[i*6+1])->second ;
      aTerm.theBetaId = std::stoi(terms[i*6+2]) ;
      aTerm.theBetaName = terms[i*6+3] ;
      aTerm.theVar = expressions.find(terms[i*6+4])->second ;
      aTerm.theVarId = std::stoi(terms[i*6+5]) ;
      aTerm.theVarName = terms[i*6+6] ;
      listOfTerms.push_back(aTerm) ;
    }
    theExpression = bioMemoryManagement::the()->get_bioExprLinearUtility(listOfTerms) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "_bioLogLogit") {
    bioString strNbrUtil = extractParentheses('(',')',f) ;
    bioUInt nbrUtil = std::stoi(strNbrUtil) ;
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator choice = expressions.find(items[1]) ;
    std::map<bioUInt,bioExpression*> theUtils ;
    std::map<bioUInt,bioExpression*> theAvails ;
    for (bioUInt i = 0 ; i < nbrUtil ; ++i) {
      bioUInt alt = std::stoi(items[2+3*i]) ;
      std::map<bioString,bioExpression*>::iterator u = expressions.find(items[2+3*i+1]) ;
      theUtils[alt] = u->second ;
      std::map<bioString,bioExpression*>::iterator a = expressions.find(items[2+3*i+2]) ;
      theAvails[alt] = a->second ;
      
    }

    theExpression = bioMemoryManagement::the()->get_bioExprLogLogit(choice->second,theUtils,theAvails) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "_bioLogLogitFullChoiceSet") {
    bioString strNbrUtil = extractParentheses('(',')',f) ;
    bioUInt nbrUtil = std::stoi(strNbrUtil) ;
    std::vector<bioString> items = split(f,',') ;
    std::map<bioString,bioExpression*>::iterator choice = expressions.find(items[1]) ;
    std::map<bioUInt,bioExpression*> theUtils ;
    for (bioUInt i = 0 ; i < nbrUtil ; ++i) {
      bioUInt alt = std::stoi(items[2+3*i]) ;
      std::map<bioString,bioExpression*>::iterator u = expressions.find(items[2+3*i+1]) ;
      theUtils[alt] = u->second ;
    }
    theExpression = bioMemoryManagement::the()->get_bioExprLogLogitFullChoiceSet(choice->second,
										 theUtils) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "bioMultSum") {
    bioString strNbrTerms = extractParentheses('(',')',f) ;
    bioUInt nbrTerms = std::stoi(strNbrTerms) ;
    std::vector<bioString> items = split(f,',') ;
    std::vector<bioExpression*> theExpressions ;
    for (bioUInt i = 0 ; i < nbrTerms ; ++i) {
      std::map<bioString,bioExpression*>::iterator u = expressions.find(items[1+i]) ;
      theExpressions.push_back(u->second) ;
    }

    theExpression = bioMemoryManagement::the()->get_bioExprMultSum(theExpressions) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else if (typeOfExpression == "Elem") {
    bioString strNbrExpr = extractParentheses('(',')',f) ;
    bioUInt nbrExpr = std::stoi(strNbrExpr) ;
    std::vector<bioString> items = split(f,',') ;

    
    std::map<bioString,bioExpression*>::iterator key = expressions.find(items[1]) ;
    std::map<bioUInt,bioExpression*> theExpressions ;
    for (bioUInt i = 0 ; i < nbrExpr ; ++i) {
      bioUInt alt = std::stoi(items[2+2*i]) ;
      std::map<bioString,bioExpression*>::iterator e = expressions.find(items[2+2*i+1]) ;
      if (e == expressions.end()) {
	std::stringstream str ;
	str << "No expression number: " << items[2+2*i] ;
	throw bioExceptions(__FILE__,__LINE__,str.str()) ;
      }
      theExpressions[alt] = e->second ;
      
    }

    theExpression = bioMemoryManagement::the()->get_bioExprElem(key->second,theExpressions) ;
    expressions[id] = theExpression ;
    return theExpression ;
  }
  else {
    std::stringstream str ;
    str << "Unknown expression: " << typeOfExpression << ": " << f ;
    throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  }
      
  return theExpression ;
}

bioExpression* bioFormula::getExpression() {
  return theFormula ;
}

void bioFormula::setParameters(std::vector<bioReal>* p) {
  for (std::map<bioString,bioExpression*>::iterator i = literals.begin() ;
       i != literals.end() ;
       ++i) {
    i->second->setParameters(p) ;
  }
}

void bioFormula::setFixedParameters(std::vector<bioReal>* p) {
  for (std::map<bioString,bioExpression*>::iterator i = literals.begin() ;
       i != literals.end() ;
       ++i) {
    i->second->setFixedParameters(p) ;
  }
}


void bioFormula::setDraws(std::vector< std::vector< std::vector<bioReal> > >* d) {
  for (std::map<bioString,bioExpression*>::iterator i = expressions.begin() ;
       i != expressions.end() ;
       ++i) {
    i->second->setDraws(d) ;
  }
}

void bioFormula::setData(std::vector< std::vector<bioReal> >* d) {
  for (std::map<bioString,bioExpression*>::iterator i = expressions.begin() ;
       i != expressions.end() ;
       ++i) {
    i->second->setData(d) ;
  }
}

void bioFormula::setMissingData(bioReal md) {
  for (std::map<bioString,bioExpression*>::iterator i = expressions.begin() ;
       i != expressions.end() ;
       ++i) {
    i->second->setMissingData(md) ;
  }
}


void bioFormula::setDataMap(std::vector< std::vector<bioUInt> >* dm) {
  for (std::map<bioString,bioExpression*>::iterator i = expressions.begin() ;
       i != expressions.end() ;
       ++i) {
    i->second->setDataMap(dm) ;
  }
}

void bioFormula::setRowIndex(bioUInt* r) {
  for (std::map<bioString,bioExpression*>::iterator e = expressions.begin() ;
       e != expressions.end() ;
       ++e) {
    e->second->setRowIndex(r) ;
  }

}

void bioFormula::setIndividualIndex(bioUInt* i) {
  for (std::map<bioString,bioExpression*>::iterator e = expressions.begin() ;
       e != expressions.end() ;
       ++e) {
    e->second->setIndividualIndex(i) ;
  }
}

std::ostream& operator<<(std::ostream &str, const bioFormula& x) {
  if (x.theFormula != NULL) {
    str << x.theFormula->print() ;
  }
  return str ;
}

