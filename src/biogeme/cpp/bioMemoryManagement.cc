//-*-c++-*------------------------------------------------------------
//
// File name : bioMemoryManagement.cc
// @date   Sat Sep 26 12:22:24 2020
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#include "bioMemoryManagement.h"
#include "bioDebug.h"
#include "bioExpression.h"
#include "bioExprFreeParameter.h"
#include "bioExprFixedParameter.h"
#include "bioExprVariable.h"
#include "bioExprDraws.h"
#include "bioExprRandomVariable.h"
#include "bioExprNumeric.h"
#include "bioExprPlus.h"
#include "bioExprMinus.h"
#include "bioExprTimes.h"
#include "bioExprDivide.h"
#include "bioExprPower.h"
#include "bioExprAnd.h"
#include "bioExprOr.h"
#include "bioExprEqual.h"
#include "bioExprNotEqual.h"
#include "bioExprLess.h"
#include "bioExprLessOrEqual.h"
#include "bioExprGreater.h"
#include "bioExprGreaterOrEqual.h"
#include "bioExprMin.h"
#include "bioExprMax.h"
#include "bioExprUnaryMinus.h"
#include "bioExprMontecarlo.h"
#include "bioExprNormalCdf.h"
#include "bioExprPanelTrajectory.h"
#include "bioExprExp.h"
#include "bioExprLog.h"
#include "bioExprLogzero.h"
#include "bioExprDerive.h"
#include "bioExprIntegrate.h"
#include "bioExprLogLogit.h"
#include "bioExprLogLogitFullChoiceSet.h"
#include "bioExprMultSum.h"
#include "bioExprElem.h"
#include "bioSeveralExpressions.h"

bioMemoryManagement::bioMemoryManagement() {

}

bioMemoryManagement::~bioMemoryManagement() {
  releaseAllMemory() ;
}


void bioMemoryManagement::releaseAllMemory() {
  for (std::vector<bioExprFreeParameter*>::iterator i = a_bioExprFreeParameter.begin() ;
       i != a_bioExprFreeParameter.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprFreeParameter.clear() ;


  for (std::vector<bioExprFixedParameter*>::iterator i = a_bioExprFixedParameter.begin() ;
       i != a_bioExprFixedParameter.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprFixedParameter.clear() ;

  for (std::vector<bioExprFixedParameter*>::iterator i = a_bioExprFixedParameter.begin() ;
       i != a_bioExprFixedParameter.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprFixedParameter.clear() ;
    
  for (std::vector<bioExprVariable*>::iterator i = a_bioExprVariable.begin() ;
       i != a_bioExprVariable.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprVariable.clear() ;

  for (std::vector<bioExprDraws*>::iterator i = a_bioExprDraws.begin() ;
       i != a_bioExprDraws.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprDraws.clear() ;

  for (std::vector<bioExprRandomVariable*>::iterator i = a_bioExprRandomVariable.begin() ;
       i != a_bioExprRandomVariable.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprRandomVariable.clear() ;
  for (std::vector<bioExprNumeric*>::iterator i = a_bioExprNumeric.begin() ;
       i != a_bioExprNumeric.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprNumeric.clear() ;
  for (std::vector<bioExprPlus*>::iterator i = a_bioExprPlus.begin() ;
       i != a_bioExprPlus.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprPlus.clear() ;
  for (std::vector<bioExprMinus*>::iterator i = a_bioExprMinus.begin() ;
       i != a_bioExprMinus.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprMinus.clear() ;
  for (std::vector<bioExprTimes*>::iterator i = a_bioExprTimes.begin() ;
       i != a_bioExprTimes.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprTimes.clear() ;
  for (std::vector<bioExprDivide*>::iterator i = a_bioExprDivide.begin() ;
       i != a_bioExprDivide.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprDivide.clear() ;
  for (std::vector<bioExprPower*>::iterator i = a_bioExprPower.begin() ;
       i != a_bioExprPower.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprPower.clear() ;
  for (std::vector<bioExprAnd*>::iterator i = a_bioExprAnd.begin() ;
       i != a_bioExprAnd.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprAnd.clear() ;
  for (std::vector<bioExprOr*>::iterator i = a_bioExprOr.begin() ;
       i != a_bioExprOr.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprOr.clear() ;
  for (std::vector<bioExprEqual*>::iterator i = a_bioExprEqual.begin() ;
       i != a_bioExprEqual.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprEqual.clear() ;
  for (std::vector<bioExprNotEqual*>::iterator i = a_bioExprNotEqual.begin() ;
       i != a_bioExprNotEqual.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprNotEqual.clear() ;
  for (std::vector<bioExprLess*>::iterator i = a_bioExprLess.begin() ;
       i != a_bioExprLess.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprLess.clear() ;
  for (std::vector<bioExprLessOrEqual*>::iterator i = a_bioExprLessOrEqual.begin() ;
       i != a_bioExprLessOrEqual.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprLessOrEqual.clear() ;
  for (std::vector<bioExprGreater*>::iterator i = a_bioExprGreater.begin() ;
       i != a_bioExprGreater.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprGreater.clear() ;
  for (std::vector<bioExprGreaterOrEqual*>::iterator i = a_bioExprGreaterOrEqual.begin() ;
       i != a_bioExprGreaterOrEqual.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprGreaterOrEqual.clear() ;
  for (std::vector<bioExprMin*>::iterator i = a_bioExprMin.begin() ;
       i != a_bioExprMin.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprMin.clear() ;
  for (std::vector<bioExprMax*>::iterator i = a_bioExprMax.begin() ;
       i != a_bioExprMax.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprMax.clear() ;
  for (std::vector<bioExprUnaryMinus*>::iterator i = a_bioExprUnaryMinus.begin() ;
       i != a_bioExprUnaryMinus.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprUnaryMinus.clear() ;
  for (std::vector<bioExprMontecarlo*>::iterator i = a_bioExprMontecarlo.begin() ;
       i != a_bioExprMontecarlo.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprMontecarlo.clear() ;
  for (std::vector<bioExprNormalCdf*>::iterator i = a_bioExprNormalCdf.begin() ;
       i != a_bioExprNormalCdf.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprNormalCdf.clear() ;
  for (std::vector<bioExprPanelTrajectory*>::iterator i = a_bioExprPanelTrajectory.begin() ;
       i != a_bioExprPanelTrajectory.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprPanelTrajectory.clear() ;
  for (std::vector<bioExprExp*>::iterator i = a_bioExprExp.begin() ;
       i != a_bioExprExp.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprExp.clear() ;
  for (std::vector<bioExprLog*>::iterator i = a_bioExprLog.begin() ;
       i != a_bioExprLog.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprLog.clear() ;
  for (std::vector<bioExprLogzero*>::iterator i = a_bioExprLogzero.begin() ;
       i != a_bioExprLogzero.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprLogzero.clear() ;
  for (std::vector<bioExprDerive*>::iterator i = a_bioExprDerive.begin() ;
       i != a_bioExprDerive.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprDerive.clear() ;
  for (std::vector<bioExprIntegrate*>::iterator i = a_bioExprIntegrate.begin() ;
       i != a_bioExprIntegrate.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprIntegrate.clear() ;
  for (std::vector<bioExprLinearUtility*>::iterator i = a_bioExprLinearUtility.begin() ;
       i != a_bioExprLinearUtility.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprLinearUtility.clear() ;
  for (std::vector<bioExprLogLogit*>::iterator i = a_bioExprLogLogit.begin() ;
       i != a_bioExprLogLogit.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprLogLogit.clear() ;
  for (std::vector<bioExprLogLogitFullChoiceSet*>::iterator i = a_bioExprLogLogitFullChoiceSet.begin() ;
       i != a_bioExprLogLogitFullChoiceSet.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprLogLogitFullChoiceSet.clear() ;
  for (std::vector<bioExprMultSum*>::iterator i = a_bioExprMultSum.begin() ;
       i != a_bioExprMultSum.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprMultSum.clear() ;
  for (std::vector<bioExprElem*>::iterator i = a_bioExprElem.begin() ;
       i != a_bioExprElem.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioExprElem.clear() ;
  
  for (std::vector<bioSeveralExpressions*>::iterator i = a_bioSeveralExpressions.begin() ;
       i != a_bioSeveralExpressions.end() ;
       ++i) {
    delete(*i) ;
  }
  a_bioSeveralExpressions.clear() ;
  
}

bioMemoryManagement* bioMemoryManagement::the() {
  static bioMemoryManagement* singleInstance = NULL;
  if (singleInstance == NULL) {
    singleInstance = new bioMemoryManagement() ;
  } 
  return singleInstance ;
  
}

bioExprFreeParameter* bioMemoryManagement::get_bioExprFreeParameter(bioUInt literalId,
								     bioUInt parameterId,
								     bioString name) {
  bioExprFreeParameter* ptr = new bioExprFreeParameter(literalId,
						       parameterId,
						       name) ;
  a_bioExprFreeParameter.push_back(ptr) ;
  return ptr ;
}

bioExprFixedParameter* bioMemoryManagement::get_bioExprFixedParameter(bioUInt literalId,
								      bioUInt parameterId,
								      bioString name) {
  bioExprFixedParameter* ptr = new bioExprFixedParameter(literalId,
							 parameterId,
							 name) ;
  a_bioExprFixedParameter.push_back(ptr) ;
  return ptr ;
}

bioExprVariable* bioMemoryManagement::get_bioExprVariable(bioUInt literalId,
						      bioUInt variableId,
						      bioString name) {
  bioExprVariable* ptr = new bioExprVariable(literalId,
					     variableId,
					     name) ;
  a_bioExprVariable.push_back(ptr) ;
  return ptr ;
}

bioExprDraws* bioMemoryManagement::get_bioExprDraws(bioUInt uniqueId,
					     bioUInt drawId,
					     bioString name) {
  bioExprDraws* ptr = new bioExprDraws(uniqueId,
			 drawId,
			 name) ;
  a_bioExprDraws.push_back(ptr) ;
  return ptr ;
}

bioExprRandomVariable* bioMemoryManagement::get_bioExprRandomVariable(bioUInt literalId,
									bioUInt id,
									bioString name) {
  bioExprRandomVariable* ptr = new bioExprRandomVariable(literalId,
							 id,
							 name) ;
  a_bioExprRandomVariable.push_back(ptr) ;
  return ptr ;
}

bioExprNumeric* bioMemoryManagement::get_bioExprNumeric(bioReal v) {
  bioExprNumeric* ptr = new bioExprNumeric(v) ;
  a_bioExprNumeric.push_back(ptr) ;
  return ptr ;
}

bioExprPlus* bioMemoryManagement::get_bioExprPlus(bioExpression* ell, bioExpression* r) {
  bioExprPlus* ptr = new bioExprPlus(ell, r) ;
  a_bioExprPlus.push_back(ptr) ;
  return ptr ;
}

bioExprMinus* bioMemoryManagement::get_bioExprMinus(bioExpression* ell, bioExpression* r) {
  bioExprMinus* ptr = new bioExprMinus(ell, r) ;
  a_bioExprMinus.push_back(ptr) ;
  return ptr ;
}

bioExprTimes* bioMemoryManagement::get_bioExprTimes(bioExpression* ell, bioExpression* r) {
  bioExprTimes* ptr = new bioExprTimes(ell, r) ;
  a_bioExprTimes.push_back(ptr) ;
  return ptr ;
}

bioExprDivide* bioMemoryManagement::get_bioExprDivide(bioExpression* ell, bioExpression* r) {
  bioExprDivide* ptr = new bioExprDivide(ell, r) ;
  a_bioExprDivide.push_back(ptr) ;
  return ptr ;
}

bioExprPower* bioMemoryManagement::get_bioExprPower(bioExpression* ell, bioExpression* r) {
  bioExprPower* ptr = new bioExprPower(ell, r) ;
  a_bioExprPower.push_back(ptr) ;
  return ptr ;
}

bioExprAnd* bioMemoryManagement::get_bioExprAnd(bioExpression* ell, bioExpression* r) {
  bioExprAnd* ptr = new bioExprAnd(ell, r) ;
  a_bioExprAnd.push_back(ptr) ;
  return ptr ;
}

bioExprOr* bioMemoryManagement::get_bioExprOr(bioExpression* ell, bioExpression* r) {
  bioExprOr* ptr = new bioExprOr(ell, r) ;
  a_bioExprOr.push_back(ptr) ;
  return ptr ;
}

bioExprEqual* bioMemoryManagement::get_bioExprEqual(bioExpression* ell, bioExpression* r) {
  bioExprEqual* ptr = new bioExprEqual(ell, r) ;
  a_bioExprEqual.push_back(ptr) ;
  return ptr ;
}

bioExprNotEqual* bioMemoryManagement::get_bioExprNotEqual(bioExpression* ell, bioExpression* r) {
  bioExprNotEqual* ptr = new bioExprNotEqual(ell, r) ;
  a_bioExprNotEqual.push_back(ptr) ;
  return ptr ;
}

bioExprLess* bioMemoryManagement::get_bioExprLess(bioExpression* ell, bioExpression* r) {
  bioExprLess* ptr = new bioExprLess(ell, r) ;
  a_bioExprLess.push_back(ptr) ;
  return ptr ;
}

bioExprLessOrEqual* bioMemoryManagement::get_bioExprLessOrEqual(bioExpression* ell,
								bioExpression* r) {
  bioExprLessOrEqual* ptr = new bioExprLessOrEqual(ell, r) ;
  a_bioExprLessOrEqual.push_back(ptr) ;
  return ptr ;
}

bioExprGreater* bioMemoryManagement::get_bioExprGreater(bioExpression* ell,
							bioExpression* r) {
  bioExprGreater* ptr = new bioExprGreater(ell, r) ;
  a_bioExprGreater.push_back(ptr) ;
  return ptr ;
}

bioExprGreaterOrEqual* bioMemoryManagement::get_bioExprGreaterOrEqual(bioExpression* ell,
								      bioExpression* r) {
  bioExprGreaterOrEqual* ptr = new bioExprGreaterOrEqual(ell, r) ;
  a_bioExprGreaterOrEqual.push_back(ptr) ;
  return ptr ;
}

bioExprMin* bioMemoryManagement::get_bioExprMin(bioExpression* ell,
						bioExpression* r) {
  bioExprMin* ptr = new bioExprMin(ell, r) ;
  a_bioExprMin.push_back(ptr) ;
  return ptr ;
}

bioExprMax* bioMemoryManagement::get_bioExprMax(bioExpression* ell,
						bioExpression* r) {
  bioExprMax* ptr = new bioExprMax(ell, r) ;
  a_bioExprMax.push_back(ptr) ;
  return ptr ;
}

bioExprUnaryMinus* bioMemoryManagement::get_bioExprUnaryMinus(bioExpression* c) {
  bioExprUnaryMinus* ptr = new bioExprUnaryMinus(c) ;
  a_bioExprUnaryMinus.push_back(ptr) ;
  return ptr ;
}

bioExprMontecarlo* bioMemoryManagement::get_bioExprMontecarlo(bioExpression* c) {
  bioExprMontecarlo* ptr = new bioExprMontecarlo(c) ;
  a_bioExprMontecarlo.push_back(ptr) ;
  return ptr ;
}

bioExprNormalCdf* bioMemoryManagement::get_bioExprNormalCdf(bioExpression* c) {
  bioExprNormalCdf* ptr = new bioExprNormalCdf(c) ;
  a_bioExprNormalCdf.push_back(ptr) ;
  return ptr ;
}

bioExprPanelTrajectory* bioMemoryManagement::get_bioExprPanelTrajectory(bioExpression* c) {
  bioExprPanelTrajectory* ptr = new bioExprPanelTrajectory(c) ;
  a_bioExprPanelTrajectory.push_back(ptr) ;
  return ptr ;
}

bioExprExp* bioMemoryManagement::get_bioExprExp(bioExpression* c) {
  bioExprExp* ptr = new bioExprExp(c) ;
  a_bioExprExp.push_back(ptr) ;
  return ptr ;
}

bioExprLog* bioMemoryManagement::get_bioExprLog(bioExpression* c) {
  bioExprLog* ptr = new bioExprLog(c) ;
  a_bioExprLog.push_back(ptr) ;
  return ptr ;
}

bioExprLogzero* bioMemoryManagement::get_bioExprLogzero(bioExpression* c) {
  bioExprLogzero* ptr = new bioExprLogzero(c) ;
  a_bioExprLogzero.push_back(ptr) ;
  return ptr ;
}

bioExprDerive* bioMemoryManagement::get_bioExprDerive(bioExpression* c, bioUInt lid) {
  bioExprDerive* ptr = new bioExprDerive(c, lid) ;
  a_bioExprDerive.push_back(ptr) ;
  return ptr ;
}

bioExprIntegrate* bioMemoryManagement::get_bioExprIntegrate(bioExpression* c, bioUInt lid) {
  bioExprIntegrate* ptr = new bioExprIntegrate(c, lid) ;
  a_bioExprIntegrate.push_back(ptr) ;
  return ptr ;
}

bioExprLinearUtility* bioMemoryManagement::get_bioExprLinearUtility(std::vector<bioLinearTerm> t) {
  bioExprLinearUtility* ptr = new bioExprLinearUtility(t) ;
  a_bioExprLinearUtility.push_back(ptr) ;
  return ptr ;
}

bioExprLogLogit* bioMemoryManagement::get_bioExprLogLogit(bioExpression* c,
							  std::map<bioUInt,bioExpression*> u,
							  std::map<bioUInt,bioExpression*> a) {
  bioExprLogLogit* ptr = new bioExprLogLogit(c, u, a) ;
  a_bioExprLogLogit.push_back(ptr) ;
  return ptr ;
}

bioExprLogLogitFullChoiceSet* bioMemoryManagement::get_bioExprLogLogitFullChoiceSet(bioExpression* c,
							  std::map<bioUInt,bioExpression*> u) {
  bioExprLogLogitFullChoiceSet* ptr = new bioExprLogLogitFullChoiceSet(c, u) ;
  a_bioExprLogLogitFullChoiceSet.push_back(ptr) ;
  return ptr ;
}

bioExprMultSum* bioMemoryManagement::get_bioExprMultSum(std::vector<bioExpression*> e) {
  bioExprMultSum* ptr = new bioExprMultSum(e) ;
  a_bioExprMultSum.push_back(ptr) ;
  return ptr ;
}

bioExprElem* bioMemoryManagement::get_bioExprElem(bioExpression* k,
						  std::map<bioUInt,bioExpression*> d) {
  bioExprElem* ptr = new bioExprElem(k, d) ;
  a_bioExprElem.push_back(ptr) ;
  return ptr ;
}

bioSeveralExpressions* bioMemoryManagement::get_bioSeveralExpressions(std::vector<bioExpression*> exprs) {
  bioSeveralExpressions* ptr = new bioSeveralExpressions(exprs) ;
  a_bioSeveralExpressions.push_back(ptr) ;
  return ptr ;
}


