//-*-c++-*------------------------------------------------------------
//
// File name : patSpecParser.y
// Michel Bierlaire, EPFL
// Date :      Tue Nov  7 14:23:33 2000
//
//--------------------------------------------------------------------
//

%name patBisonSpec
%define ERROR_BODY = 0
%define LEX_BODY = 0

%define MEMBERS patSpecScanner scanner; patModelSpec *pModel; virtual ~patBisonSpec() {};
%define CONSTRUCTOR_PARAM const patString& fname_
%define CONSTRUCTOR_INIT : scanner(fname_) , pModel(NULL)

%header{
  
#include <fstream>
#include <sstream>
#include <assert.h>

#include "patLoop.h"
#include "patDisplay.h"
#include "patConst.h"
#include "patModelSpec.h"
#include "patAlternative.h"
#include "patArithNode.h"
#include "patArithConstant.h"
#include "patArithVariable.h"
#include "patArithBinaryPlus.h"
#include "patArithBinaryMinus.h"
#include "patArithMult.h"
#include "patArithDivide.h"
#include "patArithPower.h"
#include "patArithEqual.h"
#include "patArithNotEqual.h"
#include "patArithOr.h"
#include "patArithAnd.h"
#include "patArithLess.h"
#include "patArithLessEqual.h"
#include "patArithGreater.h"
#include "patArithGreaterEqual.h"
#include "patArithUnaryMinus.h"
#include "patArithNot.h"
#include "patArithSqrt.h"
#include "patArithLog.h"
#include "patArithExp.h"
#include "patArithAbs.h"
#include "patArithInt.h"
#include "patArithMax.h"
#include "patArithMin.h"
#include "patArithMod.h"
#include "patArithDeriv.h"
#include "patArithNormalRandom.h"
#include "patArithUnifRandom.h"
#include "patLinearConstraint.h"
#include "patNonLinearConstraint.h"
#include "patThreeStrings.h"
#include "patOneZhengFosgerau.h"

#undef yyFlexLexer
#define yyFlexLexer patSpecFlex
#include <FlexLexer.h>

class patSpecScanner : public patSpecFlex {

private:
                                    // filename to be scanned
  patString _filename;

public:
                                    // void ctor
  patSpecScanner()
    : patSpecFlex() {
  }
                                    // ctor with filename argument
  patSpecScanner(const patString& fname_)
    : patSpecFlex(), _filename( fname_ )  {
    //    cout << "Opening " << fname_ << endl << endl;
    ifstream *is = new ifstream( fname_.c_str() ); 
    if ( !is || (*is).fail() ) {
      WARNING("Error:: cannot open input file <" << fname_ << ">") ;
      // exit(1) ;
      return ;
    }
    else {
      switch_streams( is, 0 );
    }
  }
                                    // dtor

  ~patSpecScanner() { delete yyin; }

                                    // utility functions

  const patString& filename() const { return _filename; }

  patString removeDelimeters( const patString deli="\"\"" ) {
    
    
    patString str = YYText();

    patString::size_type carret = str.find("\n") ;
    if (carret < str.size()) str.erase(carret) ;
    carret = str.find("\r") ;
    if (carret < str.size()) str.erase(carret) ;
    patString::size_type deb = str.find( deli[0] ) ;
    if (deb == str.size()) {
      return ( str ) ;
    }
    str.erase( deb , 1 );
    
    patString::size_type fin = str.find( deli[1] ) ;
    if (fin >= str.size()) {
      WARNING(str) ;
      WARNING("Unmatched delimiters (" << filename() << ":" << 
	      lineno() << ") ") ;
      return( str ) ;
    }
    str.erase( fin , 1 );
    return ( str );
  }

  patString value() {
    patString str = YYText() ;
    return str; 
  }

  // char* value() { return (char*) YYText(); }

  void errorQuit( int err ) {
    cout << "Error = " << err << endl;
    if ( err == 0 ) return;
    WARNING("Problem in parsing"
	    << " (" << filename() << ":" << lineno() << ") "
	    << "Field: <" << YYText() << ">") ;
    if ( err < 0 ) {
      return ;
      //exit( 1 );
    }
  }
};



%}

// *** Declare here all data types that will be read by the parser

%union {
  long            itype;
  float            ftype;
  patString*       stype;
  patUtilTerm*     uttype ;
  patUtilFunction* uftype ;
  list<long>*     listshorttype ;
  patArithNode*    arithType ;
  patArithRandom*    arithRandomType ;
  list<patString>* liststringtype ;
  patConstraintTerm* cttype ;
  patConstraintEquation* cetype ;
  patLinearConstraint* lctype ;
  patListLinearConstraint* llctype ;
  patNonLinearConstraint* nlctype ;
  patListNonLinearConstraints* lnlctype ;
  patLoop*        loopType ;
  patThreeStrings*  discreteTermType ;
  vector<patThreeStrings >* discreteDistType ;
}

%token pat_gevDataFile
%token pat_gevModelDescription
%token pat_gevChoice
%token pat_gevPanel
%token pat_gevWeight
%token pat_gevBeta
%token pat_gevBoxCox
%token pat_gevBoxTukey
%token pat_gevLatex1
%token pat_gevLatex2
%token pat_gevMu
%token pat_gevSampleEnum
%token pat_gevGnuplot
%token pat_gevUtilities
%token pat_gevGeneralizedUtilities
%token pat_gevDerivatives
%token pat_gevParameterCovariances
%token pat_gevExpr
%token pat_gevGroup
%token pat_gevExclude
%token pat_gevScale
%token pat_gevModel
%token pat_gevNLNests
%token pat_gevCNLAlpha
%token pat_gevCNLNests
%token pat_gevRatios
%token pat_gevDraws
%token pat_gevConstraintNestCoef
%token pat_gevConstantProduct
%token pat_gevNetworkGEVNodes
%token pat_gevNetworkGEVLinks
%token pat_gevLinearConstraints
%token pat_gevNonLinearEqualityConstraints
%token pat_gevNonLinearInequalityConstraints
%token pat_gevLogitKernelSigmas
%token pat_gevLogitKernelFactors
%token pat_gevDiscreteDistributions
%token pat_gevSelectionBias
%token pat_gevSNP
%token pat_gevAggregateLast
%token pat_gevAggregateWeight
%token pat_gevMassAtZero
%token pat_gevOrdinalLogit
%token pat_gevRegressionModels
%token pat_gevDurationModel
%token pat_gevZhengFosgerau
%token pat_gevGeneralizedExtremeValue
%token pat_gevIIATest
%token pat_gevProbaStandardErrors

%token pat_gevBP
%token pat_gevOL
%token pat_gevMNL
%token pat_gevNL
%token pat_gevCNL
%token pat_gevNGEV
%token pat_gevNONE
%token pat_gevROOT
%token pat_gevCOLUMNS
%token pat_gevLOOP
%token pat_gevDERIV
%token pat_gevACQ
%token pat_gevSIGMA_ACQ
%token pat_gevLOG_ACQ
%token pat_gevVAL
%token pat_gevSIGMA_VAL
%token pat_gevLOG_VAL
%token pat_gevE
%token pat_gevP


%left patOR patAND
%right patEQUAL patNOTEQUAL  patLESS patLESSEQUAL patGREAT patGREATEQUAL
%left  patNOT
%left  patPLUS patMINUS
%left patMULT patDIVIDE
%right patMOD
%right patPOWER
%left  patUNARYMINUS
%right patMAX patMIN patSQRT patLOG patEXP patABS patINT


%token patEQUAL

%token patOPPAR
%token patCLPAR
%token patOPBRA
%token patCLBRA
%token patOPCUR
%token patCLCUR
%token patCOMMA
%token patCOLON
		            
%token patINTEGER
%token patREAL
%token patTIME

%token patNAME
%token patSTRING

%token patPAIR


%type <ftype>    numberParam
%type <ftype>    floatParam
%type <itype>    intParam
%type <stype>    stringParam
%type <stype>    anystringParam
%type <stype>    nameParam
%type <arithType> expression simple_expression  unary_expression binary_expression deriv_expression 

%type <arithRandomType> random_expression unirandom_expression any_random_expression  

%type <uttype>  utilTerm 
%type <uftype>  utilExpression 
%type <loopType> aloop

 %type <stype> pairParam parameter ;
%type <listshorttype> listId 
 %type <cttype> eqTerm ;
 %type <cetype> equation ;
 %type <lctype> oneConstraint ;
 %type <llctype> constraintsList ;
 %type <nlctype> oneNonLinearConstraint ;
 %type <lnlctype> nonLinearConstraintsList ;
%type <liststringtype> listOfNames ;
%type <discreteTermType> oneDiscreteTerm ;
%type <discreteDistType> listOfDiscreteTerms ;
%start everything

%%

//--------------------------------------------------------------------
// Beginning  grammer rules.
//--------------------------------------------------------------------

everything : sections {
               DEBUG_MESSAGE("Finished parsing  <"
	       << scanner.filename() << ">");
}

sections : section | 
sections section ;

section : dataFileSec
| modelDescSec 
| choiceSec 
| panelSec
| weightSec
| betaSec
| muSec
| sampleEnumSec
| gnuplotSec
| parameterCovarSec
| generalizedUtilitiesSec
| derivativesSec
| massAtZeroSec
| utilitiesSec
| exprSec
| groupSec 
| excludeSec
| scaleSec 
| modelSec 
| nlnestsSec 
| cnlnestsSec 
| cnlalphaSec
| ratiosSec
| drawsSec
| latexSec
| constraintNestSec    
| constantProductSec 
| networkGevNodeSec
| networkGevLinkSec 
| linearConstraintsSec
| nonLinearEqualityConstraintsSec
| nonLinearInequalityConstraintsSec
| discreteDistSec
| selectionBiasSec
| aggregateLastSection
| aggregateWeightSection
| SNPsection
| ordinalLogitSection
| regressionModelSection
| durationModelSection
| zhengFosgerauSection
| generalizedExtremeValueSection
| IIATestSection
| probaStandardErrorsSection

;

//////////////////////////////////////////////

dataFileSec: pat_gevDataFile dataFileColumns
;

dataFileColumns: pat_gevCOLUMNS patEQUAL intParam {
  DETAILED_MESSAGE("Section [DataFile] is now obsolete") ;
}

modelDescSec:  pat_gevModelDescription listOfNames {
  assert(pModel != NULL) ;
  pModel->setModelDescription($2) ;
}


//////////////////////////////////////////
latexSec: latexHead pat_gevNONE 
| latexHead listOfLatexNames {

};

latexHead: pat_gevLatex1 | pat_gevLatex2 ;

listOfLatexNames: latexName {
  
}
| listOfLatexNames latexName {

}

latexName: nameParam stringParam {
  pModel->addLatexName(patString(*$1),patString(*$2)) ;
}

//////////////////////////////////////////
choiceSec : pat_gevChoice expression {
  assert(pModel != NULL) ;
  assert($2 != NULL) ;
  pModel->setChoice($2) ;
}

//////////////////////////////////////////
aggregateLastSection : pat_gevAggregateLast pat_gevNONE 
|
pat_gevAggregateLast expression {
  assert(pModel != NULL) ;
  assert($2 != NULL) ;
  pModel->setAggregateLast($2) ;
}

//////////////////////////////////////////
aggregateWeightSection : pat_gevAggregateWeight pat_gevNONE 
|
pat_gevAggregateWeight expression {
  assert(pModel != NULL) ;
  assert($2 != NULL) ;
  pModel->setAggregateWeight($2) ;
}


//////////////////////////////////////////////
panelSec : pat_gevPanel pat_gevNONE {

}
| pat_gevPanel panelDescription {

};

panelDescription: expression listOfNames {
  assert(pModel != NULL) ;
  assert($1 != NULL) ;
  pModel->setPanel($1) ;
  pModel->setPanelVariables($2) ;
  DELETE_PTR($2) ;
}

listOfNames : 
{
  WARNING("Empty list of names") ;
  $$ = NULL ;
}
| anystringParam {
  $$ = new list<patString> ;
  $$->push_back(*$1) ;
}
| listOfNames anystringParam {
  assert($1 != NULL) ;
  $1->push_back(*$2) ;
  $$ = $1 ;
}


massAtZeroSec : pat_gevMassAtZero  listOfMassAtZero ;

listOfMassAtZero : pat_gevNONE 
| oneMassAtZero 
| listOfMassAtZero oneMassAtZero ;

oneMassAtZero: anystringParam floatParam {
  pModel->addMassAtZero(*$1,$2) ;
}


//////////////////////////////////////////////
weightSec : pat_gevWeight pat_gevNONE {
  DEBUG_MESSAGE("No weight defined") ;
}
|
pat_gevWeight expression {
  assert(pModel != NULL) ;
  assert($2 != NULL) ;
  pModel->setWeight($2) ;
}

//////////////////////////////////////////////
betaSec : pat_gevBeta betaList ;

betaList : oneBeta | betaList oneBeta ;

oneBeta : nameParam numberParam numberParam numberParam intParam {
  assert(pModel != NULL) ;
  assert($1 != NULL) ;
  pModel->addBeta(patString(*$1),$2,$3,$4,$5!=0) ;
  delete $1 ;
}


//////////////////////////////////////////////
muSec : pat_gevMu numberParam numberParam numberParam intParam {
  assert(pModel != NULL) ;
  pModel->setMu($2,$3,$4,$5!=0) ;
}

////////////////////////////////////////////
gnuplotSec: pat_gevGnuplot nameParam numberParam numberParam {
  assert(pModel != NULL) ;
  assert($2 != NULL) ;
  pModel->setGnuplot(patString(*$2),$3,$4) ;
  DELETE_PTR($2) ;
}
//////////////////////////////////////////////
sampleEnumSec : pat_gevSampleEnum intParam {
  assert(pModel != NULL) ;
  pModel->setSampleEnumeration($2) ;
}

//////////////////////////////////////////////
parameterCovarSec : pat_gevParameterCovariances listOfCovariances ;

listOfCovariances : pat_gevNONE 
| covariance 
| listOfCovariances covariance ;

covariance: nameParam nameParam numberParam numberParam numberParam intParam {
  assert(pModel != NULL) ;
  assert($1 != NULL) ;
  assert($2 != NULL) ;
  pModel->addCovarParam(patString(*$1),patString(*$2),$3,$4,$5,$6!=0) ;
  delete $1 ;
  delete $2 ;
}

IIATestSection: pat_gevIIATest listOfIIATests

listOfIIATests: pat_gevNONE
| oneIIATest
| listOfIIATests oneIIATest ;

oneIIATest: nameParam listId {
  assert(pModel != NULL) ;
  assert($1 != NULL) ;
  pModel->addIIATest(patString(*$1),$2) ;
}

probaStandardErrorsSection: pat_gevProbaStandardErrors listOfProbaStandardErrors

listOfProbaStandardErrors: pat_gevNONE
| oneProbaStandardError
| listOfProbaStandardErrors oneProbaStandardError ;

oneProbaStandardError: nameParam nameParam numberParam {
  assert($1 != NULL) ;
  assert($2 != NULL) ;
  patError* err(NULL) ;
  pModel->addProbaStandardError(patString(*$1),patString(*$2),$3,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    yyerror("Parsing error") ;
  }
}

zhengFosgerauSection: pat_gevZhengFosgerau listOfZheng ;

listOfZheng: pat_gevNONE
| oneZheng
| listOfZheng oneZheng;

 oneZheng:
  pat_gevP patOPCUR  nameParam patCLCUR numberParam numberParam numberParam stringParam {
   DEBUG_MESSAGE("Proba " << *$3) ;
   DEBUG_MESSAGE("Bandwith = " << $5) ;
   DEBUG_MESSAGE("lb = " << $6) ;
   DEBUG_MESSAGE("ub = " << $7) ;
   DEBUG_MESSAGE("Name = " << *$8) ;

   // Probability
   patOneZhengFosgerau aZheng($5,$6,$7,patString(*$3),patString(*$8)) ;
   patError* err(NULL) ;
   pModel->addZhengFosgerau(aZheng,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    yyerror("Parsing error") ;
  }
 }
 | pat_gevE patOPCUR expression patCLCUR numberParam numberParam numberParam stringParam {
   DEBUG_MESSAGE("Expression " << *$3) ;
   DEBUG_MESSAGE("Bandwith = " << $5) ;
   DEBUG_MESSAGE("lb = " << $6) ;
   DEBUG_MESSAGE("ub = " << $7) ;
   DEBUG_MESSAGE("Name = " << *$8) ;
   patOneZhengFosgerau aZheng($5,$6,$7,patString(*$8),$3) ;
   patError* err(NULL) ;
   pModel->addZhengFosgerau(aZheng,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    yyerror("Parsing error") ;
  }
 }

generalizedExtremeValueSection : 
pat_gevGeneralizedExtremeValue nameParam {
  pModel->setGeneralizedExtremeValueParameter(*$2) ;
}
| pat_gevGeneralizedExtremeValue pat_gevNONE

durationModelSection: pat_gevDurationModel nameParam nameParam  {
  pModel->setStartingTime(patString(*$2)) ;
  pModel->setDurationParameter(patString(*$3)) ;
  DEBUG_MESSAGE("Starting time: " << *$2) ;
  DEBUG_MESSAGE("Model parameter: " << *$3) ;
}

regressionModelSection: pat_gevRegressionModels nameParam listOfRegModels {
  pModel->setRegressionObservation(patString(*$2)) ;
  DEBUG_MESSAGE("Regression dependent: " << *$2) ;
}

listOfRegModels: oneRegMode | listOfRegModels oneRegMode ;

oneRegMode: acqRegressionModel | valRegressionModel |acqSigma | valSigma ;

acqRegressionModel:  pat_gevACQ patEQUAL utilExpression {

  assert($3 != NULL) ;
  pModel->addAcqRegressionModel($3) ;
  DELETE_PTR($3) ;
}

acqSigma: pat_gevSIGMA_ACQ patEQUAL nameParam {
  pModel->setAcqSigma(patString(*$3)) ;
}



valRegressionModel:  pat_gevVAL patEQUAL utilExpression {
  pModel->addValRegressionModel($3) ;
  DELETE_PTR($3) ;
}

valSigma: pat_gevSIGMA_VAL patEQUAL nameParam {
  pModel->setValSigma(patString(*$3)) ;
}

ordinalLogitSection: pat_gevOrdinalLogit listOfOrdinalLogit ;

listOfOrdinalLogit: pat_gevNONE
| ordinalLogit
| listOfOrdinalLogit ordinalLogit ;

ordinalLogit : intParam pat_gevNONE {
  pModel->setOrdinalLogitLeftAlternative($1) ;
} |
intParam nameParam {
  pModel->addOrdinalLogitThreshold($1,patString(*$2)) ;
};

//////////////////////////////////////////////
generalizedUtilitiesSec : pat_gevGeneralizedUtilities generalizedUtilities ; 


generalizedUtilities: pat_gevNONE 
| generalizedUtility 
| generalizedUtilities generalizedUtility ;

generalizedUtility: intParam expression {
  assert(pModel != NULL) ;
  pModel->addNonLinearUtility($1,$2) ;
}
//////////////////////////////////////////////
derivativesSec : pat_gevDerivatives derivatives ; 


derivatives: pat_gevNONE 
| oneDerivative 
| derivatives oneDerivative ;

oneDerivative: intParam nameParam expression {
  assert(pModel != NULL) ;
  assert($2 != NULL) ;
  pModel->addDerivative($1,patString(*$2),$3) ;
}



//////////////////////////////////////////////

exprSec: pat_gevExpr listExpr ;

listExpr: pat_gevNONE 
| exprDef
| listExpr exprDef ;  

exprDef: nameParam patEQUAL expression {
  assert(pModel!= NULL) ;
  assert($1 != NULL) ;
  patError* err(NULL) ;
  pModel->addExpression(patString(*$1),$3,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    yyerror("Parsing error") ;
  }
  delete $1 ;
} 
| aloop nameParam patEQUAL expression {
  assert(pModel!= NULL) ;
  assert($2 != NULL) ;
  patError* err(NULL) ;
  pModel->addExpressionLoop(patString(*$2),$4,$1,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    yyerror("Parsing error") ;
  }

  delete $2 ;
} 

aloop: pat_gevLOOP patOPCUR nameParam  intParam  intParam  intParam patCLCUR {

  patLoop* theLoop = new patLoop ;
  theLoop->variable = patString(*$3) ;
  theLoop->lower = $4 ; 
  theLoop->upper = $5 ; 
  theLoop->step = $6 ; 
  $$ = theLoop ;
}


//////////////////////////////////////////////
groupSec : pat_gevGroup pat_gevNONE {
  DEBUG_MESSAGE("No group defined") ;
  assert (pModel != NULL) ;
  pModel->setDefaultGroup() ;
//   patArithConstant* ptr = new patArithConstant(NULL) ;
//   ptr->setValue(1) ;
//   pModel->setGroup(ptr) ;
//   pModel->addScale(1,1.0,1.0,1.0,patTRUE) ;
}
| pat_gevGroup expression {
  assert(pModel != NULL) ;
  assert($2 != NULL) ;
  pModel->setGroup($2) ;
}

//////////////////////////////////////////////
excludeSec : pat_gevExclude pat_gevNONE {
  DEBUG_MESSAGE("No exclusion condition") ;
  patArithConstant* ptr = new patArithConstant(NULL) ;
  ptr->setValue(0.0) ;
  pModel->setExclude(ptr) ;
}
| pat_gevExclude expression {
  assert(pModel != NULL) ;
  assert($2 != NULL) ;
  //  DEBUG_MESSAGE("Exclude condition " << *$2) ;
  pModel->setExclude($2) ;
}

//////////////////////////////////////////////
scaleSec : pat_gevScale | pat_gevScale scaleList ;

scaleList : pat_gevNONE | oneScale | scaleList oneScale ;

oneScale : intParam numberParam numberParam numberParam intParam {
  assert (pModel != NULL) ;
  pModel->addScale($1,$2,$3,$4,$5!=0) ;
}

//////////////////////////////////////////////
modelSec : pat_gevModel modelType ;

modelType : pat_gevOL {
  DEBUG_MESSAGE("Model OL") ;
  assert (pModel != NULL) ;
  pModel->setModelType(patModelSpec::patOLtype) ;
}
| pat_gevBP {
  assert(pModel != NULL) ;
  pModel->setModelType(patModelSpec::patBPtype) ;
}
|pat_gevMNL {
  assert(pModel != NULL) ;
  pModel->setModelType(patModelSpec::patMNLtype) ;
}
| pat_gevNL {
  assert(pModel != NULL) ;
  pModel->setModelType(patModelSpec::patNLtype) ;
}
| pat_gevCNL {
  assert(pModel != NULL) ;
  pModel->setModelType(patModelSpec::patCNLtype) ;
}
| pat_gevNGEV {
  assert(pModel != NULL) ;
  pModel->setModelType(patModelSpec::patNetworkGEVtype) ;
}

//////////////////////////////////////////////

nlnestsSec : pat_gevNLNests pat_gevNONE {
} 
| pat_gevNLNests nestList ;

nestList : oneNest | nestList oneNest ;

oneNest : nameParam numberParam numberParam numberParam intParam listId  {
  assert(pModel != NULL) ;
  assert($1 != NULL) ;
  assert($6 != NULL) ;
  pModel->addNest(patString(*$1),$2,$3,$4,$5!=0,$6) ;
  delete $1 ;
  delete $6 ;
}

listId : intParam {
  $$ = new list<long> ;
  $$->push_back($1) ;
}
| listId intParam {
  assert($1 != NULL) ;
  $1->push_back($2) ;
  $$ = $1 ;
}

//////////////////////////////////////////////
cnlnestsSec : pat_gevCNLNests pat_gevNONE {
  DEBUG_MESSAGE("No nests defined for CNL model") ;
}
| pat_gevCNLNests cnlNests ;

cnlNests : oneCnlNest | cnlNests oneCnlNest ;

oneCnlNest : nameParam numberParam numberParam numberParam intParam {
  assert(pModel != NULL) ;
  assert($1 != NULL) ;
  pModel->addCNLNest(patString(*$1),$2,$3,$4,$5!=0) ;
  delete $1 ;
}

//////////////////////////////////////////////
cnlalphaSec : pat_gevCNLAlpha pat_gevNONE {
  DEBUG_MESSAGE("No alpha defined for CNL model") ;
}
| pat_gevCNLAlpha cnlAlphas ;

cnlAlphas : oneAlpha | cnlAlphas oneAlpha;

oneAlpha : nameParam nameParam numberParam numberParam numberParam intParam {
  assert(pModel != NULL) ;
  assert($1 != NULL) ;
  assert($2 != NULL) ;
  pModel->addCNLAlpha(patString(*$1),patString(*$2),$3,$4,$5,$6!=0) ;
  delete $1 ;
  delete $2 ;
  static int count = 0 ;
  ++count ;
}

//////////////////////////////////////////////
ratiosSec : pat_gevRatios ratiosList ;

ratiosList : pat_gevNONE | oneRatio | ratiosList oneRatio ;

oneRatio : nameParam nameParam nameParam {
  assert (pModel != NULL) ;
  assert ($1 != NULL) ;
  assert ($2 != NULL) ;
  assert ($3 != NULL) ;
  pModel->addRatio(patString(*$1),patString(*$2),patString(*$3)) ;
  delete $1 ;
  delete $2 ;
  delete $3 ;
}
//////////////////////////////////////////////
drawsSec: pat_gevDraws intParam {
  pModel->setNumberOfDraws($2) ;
}
//////////////////////////////////////////////
linearConstraintsSec : pat_gevLinearConstraints constraintsList {
  pModel->setListLinearConstraints($2) ;
  DELETE_PTR($2) ;
}

//////////////////////////////////////////////
nonLinearEqualityConstraintsSec : pat_gevNonLinearEqualityConstraints nonLinearConstraintsList {
  pModel->setListNonLinearEqualityConstraints($2) ;
}

//////////////////////////////////////////////
nonLinearInequalityConstraintsSec : pat_gevNonLinearInequalityConstraints nonLinearConstraintsList {
  pModel->setListNonLinearInequalityConstraints($2) ;
}


nonLinearConstraintsList : pat_gevNONE {
  patListNonLinearConstraints* ptr = NULL ;
  $$ = ptr ;
}
| oneNonLinearConstraint {
  patListNonLinearConstraints* ptr = new patListNonLinearConstraints;
  ptr->push_back(*$1) ;
  DELETE_PTR($1) ;
  $$ = ptr ;
}
| nonLinearConstraintsList oneNonLinearConstraint {
  $1->push_back(*$2) ;
  DELETE_PTR($2) ;
  $$ = $1 ;
}

oneNonLinearConstraint : expression {
  patNonLinearConstraint* ptr = new patNonLinearConstraint($1) ;
  $$ = ptr ;
}

constraintsList : pat_gevNONE {
  patListLinearConstraint* ptr = NULL ;
  $$ = ptr ;
}
| oneConstraint {
  patListLinearConstraint* ptr = new patListLinearConstraint ;
  ptr->push_back(*$1) ;
  DELETE_PTR($1) ;
  $$ = ptr ;
}
| constraintsList oneConstraint {
  $1->push_back(*$2) ;
  DELETE_PTR($2) ;
  $$ = $1 ;
}

oneConstraint :
equation patLESSEQUAL numberParam {  
  patLinearConstraint* ptr = new patLinearConstraint ;
  ptr->theEquation = *$1 ;
  ptr->theType = patLinearConstraint::patLESSEQUAL ;
  ptr->theRHS = $3 ;
  DEBUG_MESSAGE(*ptr) ;
  DELETE_PTR($1) ;
  $$ = ptr ;
}
| equation patEQUAL numberParam  {
  patLinearConstraint* ptr = new patLinearConstraint ;
  ptr->theEquation = *$1 ;
  ptr->theType = patLinearConstraint::patEQUAL ;
  ptr->theRHS = $3 ;
  DELETE_PTR($1) ;
  $$ = ptr ;
}
| equation patGREATEQUAL numberParam  {
  patLinearConstraint* ptr = new patLinearConstraint ;
  ptr->theEquation = *$1 ;
  ptr->theType = patLinearConstraint::patGREATEQUAL ;
  ptr->theRHS = $3 ;
  DELETE_PTR($1) ;
  DEBUG_MESSAGE(*ptr) ;
  $$ = ptr ;
}


eqType : patLESSEQUAL | patGREATEQUAL | patEQUAL ; 

equation: eqTerm {
  patConstraintEquation* ptr = new patConstraintEquation ;
  ptr->push_back(*$1) ;
  DELETE_PTR($1) ;
  $$ = ptr ;
}
|  patMINUS eqTerm {
  patConstraintEquation* ptr = new patConstraintEquation ;
  $2->fact = - $2->fact ;
  ptr->push_back(*$2) ;
  DELETE_PTR($2) ;
  $$ = ptr ;
}

| equation patPLUS eqTerm  {
  $1->push_back(*$3) ;
  DELETE_PTR($3) ;
  $$ = $1 ;
}
| equation patMINUS eqTerm {
  $3->fact = - $3->fact ;
  $1->push_back(*$3) ;
  DELETE_PTR($3) ;
  $$ = $1 ;
}
eqTerm: parameter {
  patConstraintTerm* ptr = new patConstraintTerm ;
  ptr->fact = 1.0 ;
  ptr->param = *$1 ;
  DELETE_PTR($1) ;
  $$ = ptr ;
}
| numberParam patMULT parameter {
  patConstraintTerm* ptr = new patConstraintTerm ;
  ptr->fact = $1 ;
  ptr->param = *$3 ;
  DELETE_PTR($3) ;
  $$ = ptr ;
}

parameter : nameParam {
  $$ = $1 ;
}
| pairParam {
  $$ = $1 ;
};

pairParam : patOPPAR nameParam patCOMMA nameParam patCLPAR {
  patString* ptr = new patString(pModel->buildLinkName(*$2,*$4)) ;
  DELETE_PTR($2) ;
  DELETE_PTR($4) ;
  $$ = ptr ;
}; 

//////////////////////////////////////////////
networkGevNodeSec : pat_gevNetworkGEVNodes networkGevNodeList ;
networkGevNodeList : pat_gevNONE | oneNetworkGevNode | networkGevNodeList oneNetworkGevNode ;
oneNetworkGevNode :  nameParam numberParam numberParam numberParam intParam {
  assert(pModel != NULL) ;
  assert($1 != NULL) ;
  pModel->addNetworkGevNode(patString(*$1),$2,$3,$4,$5!=0) ;
  delete $1 ;
}

//////////////////////////////////////////////
networkGevLinkSec : pat_gevNetworkGEVLinks  networkGevLinkList ;
networkGevLinkList :  pat_gevNONE | oneNetworkGevLink | networkGevLinkList oneNetworkGevLink ;
oneNetworkGevLink :  nameParam nameParam numberParam numberParam numberParam intParam {
  assert(pModel != NULL) ;
  assert($1 != NULL) ;
  assert($2 != NULL) ;

  pModel->addNetworkGevLink(patString(*$1),patString(*$2),$3,$4,$5,$6!=0) ;
  delete $1 ;
  delete $2 ;
}

//////////////////////////////////////////////
constantProductSec : pat_gevConstantProduct constProdList ;

constProdList: pat_gevNONE | oneConstProd | constProdList oneConstProd ;

oneConstProd: nameParam nameParam numberParam {
  assert(pModel != NULL) ;
  assert($1 != NULL) ;
  assert($2 != NULL) ;
  stringstream str ;
  str << "Syntax error on line " <<  scanner.lineno() << endl ;
  str << "Section [ConstantProduct] is obsolete. " ; 
  str << "Add the following line in section [NonLinearEqualityConstraints] instead: " ;
  str << *$1 << "*" << *$2 << "-" << $3 ;

  
  //pModel->addConstantProduct(patString(*$1),patString(*$2),$3) ;
  delete $1 ;
  delete $2 ;
  pModel->syntaxError = new patErrMiscError(str.str());
  //  exit(-1) ;

}

//////////////////////////////////////////////
constraintNestSec : pat_gevConstraintNestCoef constraintNestList ;

constraintNestList: pat_gevNONE | oneConstraintNest | constraintNestList oneConstraintNest ;

oneConstraintNest : nameParam patEQUAL nameParam {
  assert(pModel != NULL) ;
  assert($1 != NULL) ;
  assert($3 != NULL) ;
  DEBUG_MESSAGE("Constraining nest parameters for " << *$1 << " and " << *$3) ;
  pModel->addConstraintNest(patString(*$1),patString(*$3)) ;
  delete $1 ;
  delete $3 ;
};


////////////////////////

SNPsection: pat_gevSNP defSnpTerms ;

defSnpTerms: pat_gevNONE | nonEmptySnpTerms ;

nonEmptySnpTerms: nameParam listOfSnpTerms {
  assert($1 != NULL) ;
  pModel->setSnpBaseParameter(patString(*$1)) ;
  delete $1 ;
}
 
listOfSnpTerms: oneSnpTerm  | listOfSnpTerms oneSnpTerm ;

oneSnpTerm: intParam nameParam {
  assert($2 != NULL) ;
  pModel->addSnpTerm($1,patString(*$2)) ;
  delete $2 ;

}

////////////////////////////////////////////
selectionBiasSec: pat_gevSelectionBias listOfSelectionBias ;

listOfSelectionBias: pat_gevNONE | oneSelectionBias | listOfSelectionBias oneSelectionBias ;

oneSelectionBias: intParam nameParam {
  assert($2 != NULL) ;
  pModel->addSelectionBiasParameter($1,*$2) ;
}
////////////////////////////////////////////
discreteDistSec : pat_gevDiscreteDistributions listOfDiscreteParameters ;

listOfDiscreteParameters : pat_gevNONE | oneDiscreteParameter | listOfDiscreteParameters oneDiscreteParameter ;

oneDiscreteParameter: nameParam patLESS listOfDiscreteTerms patGREAT {
  patError* err(NULL) ;
  pModel->addDiscreteParameter(*$1,*$3,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    yyerror("Parsing error") ;
  }
  DELETE_PTR($3) ;
  DEBUG_MESSAGE("Identified discrete param " << *$1) ;
  
}

listOfDiscreteTerms : oneDiscreteTerm {
  vector<patThreeStrings>* ptr = new vector<patThreeStrings> ;
  ptr->push_back(*$1) ;
  $$ = ptr ;
}| listOfDiscreteTerms oneDiscreteTerm {
  assert($1 != NULL) ;
  $1->push_back(*$2) ;
  $$ = $1 ;
}

oneDiscreteTerm : nameParam patOPPAR nameParam patCLPAR {
  $$= new patThreeStrings(*$1,patString(),*$3) ;
  DEBUG_MESSAGE("-> discrete term " << *$1 << "(" << *$3 << ")") ;
}
| any_random_expression patOPPAR nameParam patCLPAR {
  $$= new patThreeStrings($1->getLocationParameter(),
			  $1->getScaleParameter(),
			  *$3) ;
  DEBUG_MESSAGE("-> discrete term " << $1->getLocationParameter() << "[" <<  $1->getScaleParameter() << "]" << "(" << *$3 << ")") ;
}

//////////////////////////////////////////////

utilitiesSec : pat_gevUtilities utilList ;

utilList : util | utilList util ;

util : intParam nameParam nameParam utilExpression {
  assert(pModel != NULL) ;
  assert($2 != NULL) ;
  assert($3 != NULL) ;
  assert($4 != NULL) ;
   patError* err(NULL) ;
   pModel->addUtil($1,patString(*$2),patString(*$3),$4,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    yyerror("Parsing error") ;
  }
  delete $2 ;
  delete $3 ;
  delete $4 ;
}

utilExpression : pat_gevNONE {
  $$ = new patUtilFunction ;
}
| utilTerm {
  assert ( $1 != NULL) ;
  $$ = new patUtilFunction ;
  $$->push_back(*$1) ;
  delete $1 ;
}
| utilExpression patPLUS utilTerm {
  
  assert($1 != NULL) ;
  assert($3 != NULL) ;
  $1->push_back(*$3) ;
  delete $3 ;
  $$ = $1 ;
}



utilTerm : nameParam patMULT nameParam {
  assert($1 != NULL) ;
  assert($3 != NULL) ;
  patUtilTerm* term = new patUtilTerm ;
  term->beta = patString(*$1) ;
  term->x = patString(*$3) ;
  term->random = patFALSE ;
  assert(pModel != NULL) ;
  pModel->addAttribute(*$3) ;
  delete $1 ;
  delete $3 ;
  $$ = term ;
} 
| any_random_expression patMULT nameParam {
  assert($1 != NULL) ;
  assert($3 != NULL) ;
  patUtilTerm* term = new patUtilTerm ;
  term->beta = $1->getOperatorName() ;
  term->betaIndex = patBadId ;
  term->randomParameter = $1 ;
  term->x = patString(*$3) ;
  term->random = patTRUE ;
  assert(pModel != NULL) ;
  pModel->addAttribute(*$3) ;
  delete $3 ;
  $$ = term ;
}

expression : simple_expression {
  $$ = $1 ;
}
| unary_expression {
  $$ = $1 ;
}
| binary_expression {
  $$ = $1 ;
}
| any_random_expression {
  $$ = $1 ;
}
| patOPPAR expression patCLPAR {
  $$ = $2 ;
} 
| deriv_expression {
  $$ = $1 ;
} ;


any_random_expression : random_expression {
  $$ = $1 ;
}
| unirandom_expression {
  $$ = $1 ;
}

deriv_expression: pat_gevDERIV patOPPAR expression patCOMMA nameParam patCLPAR {
  assert($3 != NULL) ;
  assert($5 != NULL) ;
  patArithDeriv* ptr = new patArithDeriv(NULL,$3,patString(*$5)) ;
  assert(ptr != NULL) ;
  $3->setParent(ptr) ;
  delete $5 ;
  $$ = ptr ;
}

random_expression: nameParam patOPBRA nameParam patCLBRA {
  //  pModel->addDrawHeader(*$1,*$3);
  patArithNormalRandom* ptr = new patArithNormalRandom(NULL) ;
  assert(ptr != NULL) ;
  ptr->setLocationParameter(*$1) ;
  ptr->setScaleParameter(*$3) ;
  patArithRandom* oldParam = pModel->addRandomExpression(ptr) ;
  delete $1 ;
  delete $3 ;
  if (oldParam == NULL) {
    $$ = ptr ;
  }
  else {
    delete ptr ;
    $$ = oldParam ;
  }
}

unirandom_expression: nameParam patOPCUR nameParam patCLCUR {
  //  pModel->addDrawHeader(*$1,*$3);
  patArithUnifRandom* ptr = new patArithUnifRandom(NULL) ;
  assert(ptr != NULL) ;
  ptr->setLocationParameter(*$1) ;
  ptr->setScaleParameter(*$3) ;
  patArithRandom* oldParam = pModel->addRandomExpression(ptr) ;
  delete $1 ;
  delete $3 ;
  if (oldParam == NULL) {
    $$ = ptr ;
  }
  else {
    delete ptr ;
    $$ = oldParam ;
  }
}



unary_expression: patMINUS expression %prec patUNARYMINUS {
  assert($2 != NULL) ;
  patArithUnaryMinus* ptr = new patArithUnaryMinus(NULL,$2) ;
  assert(ptr != NULL) ;
  $2->setParent(ptr) ;
  $$ = ptr ;
    
}
| patNOT expression {
  assert($2 != NULL) ;
  patArithNot* ptr = new patArithNot(NULL,$2) ;
  assert(ptr != NULL) ;
  $2->setParent(ptr) ;
  $$ = ptr ;

}
| patSQRT patOPPAR expression patCLPAR {
  assert($3 != NULL) ;
  patArithSqrt* ptr = new patArithSqrt(NULL,$3) ;
  assert(ptr != NULL) ;
  $3->setParent(ptr) ;
  $$ = ptr ;
}
| patLOG patOPPAR expression patCLPAR {
  assert($3 != NULL) ;
  patArithLog* ptr = new patArithLog(NULL,$3) ;
  assert(ptr != NULL) ;
  $3->setParent(ptr) ;
  $$ = ptr ;

}
| patEXP patOPPAR expression patCLPAR {
  assert($3 != NULL) ;
  patArithExp* ptr = new patArithExp(NULL,$3) ;
  assert(ptr != NULL) ;
  $3->setParent(ptr) ;
  $$ = ptr ;
}
| patABS patOPPAR expression patCLPAR {
  assert($3 != NULL) ;
  patArithAbs* ptr = new patArithAbs(NULL,$3) ;
  assert(ptr != NULL) ;
  $3->setParent(ptr) ;
  $$ = ptr ;

}
| patINT patOPPAR expression patCLPAR {
  assert($3 != NULL) ;
  patArithInt* ptr = new patArithInt(NULL,$3) ;
  assert(ptr != NULL) ;
  $3->setParent(ptr) ;
  $$ = ptr ;
}
| patOPPAR unary_expression patCLPAR {
  $$ = $2;  
}


binary_expression: expression patPLUS expression {
  assert($1 != NULL) ;
  assert($3 != NULL) ;
  patArithBinaryPlus* ptr = new patArithBinaryPlus(NULL,$1,$3) ;
  assert(ptr != NULL) ;
  $1->setParent(ptr) ;
  $3->setParent(ptr) ;
  $$ = ptr ;
}
|  expression patMINUS expression {
  patArithBinaryMinus* ptr = new patArithBinaryMinus(NULL,$1,$3) ;
  assert(ptr != NULL) ;
  assert($1 != NULL) ;
  assert($3 != NULL) ;
  $1->setParent(ptr) ;
  $3->setParent(ptr) ;
  $$ = ptr ;

}
| expression patMULT expression {
  patArithMult* ptr = new patArithMult(NULL,$1,$3) ;
  assert(ptr != NULL) ;
  assert($1 != NULL) ;
  assert($3 != NULL) ;
  $1->setParent(ptr) ;
  $3->setParent(ptr) ;
  $$ = ptr ;

}
| expression patDIVIDE expression {
  patArithDivide* ptr = new patArithDivide(NULL,$1,$3) ;
  assert(ptr != NULL) ;
  assert($1 != NULL) ;
  assert($3 != NULL) ;
  $1->setParent(ptr) ;
  $3->setParent(ptr) ;
  $$ = ptr ;
}
| expression patPOWER expression {
  patArithPower* ptr = new patArithPower(NULL,$1,$3) ;
  assert(ptr != NULL) ;
  assert($1 != NULL) ;
  assert($3 != NULL) ;
  $1->setParent(ptr) ;
  $3->setParent(ptr) ;
  $$ = ptr ;
}
| expression patEQUAL expression {
  patArithEqual* ptr = new patArithEqual(NULL,$1,$3) ;
  assert(ptr != NULL) ;
  assert($1 != NULL) ;
  assert($3 != NULL) ;
  $1->setParent(ptr) ;
  $3->setParent(ptr) ;
  $$ = ptr ;

}
| expression patNOTEQUAL expression {
  patArithNotEqual* ptr = new patArithNotEqual(NULL,$1,$3) ;
  assert(ptr != NULL) ;
  assert($1 != NULL) ;
  assert($3 != NULL) ;
  $1->setParent(ptr) ;
  $3->setParent(ptr) ;
  $$ = ptr ;

}
| expression patOR expression {
  patArithOr* ptr = new patArithOr(NULL,$1,$3) ;
  assert(ptr != NULL) ;
  assert($1 != NULL) ;
  assert($3 != NULL) ;
  $1->setParent(ptr) ;
  $3->setParent(ptr) ;
  $$ = ptr ;

}
| expression patAND expression {
  patArithAnd* ptr = new patArithAnd(NULL,$1,$3) ;
  assert(ptr != NULL) ;
  assert($1 != NULL) ;
  assert($3 != NULL) ;
  $1->setParent(ptr) ;
  $3->setParent(ptr) ;
  $$ = ptr ;

}
| expression patLESS expression {
  patArithLess* ptr = new patArithLess(NULL,$1,$3) ;
  assert(ptr != NULL) ;
  assert($1 != NULL) ;
  assert($3 != NULL) ;
  $1->setParent(ptr) ;
  $3->setParent(ptr) ;
  $$ = ptr ;

}
| expression patLESSEQUAL expression {
  patArithLessEqual* ptr = new patArithLessEqual(NULL,$1,$3) ;
  assert(ptr != NULL) ;
  assert($1 != NULL) ;
  assert($3 != NULL) ;
  $1->setParent(ptr) ;
  $3->setParent(ptr) ;
  $$ = ptr ;

}
| expression patGREAT expression {
  patArithGreater* ptr = new patArithGreater(NULL,$1,$3) ;
  assert(ptr != NULL) ;
  assert($1 != NULL) ;
  assert($3 != NULL) ;
  $1->setParent(ptr) ;
  $3->setParent(ptr) ;
  $$ = ptr ;

}
| expression patGREATEQUAL expression {
  patArithGreaterEqual* ptr = new patArithGreaterEqual(NULL,$1,$3) ;
  assert(ptr != NULL) ;
  assert($1 != NULL) ;
  assert($3 != NULL) ;
  $1->setParent(ptr) ;
  $3->setParent(ptr) ;
  $$ = ptr ;

}
| patMAX patOPPAR expression patCOMMA expression patCLPAR {
  patArithMax* ptr = new patArithMax(NULL,$3,$5) ;
  assert(ptr != NULL) ;
  assert($3 != NULL) ;
  assert($5 != NULL) ;
  $3->setParent(ptr) ;
  $5->setParent(ptr) ;
  $$ = ptr ;

}
| patMIN patOPPAR expression patCOMMA expression patCLPAR {
  patArithMin* ptr = new patArithMin(NULL,$3,$5) ;
  assert(ptr != NULL) ;
  assert($3 != NULL) ;
  assert($5 != NULL) ;
  $3->setParent(ptr) ;
  $5->setParent(ptr) ;
  $$ = ptr ;

}
| expression patMOD expression {
  patArithMod* ptr = new patArithMod(NULL,$1,$3) ;
  assert(ptr != NULL) ;
  assert($1 != NULL) ;
  assert($3 != NULL) ;
  $1->setParent(ptr) ;
  $3->setParent(ptr) ;
  $$ = ptr ;
  
}
| patOPPAR binary_expression patCLPAR {
  $$ = $2;  
}

simple_expression: numberParam {
  patArithConstant* ptr = new patArithConstant(NULL) ;
  assert(ptr != NULL) ;
  ptr->setValue($1) ;
  $$= ptr ;
}
| nameParam {
  assert($1 != NULL);
  patArithVariable* ptr = new patArithVariable(NULL) ;
  assert(ptr != NULL) ;
  ptr->setName(*$1) ;
  delete $1 ;
  $$ = ptr ;
}

numberParam: floatParam {
  $$ = $1 ;
}
| intParam {
  $$ = float($1) ;
};

anystringParam: stringParam {
  $$ = $1 ;
}
| nameParam {
  $$ = $1 ;
}

stringParam : patSTRING {
  patString* str = new patString((scanner.removeDelimeters()));
  $$ = str ;
}

nameParam : patNAME {
  patString* str = new patString(scanner.value());
  //Remove the last character which is [ \t\n]
  str->erase(str->end()-1) ;
  $$ = str ;
}

floatParam : patREAL {
  $$ = atof( scanner.value().c_str() );
}
;

intParam: patINTEGER {
  $$ = atoi( scanner.value().c_str() );
}
;
//--------------------------------------------------------------------
// End of basic grammer rules
//--------------------------------------------------------------------

%%

%header{
//--------------------------------------------------------------------
// Following pieces of code will be verbosely copied into the parser.
//--------------------------------------------------------------------

class patSpecParser: public patBisonSpec {

public:
                                    // ctor with filename argument

  patSpecParser( const patString& fname_ ) :	
    patBisonSpec( fname_.c_str() ) {}
  
                                    // dtor
  virtual ~patSpecParser () { }
                                    // Utility functions

  patString filename() const { return scanner.filename(); }

  void yyerror( char* msg ) {
    stringstream str ;
    str << "Syntax error in file [" << filename() << "] at line " << scanner.lineno() << endl  ;
    str << "Unidentified token: <" << scanner.YYText() << ">" << endl ;
    str << "Please check the syntax. A common mistake is to forget the" << endl ;
    str << "mandatory blank space terminating each variable name." ; ;
    pModel->syntaxError = new patErrMiscError(str.str()) ;
    //    exit( 1 );
  }

  int yylex() { return scanner.yylex(); }

  patBoolean parse( patModelSpec *p) {
    if ( p && pModel)  {
      WARNING("\nError:: cannot parse <" << filename() << "> twice");
      return( patFALSE );
    }
    else {
      pModel = p ;
      DEBUG_MESSAGE("About to parse") ;
      yyparse();
      return(patTRUE);
    }
  }
};

 %}

