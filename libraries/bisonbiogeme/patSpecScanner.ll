/*-*-c++-*------------------------------------------------------------
//
// File name : patSpecScanner.l
// Michel Bierlaire, EPFL
// Date :      Tue Nov  7 14:22:36 2000
//
//--------------------------------------------------------------------
*/

%option noyywrap
%option yyclass="patSpecFlex"
%option yylineno

%{

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patSpecParser.hh"
#include <string>
#define YY_BREAK

%}


pat_gevDataFile    ("[DataFile]")
pat_gevModelDescription   ("[ModelDescription]")
pat_gevChoice    ("[Choice]")
pat_gevPanel     ("[PanelData]")
pat_gevWeight    ("[Weight]")
pat_gevBeta    ("[Beta]")
pat_gevLatex1    ("[LaTeX]")
pat_gevLatex2    ("[Latex]")
pat_gevBoxCox    ("[Box-Cox]")
pat_gevBoxTukey    ("[Box-Tukey]")
pat_gevMu    ("[Mu]")
pat_gevSampleEnum    ("[SampleEnum]")
pat_gevGnuplot    ("[Gnuplot]")
pat_gevUtilities ("[Utilities]")
pat_gevGeneralizedUtilities ("[GeneralizedUtilities]")
pat_gevDerivatives ("[Derivatives]")
pat_gevParameterCovariances ("[ParameterCovariances]")
pat_gevGroup    ("[Group]")
pat_gevExclude     ("[Exclude]") 
pat_gevExpr    ("[Expressions]")
pat_gevScale    ("[Scale]")
pat_gevModel    ("[Model]")
pat_gevNLNests    ("[NLNests]")
pat_gevCNLAlpha    ("[CNLAlpha]")
pat_gevCNLNests    ("[CNLNests]")
pat_gevRatios      ("[Ratios]")
pat_gevDraws      ("[Draws]")
pat_gevMassAtZero ("[MassAtZero]")
pat_gevConstraintNestCoef ("[ConstraintNestCoef]")
pat_gevConstantProduct ("[ConstantProduct]") 
pat_gevNetworkGEVNodes ("[NetworkGEVNodes]")
pat_gevNetworkGEVLinks ("[NetworkGEVLinks]") 
pat_gevLinearConstraints ("[LinearConstraints]")
pat_gevNonLinearEqualityConstraints ("[NonLinearEqualityConstraints]")
pat_gevNonLinearInequalityConstraints ("[NonLinearInequalityConstraints]")
pat_gevLogitKernelSigmas ("[LogitKernelSigmas]") 
pat_gevLogitKernelFactors ("[LogitKernelFactors]")
pat_gevDiscreteDistributions ("[DiscreteDistributions]")
pat_gevSelectionBias ("[SelectionBias]")
pat_gevSNP ("[SNP]")
pat_gevAggregateLast ("[AggregateLast]")
pat_gevAggregateWeight ("[AggregateWeight]")
pat_gevOrdinalLogit  ("[OrdinalLogit]")
pat_gevRegressionModels  ("[RegressionModels]")
pat_gevDurationModel  ("[DurationModelGamma]")
pat_gevZhengFosgerau ("[ZhengFosgerau]")
pat_gevGeneralizedExtremeValue("[GeneralizedExtremeValue]")
pat_gevIIATest ("[IIATest]")
pat_gevProbaStandardErrors ("[ProbaStandardErrors]")
pat_gevBP ("$BP")
pat_gevOL ("$OL")
pat_gevMNL ("$MNL")
pat_gevNL ("$NL")
pat_gevCNL ("$CNL")
pat_gevNGEV ("$NGEV")
pat_gevNONE ("$NONE")
pat_gevBEGIN ("$BEGIN")
pat_gevEND ("$END")
pat_gevROOT ("$ROOT")
pat_gevCOLUMNS ("$COLUMNS")
pat_gevLOOP ("$LOOP")
pat_gevDERIV ("$DERIV")
pat_gevACQ ("$ACQ")
pat_gevSIGMA_ACQ ("$SIGMA_ACQ")
pat_gevLOG_ACQ ("$LOG_ACQ")
pat_gevVAL ("$VAL")
pat_gevSIGMA_VAL ("$SIGMA_VAL")
pat_gevLOG_VAL ("$LOG_VAL")
pat_gevE ("$E") 
pat_gevP ("$P") 

C	(#.*)|("//".*)
WS	[\t ;\n\r]+

D	[0-9]
H       [0-9A-Fa-f]
E	[eE][-+]?{D}+
N	{D}+
X       {H}+
F1	{D}+\.{D}*({E})?
F2	{D}*\.{D}+({E})?
F3	{D}+({E})?

I	[-+]?({N})
R	[-+]?({F1}|{F2}|{F3})

	   //STR	["]([^"^\n^\r]*)[\n"]
STR	["]([^"]*)["]
NAME	[A-Za-z_]([A-Za-z0-9_\-]*)[ \t\n\r]

MULT       ("\*")
PLUS       ("+")
MINUS      ("-")
DIVIDE     ("/")
POWER      ("^")|("**")
MAX        ("max")|("MAX")
MIN        ("min")|("MIN")
MOD        ("mod")|("MOD")|("%")
SQRT       ("sqrt")|("SQRT")
LOG        ("log")|("LOG")|("ln")|("LN")
EXP        ("exp")|("EXP")
ABS        ("abs")|("ABS")
INT        ("int")|("INT")
EQUAL      ("=")|("==")
NOTEQUAL   ("<>")|("!=")
NOT        ("!")
OR         ("||")|("OR")
AND        ("&&")|("AND")
LESS       ("<")
LESSEQUAL  ("<=")
GREAT      (">")
GREATEQUAL (">=")


HH	(([01]?[0-9])|([2][0-3]))
MM	([0-5]?[0-9])
SS	([0-5]?[0-9])
TIME	{HH}[:]{MM}[:]{SS}

PAIR	{N}-{N}

OB	{WS}?[{]{WS}?
CB	{WS}?[}]{WS}?

COMMA   ","
COLON   ":"
OPPAR   "("
CLPAR   ")"

OPCUR   "{"
CLCUR   "}"

OPBRA   "["
CLBRA   "]"

%%

"/*"	{	/* skip comments */
  int c;
  while ( (c = yyinput()) != 0 ) {
    if ( c == '*' ) {
      if ( (c = yyinput()) == '/' ) break;
      else unput( c );
    }
  }
  break;
}

{WS}	{ break; }
{C}	{ break; }


{pat_gevDataFile} {return patSpecParser::pat_gevDataFile ;}
{pat_gevModelDescription} {return patSpecParser::pat_gevModelDescription ;}
{pat_gevChoice} {return patSpecParser::pat_gevChoice ;}
{pat_gevPanel} {return patSpecParser::pat_gevPanel ;}
{pat_gevWeight} {return patSpecParser::pat_gevWeight ;}
{pat_gevBeta} {return patSpecParser::pat_gevBeta ;}
{pat_gevBoxCox} {return patSpecParser::pat_gevBoxCox ;}
{pat_gevBoxTukey} {return patSpecParser::pat_gevBoxTukey ;}
{pat_gevLatex1} {return patSpecParser::pat_gevLatex1 ;}
{pat_gevLatex2} {return patSpecParser::pat_gevLatex2 ;}
{pat_gevMu} {return patSpecParser::pat_gevMu ;}
{pat_gevSampleEnum} {return patSpecParser::pat_gevSampleEnum ;}
{pat_gevGnuplot} {return patSpecParser::pat_gevGnuplot ;}
{pat_gevUtilities} {return patSpecParser::pat_gevUtilities ;}
{pat_gevGeneralizedUtilities} {return patSpecParser::pat_gevGeneralizedUtilities ;}
{pat_gevDerivatives} {return patSpecParser::pat_gevDerivatives ;}
{pat_gevParameterCovariances} {return patSpecParser::pat_gevParameterCovariances ;}
{pat_gevExpr} {return patSpecParser::pat_gevExpr ;}
{pat_gevGroup} {return patSpecParser::pat_gevGroup ;}
{pat_gevExclude} {return patSpecParser::pat_gevExclude ;}
{pat_gevScale} {return patSpecParser::pat_gevScale ;}
{pat_gevModel} { return patSpecParser::pat_gevModel ;}
{pat_gevNLNests} {return patSpecParser::pat_gevNLNests ;}
{pat_gevCNLAlpha} {return patSpecParser::pat_gevCNLAlpha ;}
{pat_gevCNLNests} {return patSpecParser::pat_gevCNLNests ;}
{pat_gevRatios} {return patSpecParser::pat_gevRatios ;}
{pat_gevDraws} {return patSpecParser::pat_gevDraws ;}
{pat_gevConstraintNestCoef} {return patSpecParser::pat_gevConstraintNestCoef ;}
{pat_gevConstantProduct} {return patSpecParser::pat_gevConstantProduct ; }
{pat_gevNetworkGEVNodes} {return patSpecParser::pat_gevNetworkGEVNodes ; }
{pat_gevNetworkGEVLinks} {return patSpecParser::pat_gevNetworkGEVLinks ; }
{pat_gevLinearConstraints} {return patSpecParser::pat_gevLinearConstraints ;}
{pat_gevNonLinearEqualityConstraints} {return patSpecParser::pat_gevNonLinearEqualityConstraints ;}
{pat_gevNonLinearInequalityConstraints} {return patSpecParser::pat_gevNonLinearInequalityConstraints ;}
{pat_gevBP} {return patSpecParser::pat_gevBP ;}
{pat_gevOL} {return patSpecParser::pat_gevOL ;}
{pat_gevMNL} {return patSpecParser::pat_gevMNL ;}
{pat_gevP} {return patSpecParser::pat_gevP ;}
{pat_gevE} {return patSpecParser::pat_gevE ;}
{pat_gevNL} { return patSpecParser::pat_gevNL; }
{pat_gevCNL} {return patSpecParser::pat_gevCNL ;}
{pat_gevNGEV} {return patSpecParser::pat_gevNGEV ;}
{pat_gevNONE} {return patSpecParser::pat_gevNONE ;}
{pat_gevCOLUMNS} {return patSpecParser::pat_gevCOLUMNS ;}
{pat_gevLOOP} {return patSpecParser::pat_gevLOOP ;}
{pat_gevACQ} {return patSpecParser::pat_gevACQ ;}
{pat_gevSIGMA_ACQ} {return patSpecParser::pat_gevSIGMA_ACQ ;}
{pat_gevVAL} {return patSpecParser::pat_gevVAL ;}
{pat_gevSIGMA_VAL} {return patSpecParser::pat_gevSIGMA_VAL ;}
{pat_gevDERIV} {return patSpecParser::pat_gevDERIV ;}
{pat_gevMassAtZero} {return patSpecParser::pat_gevMassAtZero ;}
{pat_gevDiscreteDistributions} {return patSpecParser::pat_gevDiscreteDistributions ;}
{pat_gevSelectionBias} {return patSpecParser::pat_gevSelectionBias ;}
{pat_gevSNP} {return patSpecParser::pat_gevSNP ;}
{pat_gevAggregateLast} {return patSpecParser::pat_gevAggregateLast ;}
{pat_gevAggregateWeight} {return patSpecParser::pat_gevAggregateWeight ;}
{pat_gevOrdinalLogit} {return patSpecParser::pat_gevOrdinalLogit ;}
{pat_gevRegressionModels} {return patSpecParser::pat_gevRegressionModels ;}
{pat_gevDurationModel} {return patSpecParser::pat_gevDurationModel ;}
{pat_gevZhengFosgerau} {return patSpecParser::pat_gevZhengFosgerau ;}
{pat_gevGeneralizedExtremeValue} {return patSpecParser::pat_gevGeneralizedExtremeValue ;}
{pat_gevIIATest} {return patSpecParser::pat_gevIIATest ;}
{pat_gevProbaStandardErrors} {return patSpecParser::pat_gevProbaStandardErrors ;}

{OPCUR} {return patSpecParser::patOPCUR ;}
{CLCUR} {return patSpecParser::patCLCUR ;}
{OPBRA} {return patSpecParser::patOPBRA ;}
{CLBRA} {return patSpecParser::patCLBRA ;}
{OPPAR} {return patSpecParser::patOPPAR ;}
{CLPAR} {return patSpecParser::patCLPAR ;}
{COMMA}	{return patSpecParser::patCOMMA ;}
{COLON}	{return patSpecParser::patCOLON ;}
	            
{I}	{ return patSpecParser::patINTEGER; }
{R}	{ return patSpecParser::patREAL; }
{TIME}  { return patSpecParser::patTIME; }

{NAME}  { return patSpecParser::patNAME; }
{STR}	{ return patSpecParser::patSTRING; }

{MULT}  { return patSpecParser::patMULT;}
{PLUS}  { return patSpecParser::patPLUS;}
{MINUS}  { return patSpecParser::patMINUS;}
{DIVIDE}  { return patSpecParser::patDIVIDE;}
{POWER}  { return patSpecParser::patPOWER;}
{EQUAL}  { return patSpecParser::patEQUAL;}
{NOTEQUAL}  { return patSpecParser::patNOTEQUAL;}
{NOT}  { return patSpecParser::patNOT;}
{OR}  { return patSpecParser::patOR;}
{AND}  { return patSpecParser::patAND;}
{LESS}  { return patSpecParser::patLESS;}
{LESSEQUAL}  { return patSpecParser::patLESSEQUAL;}
{GREAT}  { return patSpecParser::patGREAT;}
{GREATEQUAL}  { return patSpecParser::patGREATEQUAL;}
{MAX}  { return patSpecParser::patMAX;}
{MIN}  { return patSpecParser::patMIN;}
{MOD}  { return patSpecParser::patMOD;}
{SQRT}  { return patSpecParser::patSQRT;}
{LOG}  { return patSpecParser::patLOG;}
{EXP}  { return patSpecParser::patEXP;}
{ABS}  { return patSpecParser::patABS;}
{INT}  { return patSpecParser::patINT;}


{PAIR}  { return patSpecParser::patPAIR; }

%%


int patSpecFlex::yywrap()
{
   return 1;
}
