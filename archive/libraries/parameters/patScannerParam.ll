/*-*-c++-*------------------------------------------------------------
//
// File name : patScannerParam.l
// File automatically generated by ./automaticParser
// Michel Bierlaire, EPFL
// Date :      Sun Aug  3 09:53:14 2008
//
//--------------------------------------------------------------------
*/

%option noyywrap
%option yyclass="patParamFlex"
%option yylineno


%{

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patParserParam.hh"
#include <string>
#define YY_BREAK

%}

%option yyclass="patFlexParam"
%option noyywrap
%option yylineno

pat_BasicTrustRegionSection      ("[BasicTrustRegion]")
pat_BTRMaxGcpIter    ("BTRMaxGcpIter")
pat_BTRArmijoBeta1    ("BTRArmijoBeta1")
pat_BTRArmijoBeta2    ("BTRArmijoBeta2")
pat_BTRStartDraws    ("BTRStartDraws")
pat_BTRIncreaseDraws    ("BTRIncreaseDraws")
pat_BTREta1    ("BTREta1")
pat_BTREta2    ("BTREta2")
pat_BTRGamma1    ("BTRGamma1")
pat_BTRGamma2    ("BTRGamma2")
pat_BTRInitRadius    ("BTRInitRadius")
pat_BTRIncreaseTRRadius    ("BTRIncreaseTRRadius")
pat_BTRUnfeasibleCGIterations    ("BTRUnfeasibleCGIterations")
pat_BTRForceExactHessianIfMnl    ("BTRForceExactHessianIfMnl")
pat_BTRExactHessian    ("BTRExactHessian")
pat_BTRCheapHessian    ("BTRCheapHessian")
pat_BTRQuasiNewtonUpdate    ("BTRQuasiNewtonUpdate")
pat_BTRInitQuasiNewtonWithTrueHessian    ("BTRInitQuasiNewtonWithTrueHessian")
pat_BTRInitQuasiNewtonWithBHHH    ("BTRInitQuasiNewtonWithBHHH")
pat_BTRMaxIter    ("BTRMaxIter")
pat_BTRTypf    ("BTRTypf")
pat_BTRTolerance    ("BTRTolerance")
pat_BTRMaxTRRadius    ("BTRMaxTRRadius")
pat_BTRMinTRRadius    ("BTRMinTRRadius")
pat_BTRUsePreconditioner    ("BTRUsePreconditioner")
pat_BTRSingularityThreshold    ("BTRSingularityThreshold")
pat_BTRKappaEpp    ("BTRKappaEpp")
pat_BTRKappaLbs    ("BTRKappaLbs")
pat_BTRKappaUbs    ("BTRKappaUbs")
pat_BTRKappaFrd    ("BTRKappaFrd")
pat_BTRSignificantDigits    ("BTRSignificantDigits")
pat_CondTrustRegionSection      ("[CondTrustRegion]")
pat_CTRAETA0    ("CTRAETA0")
pat_CTRAETA1    ("CTRAETA1")
pat_CTRAETA2    ("CTRAETA2")
pat_CTRAGAMMA1    ("CTRAGAMMA1")
pat_CTRAGAMMA2    ("CTRAGAMMA2")
pat_CTRAEPSILONC    ("CTRAEPSILONC")
pat_CTRAALPHA    ("CTRAALPHA")
pat_CTRAMU    ("CTRAMU")
pat_CTRAMAXNBRFUNCTEVAL    ("CTRAMAXNBRFUNCTEVAL")
pat_CTRAMAXLENGTH    ("CTRAMAXLENGTH")
pat_CTRAMAXDATA    ("CTRAMAXDATA")
pat_CTRANBROFBESTPTS    ("CTRANBROFBESTPTS")
pat_CTRAPOWER    ("CTRAPOWER")
pat_CTRAMAXRAD    ("CTRAMAXRAD")
pat_CTRAMINRAD    ("CTRAMINRAD")
pat_CTRAUPPERBOUND    ("CTRAUPPERBOUND")
pat_CTRALOWERBOUND    ("CTRALOWERBOUND")
pat_CTRAGAMMA3    ("CTRAGAMMA3")
pat_CTRAGAMMA4    ("CTRAGAMMA4")
pat_CTRACOEFVALID    ("CTRACOEFVALID")
pat_CTRACOEFGEN    ("CTRACOEFGEN")
pat_CTRAEPSERROR    ("CTRAEPSERROR")
pat_CTRAEPSPOINT    ("CTRAEPSPOINT")
pat_CTRACOEFNORM    ("CTRACOEFNORM")
pat_CTRAMINSTEP    ("CTRAMINSTEP")
pat_CTRAMINPIVOTVALUE    ("CTRAMINPIVOTVALUE")
pat_CTRAGOODPIVOTVALUE    ("CTRAGOODPIVOTVALUE")
pat_CTRAFINEPS    ("CTRAFINEPS")
pat_CTRAFINEPSREL    ("CTRAFINEPSREL")
pat_CTRACHECKEPS    ("CTRACHECKEPS")
pat_CTRACHECKTESTEPS    ("CTRACHECKTESTEPS")
pat_CTRACHECKTESTEPSREL    ("CTRACHECKTESTEPSREL")
pat_CTRAVALMINGAUSS    ("CTRAVALMINGAUSS")
pat_CTRAFACTOFPOND    ("CTRAFACTOFPOND")
pat_ConjugateGradientSection      ("[ConjugateGradient]")
pat_Precond    ("Precond")
pat_Epsilon    ("Epsilon")
pat_CondLimit    ("CondLimit")
pat_PrecResidu    ("PrecResidu")
pat_MaxCGIter    ("MaxCGIter")
pat_TolSchnabelEskow    ("TolSchnabelEskow")
pat_DefaultValuesSection      ("[DefaultValues]")
pat_MaxIter    ("MaxIter")
pat_InitStep    ("InitStep")
pat_MinStep    ("MinStep")
pat_MaxEval    ("MaxEval")
pat_NbrRun    ("NbrRun")
pat_MaxStep    ("MaxStep")
pat_AlphaProba    ("AlphaProba")
pat_StepReduc    ("StepReduc")
pat_StepIncr    ("StepIncr")
pat_ExpectedImprovement    ("ExpectedImprovement")
pat_AllowPremUnsucc    ("AllowPremUnsucc")
pat_PrematureStart    ("PrematureStart")
pat_PrematureStep    ("PrematureStep")
pat_MaxUnsuccIter    ("MaxUnsuccIter")
pat_NormWeight    ("NormWeight")
pat_FilesSection      ("[Files]")
pat_InputDirectory    ("InputDirectory")
pat_OutputDirectory    ("OutputDirectory")
pat_TmpDirectory    ("TmpDirectory")
pat_FunctionEvalExec    ("FunctionEvalExec")
pat_jonSimulator    ("jonSimulator")
pat_CandidateFile    ("CandidateFile")
pat_ResultFile    ("ResultFile")
pat_OutsifFile    ("OutsifFile")
pat_LogFile    ("LogFile")
pat_ProblemsFile    ("ProblemsFile")
pat_MITSIMorigin    ("MITSIMorigin")
pat_MITSIMinformation    ("MITSIMinformation")
pat_MITSIMtravelTime    ("MITSIMtravelTime")
pat_MITSIMexec    ("MITSIMexec")
pat_Formule1Section      ("[Formule1]")
pat_AugmentationStep    ("AugmentationStep")
pat_ReductionStep    ("ReductionStep")
pat_SubSpaceMaxIter    ("SubSpaceMaxIter")
pat_SubSpaceConsecutiveFailure    ("SubSpaceConsecutiveFailure")
pat_WarmUpnbre    ("WarmUpnbre")
pat_GEVSection      ("[GEV]")
pat_gevInputDirectory    ("gevInputDirectory")
pat_gevOutputDirectory    ("gevOutputDirectory")
pat_gevWorkingDirectory    ("gevWorkingDirectory")
pat_gevSignificantDigitsParameters    ("gevSignificantDigitsParameters")
pat_gevDecimalDigitsTTest    ("gevDecimalDigitsTTest")
pat_gevDecimalDigitsStats    ("gevDecimalDigitsStats")
pat_gevForceScientificNotation    ("gevForceScientificNotation")
pat_gevSingularValueThreshold    ("gevSingularValueThreshold")
pat_gevPrintVarCovarAsList    ("gevPrintVarCovarAsList")
pat_gevPrintVarCovarAsMatrix    ("gevPrintVarCovarAsMatrix")
pat_gevPrintPValue    ("gevPrintPValue")
pat_gevNumberOfThreads    ("gevNumberOfThreads")
pat_gevSaveIntermediateResults    ("gevSaveIntermediateResults")
pat_gevVarCovarFromBHHH    ("gevVarCovarFromBHHH")
pat_gevDebugDataFirstRow    ("gevDebugDataFirstRow")
pat_gevDebugDataLastRow    ("gevDebugDataLastRow")
pat_gevStoreDataOnFile    ("gevStoreDataOnFile")
pat_gevBinaryDataFile    ("gevBinaryDataFile")
pat_gevDumpDrawsOnFile    ("gevDumpDrawsOnFile")
pat_gevReadDrawsFromFile    ("gevReadDrawsFromFile")
pat_gevGenerateActualSample    ("gevGenerateActualSample")
pat_gevOutputActualSample    ("gevOutputActualSample")
pat_gevNormalDrawsFile    ("gevNormalDrawsFile")
pat_gevRectangularDrawsFile    ("gevRectangularDrawsFile")
pat_gevRandomDistrib    ("gevRandomDistrib")
pat_gevMaxPrimeNumber    ("gevMaxPrimeNumber")
pat_gevWarningSign    ("gevWarningSign")
pat_gevWarningLowDraws    ("gevWarningLowDraws")
pat_gevMissingValue    ("gevMissingValue")
pat_gevGenerateFilesForDenis    ("gevGenerateFilesForDenis")
pat_gevGenerateGnuplotFile    ("gevGenerateGnuplotFile")
pat_gevGeneratePythonFile    ("gevGeneratePythonFile")
pat_gevPythonFileWithEstimatedParam    ("gevPythonFileWithEstimatedParam")
pat_gevFileForDenis    ("gevFileForDenis")
pat_gevAutomaticScalingOfLinearUtility    ("gevAutomaticScalingOfLinearUtility")
pat_gevInverseIteration    ("gevInverseIteration")
pat_gevSeed    ("gevSeed")
pat_gevOne    ("gevOne")
pat_gevMinimumMu    ("gevMinimumMu")
pat_gevSummaryParameters    ("gevSummaryParameters")
pat_gevSummaryFile    ("gevSummaryFile")
pat_gevStopFileName    ("gevStopFileName")
pat_gevCheckDerivatives    ("gevCheckDerivatives")
pat_gevBufferSize    ("gevBufferSize")
pat_gevDataFileDisplayStep    ("gevDataFileDisplayStep")
pat_gevTtestThreshold    ("gevTtestThreshold")
pat_gevGlobal    ("gevGlobal")
pat_gevAnalGrad    ("gevAnalGrad")
pat_gevAnalHess    ("gevAnalHess")
pat_gevCheapF    ("gevCheapF")
pat_gevFactSec    ("gevFactSec")
pat_gevTermCode    ("gevTermCode")
pat_gevTypx    ("gevTypx")
pat_gevTypF    ("gevTypF")
pat_gevFDigits    ("gevFDigits")
pat_gevGradTol    ("gevGradTol")
pat_gevMaxStep    ("gevMaxStep")
pat_gevItnLimit    ("gevItnLimit")
pat_gevDelta    ("gevDelta")
pat_gevAlgo    ("gevAlgo")
pat_gevScreenPrintLevel    ("gevScreenPrintLevel")
pat_gevLogFilePrintLevel    ("gevLogFilePrintLevel")
pat_gevGeneratedGroups    ("gevGeneratedGroups")
pat_gevGeneratedData    ("gevGeneratedData")
pat_gevGeneratedAttr    ("gevGeneratedAttr")
pat_gevGeneratedAlt    ("gevGeneratedAlt")
pat_gevSubSampleLevel    ("gevSubSampleLevel")
pat_gevSubSampleBasis    ("gevSubSampleBasis")
pat_gevComputeLastHessian    ("gevComputeLastHessian")
pat_gevEigenvalueThreshold    ("gevEigenvalueThreshold")
pat_gevNonParamPlotRes    ("gevNonParamPlotRes")
pat_gevNonParamPlotMaxY    ("gevNonParamPlotMaxY")
pat_gevNonParamPlotXSizeCm    ("gevNonParamPlotXSizeCm")
pat_gevNonParamPlotYSizeCm    ("gevNonParamPlotYSizeCm")
pat_gevNonParamPlotMinXSizeCm    ("gevNonParamPlotMinXSizeCm")
pat_gevNonParamPlotMinYSizeCm    ("gevNonParamPlotMinYSizeCm")
pat_svdMaxIter    ("svdMaxIter")
pat_HieLoWSection      ("[HieLoW]")
pat_hieMultinomial    ("hieMultinomial")
pat_hieTruncStructUtil    ("hieTruncStructUtil")
pat_hieUpdateHessien    ("hieUpdateHessien")
pat_hieDateInLog    ("hieDateInLog")
pat_LogitKernelFortranSection      ("[LogitKernelFortran]")
pat_bolducMaxAlts    ("bolducMaxAlts")
pat_bolducMaxFact    ("bolducMaxFact")
pat_bolducMaxNVar    ("bolducMaxNVar")
pat_NewtonLikeSection      ("[NewtonLike]")
pat_StepSecondIndividual    ("StepSecondIndividual")
pat_NLgWeight    ("NLgWeight")
pat_NLhWeight    ("NLhWeight")
pat_TointSteihaugSection      ("[TointSteihaug]")
pat_TSFractionGradientRequired    ("TSFractionGradientRequired")
pat_TSExpTheta    ("TSExpTheta")
pat_cfsqpSection      ("[cfsqp]")
pat_cfsqpMode    ("cfsqpMode")
pat_cfsqpIprint    ("cfsqpIprint")
pat_cfsqpMaxIter    ("cfsqpMaxIter")
pat_cfsqpEps    ("cfsqpEps")
pat_cfsqpEpsEqn    ("cfsqpEpsEqn")
pat_cfsqpUdelta    ("cfsqpUdelta")
pat_dfoSection      ("[dfo]")
pat_dfoAddToLWRK    ("dfoAddToLWRK")
pat_dfoAddToLIWRK    ("dfoAddToLIWRK")
pat_dfoMaxFunEval    ("dfoMaxFunEval")
pat_donlp2Section      ("[donlp2]")
pat_donlp2Epsx    ("donlp2Epsx")
pat_donlp2Delmin    ("donlp2Delmin")
pat_donlp2Smallw    ("donlp2Smallw")
pat_donlp2Epsdif    ("donlp2Epsdif")
pat_donlp2NReset    ("donlp2NReset")
pat_solvoptSection      ("[solvopt]")
pat_solvoptMaxIter    ("solvoptMaxIter")
pat_solvoptDisplay    ("solvoptDisplay")
pat_solvoptErrorArgument    ("solvoptErrorArgument")
pat_solvoptErrorFunction    ("solvoptErrorFunction")

C	(#.*)|(%.*)|("//".*)
WS	[\t ,;\n\r]+

D	[0-9]
H       [0-9A-Fa-f]
E	[eE][-+]?{D}+
N	{D}+
X       {H}+
F1	{D}+\.{D}*({E})?
F2	{D}*\.{D}+({E})?
F3	{D}+({E})?

HEXTAG  ("0x")|("0X")|("x")|("X")

I	[-+]?({N}|({HEXTAG}{X}))
R	[-+]?({F1}|{F2}|{F3})

STR	["]([^"^\n]*)[\n"]
NAME	[[]([A-Za-z \t]*)[]]

HH	(([01]?[0-9])|([2][0-3]))
MM	([0-5]?[0-9])
SS	([0-5]?[0-9])
TIME	{HH}[:]{MM}[:]{SS}

PAIR	{N}-{N}

OB	{WS}?[{]{WS}?
CB	{WS}?[}]{WS}?

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
{pat_BasicTrustRegionSection}      {return patParserParam::pat_BasicTrustRegionSection;}
{pat_BTRMaxGcpIter}  {return patParserParam::pat_BTRMaxGcpIter;}
{pat_BTRArmijoBeta1}  {return patParserParam::pat_BTRArmijoBeta1;}
{pat_BTRArmijoBeta2}  {return patParserParam::pat_BTRArmijoBeta2;}
{pat_BTRStartDraws}  {return patParserParam::pat_BTRStartDraws;}
{pat_BTRIncreaseDraws}  {return patParserParam::pat_BTRIncreaseDraws;}
{pat_BTREta1}  {return patParserParam::pat_BTREta1;}
{pat_BTREta2}  {return patParserParam::pat_BTREta2;}
{pat_BTRGamma1}  {return patParserParam::pat_BTRGamma1;}
{pat_BTRGamma2}  {return patParserParam::pat_BTRGamma2;}
{pat_BTRInitRadius}  {return patParserParam::pat_BTRInitRadius;}
{pat_BTRIncreaseTRRadius}  {return patParserParam::pat_BTRIncreaseTRRadius;}
{pat_BTRUnfeasibleCGIterations}  {return patParserParam::pat_BTRUnfeasibleCGIterations;}
{pat_BTRForceExactHessianIfMnl}  {return patParserParam::pat_BTRForceExactHessianIfMnl;}
{pat_BTRExactHessian}  {return patParserParam::pat_BTRExactHessian;}
{pat_BTRCheapHessian}  {return patParserParam::pat_BTRCheapHessian;}
{pat_BTRQuasiNewtonUpdate}  {return patParserParam::pat_BTRQuasiNewtonUpdate;}
{pat_BTRInitQuasiNewtonWithTrueHessian}  {return patParserParam::pat_BTRInitQuasiNewtonWithTrueHessian;}
{pat_BTRInitQuasiNewtonWithBHHH}  {return patParserParam::pat_BTRInitQuasiNewtonWithBHHH;}
{pat_BTRMaxIter}  {return patParserParam::pat_BTRMaxIter;}
{pat_BTRTypf}  {return patParserParam::pat_BTRTypf;}
{pat_BTRTolerance}  {return patParserParam::pat_BTRTolerance;}
{pat_BTRMaxTRRadius}  {return patParserParam::pat_BTRMaxTRRadius;}
{pat_BTRMinTRRadius}  {return patParserParam::pat_BTRMinTRRadius;}
{pat_BTRUsePreconditioner}  {return patParserParam::pat_BTRUsePreconditioner;}
{pat_BTRSingularityThreshold}  {return patParserParam::pat_BTRSingularityThreshold;}
{pat_BTRKappaEpp}  {return patParserParam::pat_BTRKappaEpp;}
{pat_BTRKappaLbs}  {return patParserParam::pat_BTRKappaLbs;}
{pat_BTRKappaUbs}  {return patParserParam::pat_BTRKappaUbs;}
{pat_BTRKappaFrd}  {return patParserParam::pat_BTRKappaFrd;}
{pat_BTRSignificantDigits}  {return patParserParam::pat_BTRSignificantDigits;}
{pat_CondTrustRegionSection}      {return patParserParam::pat_CondTrustRegionSection;}
{pat_CTRAETA0}  {return patParserParam::pat_CTRAETA0;}
{pat_CTRAETA1}  {return patParserParam::pat_CTRAETA1;}
{pat_CTRAETA2}  {return patParserParam::pat_CTRAETA2;}
{pat_CTRAGAMMA1}  {return patParserParam::pat_CTRAGAMMA1;}
{pat_CTRAGAMMA2}  {return patParserParam::pat_CTRAGAMMA2;}
{pat_CTRAEPSILONC}  {return patParserParam::pat_CTRAEPSILONC;}
{pat_CTRAALPHA}  {return patParserParam::pat_CTRAALPHA;}
{pat_CTRAMU}  {return patParserParam::pat_CTRAMU;}
{pat_CTRAMAXNBRFUNCTEVAL}  {return patParserParam::pat_CTRAMAXNBRFUNCTEVAL;}
{pat_CTRAMAXLENGTH}  {return patParserParam::pat_CTRAMAXLENGTH;}
{pat_CTRAMAXDATA}  {return patParserParam::pat_CTRAMAXDATA;}
{pat_CTRANBROFBESTPTS}  {return patParserParam::pat_CTRANBROFBESTPTS;}
{pat_CTRAPOWER}  {return patParserParam::pat_CTRAPOWER;}
{pat_CTRAMAXRAD}  {return patParserParam::pat_CTRAMAXRAD;}
{pat_CTRAMINRAD}  {return patParserParam::pat_CTRAMINRAD;}
{pat_CTRAUPPERBOUND}  {return patParserParam::pat_CTRAUPPERBOUND;}
{pat_CTRALOWERBOUND}  {return patParserParam::pat_CTRALOWERBOUND;}
{pat_CTRAGAMMA3}  {return patParserParam::pat_CTRAGAMMA3;}
{pat_CTRAGAMMA4}  {return patParserParam::pat_CTRAGAMMA4;}
{pat_CTRACOEFVALID}  {return patParserParam::pat_CTRACOEFVALID;}
{pat_CTRACOEFGEN}  {return patParserParam::pat_CTRACOEFGEN;}
{pat_CTRAEPSERROR}  {return patParserParam::pat_CTRAEPSERROR;}
{pat_CTRAEPSPOINT}  {return patParserParam::pat_CTRAEPSPOINT;}
{pat_CTRACOEFNORM}  {return patParserParam::pat_CTRACOEFNORM;}
{pat_CTRAMINSTEP}  {return patParserParam::pat_CTRAMINSTEP;}
{pat_CTRAMINPIVOTVALUE}  {return patParserParam::pat_CTRAMINPIVOTVALUE;}
{pat_CTRAGOODPIVOTVALUE}  {return patParserParam::pat_CTRAGOODPIVOTVALUE;}
{pat_CTRAFINEPS}  {return patParserParam::pat_CTRAFINEPS;}
{pat_CTRAFINEPSREL}  {return patParserParam::pat_CTRAFINEPSREL;}
{pat_CTRACHECKEPS}  {return patParserParam::pat_CTRACHECKEPS;}
{pat_CTRACHECKTESTEPS}  {return patParserParam::pat_CTRACHECKTESTEPS;}
{pat_CTRACHECKTESTEPSREL}  {return patParserParam::pat_CTRACHECKTESTEPSREL;}
{pat_CTRAVALMINGAUSS}  {return patParserParam::pat_CTRAVALMINGAUSS;}
{pat_CTRAFACTOFPOND}  {return patParserParam::pat_CTRAFACTOFPOND;}
{pat_ConjugateGradientSection}      {return patParserParam::pat_ConjugateGradientSection;}
{pat_Precond}  {return patParserParam::pat_Precond;}
{pat_Epsilon}  {return patParserParam::pat_Epsilon;}
{pat_CondLimit}  {return patParserParam::pat_CondLimit;}
{pat_PrecResidu}  {return patParserParam::pat_PrecResidu;}
{pat_MaxCGIter}  {return patParserParam::pat_MaxCGIter;}
{pat_TolSchnabelEskow}  {return patParserParam::pat_TolSchnabelEskow;}
{pat_DefaultValuesSection}      {return patParserParam::pat_DefaultValuesSection;}
{pat_MaxIter}  {return patParserParam::pat_MaxIter;}
{pat_InitStep}  {return patParserParam::pat_InitStep;}
{pat_MinStep}  {return patParserParam::pat_MinStep;}
{pat_MaxEval}  {return patParserParam::pat_MaxEval;}
{pat_NbrRun}  {return patParserParam::pat_NbrRun;}
{pat_MaxStep}  {return patParserParam::pat_MaxStep;}
{pat_AlphaProba}  {return patParserParam::pat_AlphaProba;}
{pat_StepReduc}  {return patParserParam::pat_StepReduc;}
{pat_StepIncr}  {return patParserParam::pat_StepIncr;}
{pat_ExpectedImprovement}  {return patParserParam::pat_ExpectedImprovement;}
{pat_AllowPremUnsucc}  {return patParserParam::pat_AllowPremUnsucc;}
{pat_PrematureStart}  {return patParserParam::pat_PrematureStart;}
{pat_PrematureStep}  {return patParserParam::pat_PrematureStep;}
{pat_MaxUnsuccIter}  {return patParserParam::pat_MaxUnsuccIter;}
{pat_NormWeight}  {return patParserParam::pat_NormWeight;}
{pat_FilesSection}      {return patParserParam::pat_FilesSection;}
{pat_InputDirectory}  {return patParserParam::pat_InputDirectory;}
{pat_OutputDirectory}  {return patParserParam::pat_OutputDirectory;}
{pat_TmpDirectory}  {return patParserParam::pat_TmpDirectory;}
{pat_FunctionEvalExec}  {return patParserParam::pat_FunctionEvalExec;}
{pat_jonSimulator}  {return patParserParam::pat_jonSimulator;}
{pat_CandidateFile}  {return patParserParam::pat_CandidateFile;}
{pat_ResultFile}  {return patParserParam::pat_ResultFile;}
{pat_OutsifFile}  {return patParserParam::pat_OutsifFile;}
{pat_LogFile}  {return patParserParam::pat_LogFile;}
{pat_ProblemsFile}  {return patParserParam::pat_ProblemsFile;}
{pat_MITSIMorigin}  {return patParserParam::pat_MITSIMorigin;}
{pat_MITSIMinformation}  {return patParserParam::pat_MITSIMinformation;}
{pat_MITSIMtravelTime}  {return patParserParam::pat_MITSIMtravelTime;}
{pat_MITSIMexec}  {return patParserParam::pat_MITSIMexec;}
{pat_Formule1Section}      {return patParserParam::pat_Formule1Section;}
{pat_AugmentationStep}  {return patParserParam::pat_AugmentationStep;}
{pat_ReductionStep}  {return patParserParam::pat_ReductionStep;}
{pat_SubSpaceMaxIter}  {return patParserParam::pat_SubSpaceMaxIter;}
{pat_SubSpaceConsecutiveFailure}  {return patParserParam::pat_SubSpaceConsecutiveFailure;}
{pat_WarmUpnbre}  {return patParserParam::pat_WarmUpnbre;}
{pat_GEVSection}      {return patParserParam::pat_GEVSection;}
{pat_gevInputDirectory}  {return patParserParam::pat_gevInputDirectory;}
{pat_gevOutputDirectory}  {return patParserParam::pat_gevOutputDirectory;}
{pat_gevWorkingDirectory}  {return patParserParam::pat_gevWorkingDirectory;}
{pat_gevSignificantDigitsParameters}  {return patParserParam::pat_gevSignificantDigitsParameters;}
{pat_gevDecimalDigitsTTest}  {return patParserParam::pat_gevDecimalDigitsTTest;}
{pat_gevDecimalDigitsStats}  {return patParserParam::pat_gevDecimalDigitsStats;}
{pat_gevForceScientificNotation}  {return patParserParam::pat_gevForceScientificNotation;}
{pat_gevSingularValueThreshold}  {return patParserParam::pat_gevSingularValueThreshold;}
{pat_gevPrintVarCovarAsList}  {return patParserParam::pat_gevPrintVarCovarAsList;}
{pat_gevPrintVarCovarAsMatrix}  {return patParserParam::pat_gevPrintVarCovarAsMatrix;}
{pat_gevPrintPValue}  {return patParserParam::pat_gevPrintPValue;}
{pat_gevNumberOfThreads}  {return patParserParam::pat_gevNumberOfThreads;}
{pat_gevSaveIntermediateResults}  {return patParserParam::pat_gevSaveIntermediateResults;}
{pat_gevVarCovarFromBHHH}  {return patParserParam::pat_gevVarCovarFromBHHH;}
{pat_gevDebugDataFirstRow}  {return patParserParam::pat_gevDebugDataFirstRow;}
{pat_gevDebugDataLastRow}  {return patParserParam::pat_gevDebugDataLastRow;}
{pat_gevStoreDataOnFile}  {return patParserParam::pat_gevStoreDataOnFile;}
{pat_gevBinaryDataFile}  {return patParserParam::pat_gevBinaryDataFile;}
{pat_gevDumpDrawsOnFile}  {return patParserParam::pat_gevDumpDrawsOnFile;}
{pat_gevReadDrawsFromFile}  {return patParserParam::pat_gevReadDrawsFromFile;}
{pat_gevGenerateActualSample}  {return patParserParam::pat_gevGenerateActualSample;}
{pat_gevOutputActualSample}  {return patParserParam::pat_gevOutputActualSample;}
{pat_gevNormalDrawsFile}  {return patParserParam::pat_gevNormalDrawsFile;}
{pat_gevRectangularDrawsFile}  {return patParserParam::pat_gevRectangularDrawsFile;}
{pat_gevRandomDistrib}  {return patParserParam::pat_gevRandomDistrib;}
{pat_gevMaxPrimeNumber}  {return patParserParam::pat_gevMaxPrimeNumber;}
{pat_gevWarningSign}  {return patParserParam::pat_gevWarningSign;}
{pat_gevWarningLowDraws}  {return patParserParam::pat_gevWarningLowDraws;}
{pat_gevMissingValue}  {return patParserParam::pat_gevMissingValue;}
{pat_gevGenerateFilesForDenis}  {return patParserParam::pat_gevGenerateFilesForDenis;}
{pat_gevGenerateGnuplotFile}  {return patParserParam::pat_gevGenerateGnuplotFile;}
{pat_gevGeneratePythonFile}  {return patParserParam::pat_gevGeneratePythonFile;}
{pat_gevPythonFileWithEstimatedParam}  {return patParserParam::pat_gevPythonFileWithEstimatedParam;}
{pat_gevFileForDenis}  {return patParserParam::pat_gevFileForDenis;}
{pat_gevAutomaticScalingOfLinearUtility}  {return patParserParam::pat_gevAutomaticScalingOfLinearUtility;}
{pat_gevInverseIteration}  {return patParserParam::pat_gevInverseIteration;}
{pat_gevSeed}  {return patParserParam::pat_gevSeed;}
{pat_gevOne}  {return patParserParam::pat_gevOne;}
{pat_gevMinimumMu}  {return patParserParam::pat_gevMinimumMu;}
{pat_gevSummaryParameters}  {return patParserParam::pat_gevSummaryParameters;}
{pat_gevSummaryFile}  {return patParserParam::pat_gevSummaryFile;}
{pat_gevStopFileName}  {return patParserParam::pat_gevStopFileName;}
{pat_gevCheckDerivatives}  {return patParserParam::pat_gevCheckDerivatives;}
{pat_gevBufferSize}  {return patParserParam::pat_gevBufferSize;}
{pat_gevDataFileDisplayStep}  {return patParserParam::pat_gevDataFileDisplayStep;}
{pat_gevTtestThreshold}  {return patParserParam::pat_gevTtestThreshold;}
{pat_gevGlobal}  {return patParserParam::pat_gevGlobal;}
{pat_gevAnalGrad}  {return patParserParam::pat_gevAnalGrad;}
{pat_gevAnalHess}  {return patParserParam::pat_gevAnalHess;}
{pat_gevCheapF}  {return patParserParam::pat_gevCheapF;}
{pat_gevFactSec}  {return patParserParam::pat_gevFactSec;}
{pat_gevTermCode}  {return patParserParam::pat_gevTermCode;}
{pat_gevTypx}  {return patParserParam::pat_gevTypx;}
{pat_gevTypF}  {return patParserParam::pat_gevTypF;}
{pat_gevFDigits}  {return patParserParam::pat_gevFDigits;}
{pat_gevGradTol}  {return patParserParam::pat_gevGradTol;}
{pat_gevMaxStep}  {return patParserParam::pat_gevMaxStep;}
{pat_gevItnLimit}  {return patParserParam::pat_gevItnLimit;}
{pat_gevDelta}  {return patParserParam::pat_gevDelta;}
{pat_gevAlgo}  {return patParserParam::pat_gevAlgo;}
{pat_gevScreenPrintLevel}  {return patParserParam::pat_gevScreenPrintLevel;}
{pat_gevLogFilePrintLevel}  {return patParserParam::pat_gevLogFilePrintLevel;}
{pat_gevGeneratedGroups}  {return patParserParam::pat_gevGeneratedGroups;}
{pat_gevGeneratedData}  {return patParserParam::pat_gevGeneratedData;}
{pat_gevGeneratedAttr}  {return patParserParam::pat_gevGeneratedAttr;}
{pat_gevGeneratedAlt}  {return patParserParam::pat_gevGeneratedAlt;}
{pat_gevSubSampleLevel}  {return patParserParam::pat_gevSubSampleLevel;}
{pat_gevSubSampleBasis}  {return patParserParam::pat_gevSubSampleBasis;}
{pat_gevComputeLastHessian}  {return patParserParam::pat_gevComputeLastHessian;}
{pat_gevEigenvalueThreshold}  {return patParserParam::pat_gevEigenvalueThreshold;}
{pat_gevNonParamPlotRes}  {return patParserParam::pat_gevNonParamPlotRes;}
{pat_gevNonParamPlotMaxY}  {return patParserParam::pat_gevNonParamPlotMaxY;}
{pat_gevNonParamPlotXSizeCm}  {return patParserParam::pat_gevNonParamPlotXSizeCm;}
{pat_gevNonParamPlotYSizeCm}  {return patParserParam::pat_gevNonParamPlotYSizeCm;}
{pat_gevNonParamPlotMinXSizeCm}  {return patParserParam::pat_gevNonParamPlotMinXSizeCm;}
{pat_gevNonParamPlotMinYSizeCm}  {return patParserParam::pat_gevNonParamPlotMinYSizeCm;}
{pat_svdMaxIter}  {return patParserParam::pat_svdMaxIter;}
{pat_HieLoWSection}      {return patParserParam::pat_HieLoWSection;}
{pat_hieMultinomial}  {return patParserParam::pat_hieMultinomial;}
{pat_hieTruncStructUtil}  {return patParserParam::pat_hieTruncStructUtil;}
{pat_hieUpdateHessien}  {return patParserParam::pat_hieUpdateHessien;}
{pat_hieDateInLog}  {return patParserParam::pat_hieDateInLog;}
{pat_LogitKernelFortranSection}      {return patParserParam::pat_LogitKernelFortranSection;}
{pat_bolducMaxAlts}  {return patParserParam::pat_bolducMaxAlts;}
{pat_bolducMaxFact}  {return patParserParam::pat_bolducMaxFact;}
{pat_bolducMaxNVar}  {return patParserParam::pat_bolducMaxNVar;}
{pat_NewtonLikeSection}      {return patParserParam::pat_NewtonLikeSection;}
{pat_StepSecondIndividual}  {return patParserParam::pat_StepSecondIndividual;}
{pat_NLgWeight}  {return patParserParam::pat_NLgWeight;}
{pat_NLhWeight}  {return patParserParam::pat_NLhWeight;}
{pat_TointSteihaugSection}      {return patParserParam::pat_TointSteihaugSection;}
{pat_TSFractionGradientRequired}  {return patParserParam::pat_TSFractionGradientRequired;}
{pat_TSExpTheta}  {return patParserParam::pat_TSExpTheta;}
{pat_cfsqpSection}      {return patParserParam::pat_cfsqpSection;}
{pat_cfsqpMode}  {return patParserParam::pat_cfsqpMode;}
{pat_cfsqpIprint}  {return patParserParam::pat_cfsqpIprint;}
{pat_cfsqpMaxIter}  {return patParserParam::pat_cfsqpMaxIter;}
{pat_cfsqpEps}  {return patParserParam::pat_cfsqpEps;}
{pat_cfsqpEpsEqn}  {return patParserParam::pat_cfsqpEpsEqn;}
{pat_cfsqpUdelta}  {return patParserParam::pat_cfsqpUdelta;}
{pat_dfoSection}      {return patParserParam::pat_dfoSection;}
{pat_dfoAddToLWRK}  {return patParserParam::pat_dfoAddToLWRK;}
{pat_dfoAddToLIWRK}  {return patParserParam::pat_dfoAddToLIWRK;}
{pat_dfoMaxFunEval}  {return patParserParam::pat_dfoMaxFunEval;}
{pat_donlp2Section}      {return patParserParam::pat_donlp2Section;}
{pat_donlp2Epsx}  {return patParserParam::pat_donlp2Epsx;}
{pat_donlp2Delmin}  {return patParserParam::pat_donlp2Delmin;}
{pat_donlp2Smallw}  {return patParserParam::pat_donlp2Smallw;}
{pat_donlp2Epsdif}  {return patParserParam::pat_donlp2Epsdif;}
{pat_donlp2NReset}  {return patParserParam::pat_donlp2NReset;}
{pat_solvoptSection}      {return patParserParam::pat_solvoptSection;}
{pat_solvoptMaxIter}  {return patParserParam::pat_solvoptMaxIter;}
{pat_solvoptDisplay}  {return patParserParam::pat_solvoptDisplay;}
{pat_solvoptErrorArgument}  {return patParserParam::pat_solvoptErrorArgument;}
{pat_solvoptErrorFunction}  {return patParserParam::pat_solvoptErrorFunction;}

"=" 	{ return patParserParam::patEQUAL; }

{OB}	{ return patParserParam::patOB; }
{CB}	{ return patParserParam::patCB; }
		            
{I}	{ return patParserParam::patINT; }
{R}	{ return patParserParam::patREAL; }
{TIME}  { return patParserParam::patTIME; }

{NAME}  { return patParserParam::patNAME; }
{STR}	{ return patParserParam::patSTRING; }

{PAIR}  { return patParserParam::patPAIR; }

%%

int patFlexParam::yywrap()
{
   return 1;
}
