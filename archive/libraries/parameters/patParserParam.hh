#ifndef YY_patBisonParam_h_included
#define YY_patBisonParam_h_included

#line 1 "/usr/local/lib/bison.h"
/* before anything */
#ifdef c_plusplus
#ifndef __cplusplus
#define __cplusplus
#endif
#endif
#ifdef __cplusplus
#ifndef YY_USE_CLASS
#define YY_USE_CLASS
#endif
#else
#endif
#include <stdio.h>

/* #line 14 "/usr/local/lib/bison.h" */
#line 21 "patParserParam.yy.tab.h"
#define YY_patBisonParam_ERROR_BODY  = 0
#define YY_patBisonParam_LEX_BODY  = 0
#define YY_patBisonParam_MEMBERS  patScannerParam scanner; patParameters *pParameters; virtual ~patBisonParam() {};
#define YY_patBisonParam_CONSTRUCTOR_PARAM  const string& fname_
#define YY_patBisonParam_CONSTRUCTOR_INIT  : scanner(fname_) , pParameters(NULL)
#line 19 "patParserParam.yy"

  
#include <fstream>
#include <sstream>

#include "patDisplay.h"
#include "patParameters.h"

#undef yyFlexLexer
#define yyFlexLexer patFlexParam
#include <FlexLexer.h>
#include "patConst.h"

class patScannerParam : public patFlexParam {

private:
                                    // filename to be scanned
  string _filename;

public:
                                    // void ctor
  patScannerParam()
    : patFlexParam() {
  }
                                    // ctor with filename argument
  patScannerParam(const string& fname_)
    : patFlexParam(), _filename( fname_ )  {
    ifstream *is = new ifstream( fname_.c_str() ); 
    if ( !is || (*is).fail() ) {
      cerr << "Error:: cannot open input file <";
      cerr << fname_ << ">" << endl;
      exit( 1 );
    }
    else {
      switch_streams( is, 0 );
    }
  }
                                    // dtor

  ~patScannerParam() { delete yyin; }

                                    // utility functions

  const string& filename() const { return _filename; }

  string removeDelimeters( const string deli="\"\"" ) {
    
    string str = YYText();
    string::size_type carret = str.find("\n") ;
    if (carret < str.size()) str.erase(carret) ;
    string::size_type deb = str.find( deli[0] ) ;
    if (deb == str.size()) {
      return ( str ) ;
    }
    str.erase( deb , 1 );
    
    string::size_type fin = str.find( deli[1] ) ;
    if (fin >= str.size()) {
      WARNING("Unmatched delimiters (" << filename() << ":" << 
	      lineno() << ") ") ;
      return( str ) ;
    }
    str.erase( fin , 1 );
    return ( str );
  }

  string value() {
    string str = YYText() ;
    return str; 
  }

  // char* value() { return (char*) YYText(); }

  void errorQuit( int err ) {
    cout << "Error = " << err << endl;
    if ( err == 0 ) return;
    cerr << "Problem in parsing"
	 << " (" << filename() << ":" << lineno() << ") "
	 << "Field: <" << YYText() << ">" << endl;
    if ( err < 0 ) exit( 1 );
  }
};




#line 108 "patParserParam.yy"
typedef union {
  long        itype;
  float       ftype;
  string*     stype;
} yy_patBisonParam_stype;
#define YY_patBisonParam_STYPE yy_patBisonParam_stype

#line 14 "/usr/local/lib/bison.h"
 /* %{ and %header{ and %union, during decl */
#ifndef YY_patBisonParam_COMPATIBILITY
#ifndef YY_USE_CLASS
#define  YY_patBisonParam_COMPATIBILITY 1
#else
#define  YY_patBisonParam_COMPATIBILITY 0
#endif
#endif

#if YY_patBisonParam_COMPATIBILITY != 0
/* backward compatibility */
#ifdef YYLTYPE
#ifndef YY_patBisonParam_LTYPE
#define YY_patBisonParam_LTYPE YYLTYPE
/* WARNING obsolete !!! user defined YYLTYPE not reported into generated header */
/* use %define LTYPE */
#endif
#endif
#ifdef YYSTYPE
#ifndef YY_patBisonParam_STYPE 
#define YY_patBisonParam_STYPE YYSTYPE
/* WARNING obsolete !!! user defined YYSTYPE not reported into generated header */
/* use %define STYPE */
#endif
#endif
#ifdef YYDEBUG
#ifndef YY_patBisonParam_DEBUG
#define  YY_patBisonParam_DEBUG YYDEBUG
/* WARNING obsolete !!! user defined YYDEBUG not reported into generated header */
/* use %define DEBUG */
#endif
#endif
#ifdef YY_patBisonParam_STYPE
#ifndef yystype
#define yystype YY_patBisonParam_STYPE
#endif
#endif
/* use goto to be compatible */
#ifndef YY_patBisonParam_USE_GOTO
#define YY_patBisonParam_USE_GOTO 1
#endif
#endif

/* use no goto to be clean in C++ */
#ifndef YY_patBisonParam_USE_GOTO
#define YY_patBisonParam_USE_GOTO 0
#endif

#ifndef YY_patBisonParam_PURE

/* #line 63 "/usr/local/lib/bison.h" */
#line 174 "patParserParam.yy.tab.h"

#line 63 "/usr/local/lib/bison.h"
/* YY_patBisonParam_PURE */
#endif

/* #line 65 "/usr/local/lib/bison.h" */
#line 181 "patParserParam.yy.tab.h"

#line 65 "/usr/local/lib/bison.h"
/* prefix */
#ifndef YY_patBisonParam_DEBUG

/* #line 67 "/usr/local/lib/bison.h" */
#line 188 "patParserParam.yy.tab.h"

#line 67 "/usr/local/lib/bison.h"
/* YY_patBisonParam_DEBUG */
#endif
#ifndef YY_patBisonParam_LSP_NEEDED

/* #line 70 "/usr/local/lib/bison.h" */
#line 196 "patParserParam.yy.tab.h"

#line 70 "/usr/local/lib/bison.h"
 /* YY_patBisonParam_LSP_NEEDED*/
#endif
/* DEFAULT LTYPE*/
#ifdef YY_patBisonParam_LSP_NEEDED
#ifndef YY_patBisonParam_LTYPE
typedef
  struct yyltype
    {
      int timestamp;
      int first_line;
      int first_column;
      int last_line;
      int last_column;
      char *text;
   }
  yyltype;

#define YY_patBisonParam_LTYPE yyltype
#endif
#endif
/* DEFAULT STYPE*/
#ifndef YY_patBisonParam_STYPE
#define YY_patBisonParam_STYPE int
#endif
/* DEFAULT MISCELANEOUS */
#ifndef YY_patBisonParam_PARSE
#define YY_patBisonParam_PARSE yyparse
#endif
#ifndef YY_patBisonParam_LEX
#define YY_patBisonParam_LEX yylex
#endif
#ifndef YY_patBisonParam_LVAL
#define YY_patBisonParam_LVAL yylval
#endif
#ifndef YY_patBisonParam_LLOC
#define YY_patBisonParam_LLOC yylloc
#endif
#ifndef YY_patBisonParam_CHAR
#define YY_patBisonParam_CHAR yychar
#endif
#ifndef YY_patBisonParam_NERRS
#define YY_patBisonParam_NERRS yynerrs
#endif
#ifndef YY_patBisonParam_DEBUG_FLAG
#define YY_patBisonParam_DEBUG_FLAG yydebug
#endif
#ifndef YY_patBisonParam_ERROR
#define YY_patBisonParam_ERROR yyerror
#endif

#ifndef YY_patBisonParam_PARSE_PARAM
#ifndef __STDC__
#ifndef __cplusplus
#ifndef YY_USE_CLASS
#define YY_patBisonParam_PARSE_PARAM
#ifndef YY_patBisonParam_PARSE_PARAM_DEF
#define YY_patBisonParam_PARSE_PARAM_DEF
#endif
#endif
#endif
#endif
#ifndef YY_patBisonParam_PARSE_PARAM
#define YY_patBisonParam_PARSE_PARAM void
#endif
#endif

/* TOKEN C */
#ifndef YY_USE_CLASS

#ifndef YY_patBisonParam_PURE
extern YY_patBisonParam_STYPE YY_patBisonParam_LVAL;
#endif


/* #line 143 "/usr/local/lib/bison.h" */
#line 274 "patParserParam.yy.tab.h"
#define	pat_BasicTrustRegionSection	258
#define	pat_BTRMaxGcpIter	259
#define	pat_BTRArmijoBeta1	260
#define	pat_BTRArmijoBeta2	261
#define	pat_BTRStartDraws	262
#define	pat_BTRIncreaseDraws	263
#define	pat_BTREta1	264
#define	pat_BTREta2	265
#define	pat_BTRGamma1	266
#define	pat_BTRGamma2	267
#define	pat_BTRInitRadius	268
#define	pat_BTRIncreaseTRRadius	269
#define	pat_BTRUnfeasibleCGIterations	270
#define	pat_BTRForceExactHessianIfMnl	271
#define	pat_BTRExactHessian	272
#define	pat_BTRCheapHessian	273
#define	pat_BTRQuasiNewtonUpdate	274
#define	pat_BTRInitQuasiNewtonWithTrueHessian	275
#define	pat_BTRInitQuasiNewtonWithBHHH	276
#define	pat_BTRMaxIter	277
#define	pat_BTRTypf	278
#define	pat_BTRTolerance	279
#define	pat_BTRMaxTRRadius	280
#define	pat_BTRMinTRRadius	281
#define	pat_BTRUsePreconditioner	282
#define	pat_BTRSingularityThreshold	283
#define	pat_BTRKappaEpp	284
#define	pat_BTRKappaLbs	285
#define	pat_BTRKappaUbs	286
#define	pat_BTRKappaFrd	287
#define	pat_BTRSignificantDigits	288
#define	pat_CondTrustRegionSection	289
#define	pat_CTRAETA0	290
#define	pat_CTRAETA1	291
#define	pat_CTRAETA2	292
#define	pat_CTRAGAMMA1	293
#define	pat_CTRAGAMMA2	294
#define	pat_CTRAEPSILONC	295
#define	pat_CTRAALPHA	296
#define	pat_CTRAMU	297
#define	pat_CTRAMAXNBRFUNCTEVAL	298
#define	pat_CTRAMAXLENGTH	299
#define	pat_CTRAMAXDATA	300
#define	pat_CTRANBROFBESTPTS	301
#define	pat_CTRAPOWER	302
#define	pat_CTRAMAXRAD	303
#define	pat_CTRAMINRAD	304
#define	pat_CTRAUPPERBOUND	305
#define	pat_CTRALOWERBOUND	306
#define	pat_CTRAGAMMA3	307
#define	pat_CTRAGAMMA4	308
#define	pat_CTRACOEFVALID	309
#define	pat_CTRACOEFGEN	310
#define	pat_CTRAEPSERROR	311
#define	pat_CTRAEPSPOINT	312
#define	pat_CTRACOEFNORM	313
#define	pat_CTRAMINSTEP	314
#define	pat_CTRAMINPIVOTVALUE	315
#define	pat_CTRAGOODPIVOTVALUE	316
#define	pat_CTRAFINEPS	317
#define	pat_CTRAFINEPSREL	318
#define	pat_CTRACHECKEPS	319
#define	pat_CTRACHECKTESTEPS	320
#define	pat_CTRACHECKTESTEPSREL	321
#define	pat_CTRAVALMINGAUSS	322
#define	pat_CTRAFACTOFPOND	323
#define	pat_ConjugateGradientSection	324
#define	pat_Precond	325
#define	pat_Epsilon	326
#define	pat_CondLimit	327
#define	pat_PrecResidu	328
#define	pat_MaxCGIter	329
#define	pat_TolSchnabelEskow	330
#define	pat_DefaultValuesSection	331
#define	pat_MaxIter	332
#define	pat_InitStep	333
#define	pat_MinStep	334
#define	pat_MaxEval	335
#define	pat_NbrRun	336
#define	pat_MaxStep	337
#define	pat_AlphaProba	338
#define	pat_StepReduc	339
#define	pat_StepIncr	340
#define	pat_ExpectedImprovement	341
#define	pat_AllowPremUnsucc	342
#define	pat_PrematureStart	343
#define	pat_PrematureStep	344
#define	pat_MaxUnsuccIter	345
#define	pat_NormWeight	346
#define	pat_FilesSection	347
#define	pat_InputDirectory	348
#define	pat_OutputDirectory	349
#define	pat_TmpDirectory	350
#define	pat_FunctionEvalExec	351
#define	pat_jonSimulator	352
#define	pat_CandidateFile	353
#define	pat_ResultFile	354
#define	pat_OutsifFile	355
#define	pat_LogFile	356
#define	pat_ProblemsFile	357
#define	pat_MITSIMorigin	358
#define	pat_MITSIMinformation	359
#define	pat_MITSIMtravelTime	360
#define	pat_MITSIMexec	361
#define	pat_Formule1Section	362
#define	pat_AugmentationStep	363
#define	pat_ReductionStep	364
#define	pat_SubSpaceMaxIter	365
#define	pat_SubSpaceConsecutiveFailure	366
#define	pat_WarmUpnbre	367
#define	pat_GEVSection	368
#define	pat_gevInputDirectory	369
#define	pat_gevOutputDirectory	370
#define	pat_gevWorkingDirectory	371
#define	pat_gevSignificantDigitsParameters	372
#define	pat_gevDecimalDigitsTTest	373
#define	pat_gevDecimalDigitsStats	374
#define	pat_gevForceScientificNotation	375
#define	pat_gevSingularValueThreshold	376
#define	pat_gevPrintVarCovarAsList	377
#define	pat_gevPrintVarCovarAsMatrix	378
#define	pat_gevPrintPValue	379
#define	pat_gevNumberOfThreads	380
#define	pat_gevSaveIntermediateResults	381
#define	pat_gevVarCovarFromBHHH	382
#define	pat_gevDebugDataFirstRow	383
#define	pat_gevDebugDataLastRow	384
#define	pat_gevStoreDataOnFile	385
#define	pat_gevBinaryDataFile	386
#define	pat_gevDumpDrawsOnFile	387
#define	pat_gevReadDrawsFromFile	388
#define	pat_gevGenerateActualSample	389
#define	pat_gevOutputActualSample	390
#define	pat_gevNormalDrawsFile	391
#define	pat_gevRectangularDrawsFile	392
#define	pat_gevRandomDistrib	393
#define	pat_gevMaxPrimeNumber	394
#define	pat_gevWarningSign	395
#define	pat_gevWarningLowDraws	396
#define	pat_gevMissingValue	397
#define	pat_gevGenerateFilesForDenis	398
#define	pat_gevGenerateGnuplotFile	399
#define	pat_gevGeneratePythonFile	400
#define	pat_gevPythonFileWithEstimatedParam	401
#define	pat_gevFileForDenis	402
#define	pat_gevAutomaticScalingOfLinearUtility	403
#define	pat_gevInverseIteration	404
#define	pat_gevSeed	405
#define	pat_gevOne	406
#define	pat_gevMinimumMu	407
#define	pat_gevSummaryParameters	408
#define	pat_gevSummaryFile	409
#define	pat_gevStopFileName	410
#define	pat_gevCheckDerivatives	411
#define	pat_gevBufferSize	412
#define	pat_gevDataFileDisplayStep	413
#define	pat_gevTtestThreshold	414
#define	pat_gevGlobal	415
#define	pat_gevAnalGrad	416
#define	pat_gevAnalHess	417
#define	pat_gevCheapF	418
#define	pat_gevFactSec	419
#define	pat_gevTermCode	420
#define	pat_gevTypx	421
#define	pat_gevTypF	422
#define	pat_gevFDigits	423
#define	pat_gevGradTol	424
#define	pat_gevMaxStep	425
#define	pat_gevItnLimit	426
#define	pat_gevDelta	427
#define	pat_gevAlgo	428
#define	pat_gevScreenPrintLevel	429
#define	pat_gevLogFilePrintLevel	430
#define	pat_gevGeneratedGroups	431
#define	pat_gevGeneratedData	432
#define	pat_gevGeneratedAttr	433
#define	pat_gevGeneratedAlt	434
#define	pat_gevSubSampleLevel	435
#define	pat_gevSubSampleBasis	436
#define	pat_gevComputeLastHessian	437
#define	pat_gevEigenvalueThreshold	438
#define	pat_gevNonParamPlotRes	439
#define	pat_gevNonParamPlotMaxY	440
#define	pat_gevNonParamPlotXSizeCm	441
#define	pat_gevNonParamPlotYSizeCm	442
#define	pat_gevNonParamPlotMinXSizeCm	443
#define	pat_gevNonParamPlotMinYSizeCm	444
#define	pat_svdMaxIter	445
#define	pat_HieLoWSection	446
#define	pat_hieMultinomial	447
#define	pat_hieTruncStructUtil	448
#define	pat_hieUpdateHessien	449
#define	pat_hieDateInLog	450
#define	pat_LogitKernelFortranSection	451
#define	pat_bolducMaxAlts	452
#define	pat_bolducMaxFact	453
#define	pat_bolducMaxNVar	454
#define	pat_NewtonLikeSection	455
#define	pat_StepSecondIndividual	456
#define	pat_NLgWeight	457
#define	pat_NLhWeight	458
#define	pat_TointSteihaugSection	459
#define	pat_TSFractionGradientRequired	460
#define	pat_TSExpTheta	461
#define	pat_cfsqpSection	462
#define	pat_cfsqpMode	463
#define	pat_cfsqpIprint	464
#define	pat_cfsqpMaxIter	465
#define	pat_cfsqpEps	466
#define	pat_cfsqpEpsEqn	467
#define	pat_cfsqpUdelta	468
#define	pat_dfoSection	469
#define	pat_dfoAddToLWRK	470
#define	pat_dfoAddToLIWRK	471
#define	pat_dfoMaxFunEval	472
#define	pat_donlp2Section	473
#define	pat_donlp2Epsx	474
#define	pat_donlp2Delmin	475
#define	pat_donlp2Smallw	476
#define	pat_donlp2Epsdif	477
#define	pat_donlp2NReset	478
#define	pat_solvoptSection	479
#define	pat_solvoptMaxIter	480
#define	pat_solvoptDisplay	481
#define	pat_solvoptErrorArgument	482
#define	pat_solvoptErrorFunction	483
#define	patEQUAL	484
#define	patOB	485
#define	patCB	486
#define	patINT	487
#define	patREAL	488
#define	patTIME	489
#define	patNAME	490
#define	patSTRING	491
#define	patPAIR	492


#line 143 "/usr/local/lib/bison.h"
 /* #defines token */
/* after #define tokens, before const tokens S5*/
#else
#ifndef YY_patBisonParam_CLASS
#define YY_patBisonParam_CLASS patBisonParam
#endif

#ifndef YY_patBisonParam_INHERIT
#define YY_patBisonParam_INHERIT
#endif
#ifndef YY_patBisonParam_MEMBERS
#define YY_patBisonParam_MEMBERS 
#endif
#ifndef YY_patBisonParam_LEX_BODY
#define YY_patBisonParam_LEX_BODY  
#endif
#ifndef YY_patBisonParam_ERROR_BODY
#define YY_patBisonParam_ERROR_BODY  
#endif
#ifndef YY_patBisonParam_CONSTRUCTOR_PARAM
#define YY_patBisonParam_CONSTRUCTOR_PARAM
#endif
/* choose between enum and const */
#ifndef YY_patBisonParam_USE_CONST_TOKEN
#define YY_patBisonParam_USE_CONST_TOKEN 0
/* yes enum is more compatible with flex,  */
/* so by default we use it */ 
#endif
#if YY_patBisonParam_USE_CONST_TOKEN != 0
#ifndef YY_patBisonParam_ENUM_TOKEN
#define YY_patBisonParam_ENUM_TOKEN yy_patBisonParam_enum_token
#endif
#endif

class YY_patBisonParam_CLASS YY_patBisonParam_INHERIT
{
public: 
#if YY_patBisonParam_USE_CONST_TOKEN != 0
/* static const int token ... */

/* #line 182 "/usr/local/lib/bison.h" */
#line 554 "patParserParam.yy.tab.h"
static const int pat_BasicTrustRegionSection;
static const int pat_BTRMaxGcpIter;
static const int pat_BTRArmijoBeta1;
static const int pat_BTRArmijoBeta2;
static const int pat_BTRStartDraws;
static const int pat_BTRIncreaseDraws;
static const int pat_BTREta1;
static const int pat_BTREta2;
static const int pat_BTRGamma1;
static const int pat_BTRGamma2;
static const int pat_BTRInitRadius;
static const int pat_BTRIncreaseTRRadius;
static const int pat_BTRUnfeasibleCGIterations;
static const int pat_BTRForceExactHessianIfMnl;
static const int pat_BTRExactHessian;
static const int pat_BTRCheapHessian;
static const int pat_BTRQuasiNewtonUpdate;
static const int pat_BTRInitQuasiNewtonWithTrueHessian;
static const int pat_BTRInitQuasiNewtonWithBHHH;
static const int pat_BTRMaxIter;
static const int pat_BTRTypf;
static const int pat_BTRTolerance;
static const int pat_BTRMaxTRRadius;
static const int pat_BTRMinTRRadius;
static const int pat_BTRUsePreconditioner;
static const int pat_BTRSingularityThreshold;
static const int pat_BTRKappaEpp;
static const int pat_BTRKappaLbs;
static const int pat_BTRKappaUbs;
static const int pat_BTRKappaFrd;
static const int pat_BTRSignificantDigits;
static const int pat_CondTrustRegionSection;
static const int pat_CTRAETA0;
static const int pat_CTRAETA1;
static const int pat_CTRAETA2;
static const int pat_CTRAGAMMA1;
static const int pat_CTRAGAMMA2;
static const int pat_CTRAEPSILONC;
static const int pat_CTRAALPHA;
static const int pat_CTRAMU;
static const int pat_CTRAMAXNBRFUNCTEVAL;
static const int pat_CTRAMAXLENGTH;
static const int pat_CTRAMAXDATA;
static const int pat_CTRANBROFBESTPTS;
static const int pat_CTRAPOWER;
static const int pat_CTRAMAXRAD;
static const int pat_CTRAMINRAD;
static const int pat_CTRAUPPERBOUND;
static const int pat_CTRALOWERBOUND;
static const int pat_CTRAGAMMA3;
static const int pat_CTRAGAMMA4;
static const int pat_CTRACOEFVALID;
static const int pat_CTRACOEFGEN;
static const int pat_CTRAEPSERROR;
static const int pat_CTRAEPSPOINT;
static const int pat_CTRACOEFNORM;
static const int pat_CTRAMINSTEP;
static const int pat_CTRAMINPIVOTVALUE;
static const int pat_CTRAGOODPIVOTVALUE;
static const int pat_CTRAFINEPS;
static const int pat_CTRAFINEPSREL;
static const int pat_CTRACHECKEPS;
static const int pat_CTRACHECKTESTEPS;
static const int pat_CTRACHECKTESTEPSREL;
static const int pat_CTRAVALMINGAUSS;
static const int pat_CTRAFACTOFPOND;
static const int pat_ConjugateGradientSection;
static const int pat_Precond;
static const int pat_Epsilon;
static const int pat_CondLimit;
static const int pat_PrecResidu;
static const int pat_MaxCGIter;
static const int pat_TolSchnabelEskow;
static const int pat_DefaultValuesSection;
static const int pat_MaxIter;
static const int pat_InitStep;
static const int pat_MinStep;
static const int pat_MaxEval;
static const int pat_NbrRun;
static const int pat_MaxStep;
static const int pat_AlphaProba;
static const int pat_StepReduc;
static const int pat_StepIncr;
static const int pat_ExpectedImprovement;
static const int pat_AllowPremUnsucc;
static const int pat_PrematureStart;
static const int pat_PrematureStep;
static const int pat_MaxUnsuccIter;
static const int pat_NormWeight;
static const int pat_FilesSection;
static const int pat_InputDirectory;
static const int pat_OutputDirectory;
static const int pat_TmpDirectory;
static const int pat_FunctionEvalExec;
static const int pat_jonSimulator;
static const int pat_CandidateFile;
static const int pat_ResultFile;
static const int pat_OutsifFile;
static const int pat_LogFile;
static const int pat_ProblemsFile;
static const int pat_MITSIMorigin;
static const int pat_MITSIMinformation;
static const int pat_MITSIMtravelTime;
static const int pat_MITSIMexec;
static const int pat_Formule1Section;
static const int pat_AugmentationStep;
static const int pat_ReductionStep;
static const int pat_SubSpaceMaxIter;
static const int pat_SubSpaceConsecutiveFailure;
static const int pat_WarmUpnbre;
static const int pat_GEVSection;
static const int pat_gevInputDirectory;
static const int pat_gevOutputDirectory;
static const int pat_gevWorkingDirectory;
static const int pat_gevSignificantDigitsParameters;
static const int pat_gevDecimalDigitsTTest;
static const int pat_gevDecimalDigitsStats;
static const int pat_gevForceScientificNotation;
static const int pat_gevSingularValueThreshold;
static const int pat_gevPrintVarCovarAsList;
static const int pat_gevPrintVarCovarAsMatrix;
static const int pat_gevPrintPValue;
static const int pat_gevNumberOfThreads;
static const int pat_gevSaveIntermediateResults;
static const int pat_gevVarCovarFromBHHH;
static const int pat_gevDebugDataFirstRow;
static const int pat_gevDebugDataLastRow;
static const int pat_gevStoreDataOnFile;
static const int pat_gevBinaryDataFile;
static const int pat_gevDumpDrawsOnFile;
static const int pat_gevReadDrawsFromFile;
static const int pat_gevGenerateActualSample;
static const int pat_gevOutputActualSample;
static const int pat_gevNormalDrawsFile;
static const int pat_gevRectangularDrawsFile;
static const int pat_gevRandomDistrib;
static const int pat_gevMaxPrimeNumber;
static const int pat_gevWarningSign;
static const int pat_gevWarningLowDraws;
static const int pat_gevMissingValue;
static const int pat_gevGenerateFilesForDenis;
static const int pat_gevGenerateGnuplotFile;
static const int pat_gevGeneratePythonFile;
static const int pat_gevPythonFileWithEstimatedParam;
static const int pat_gevFileForDenis;
static const int pat_gevAutomaticScalingOfLinearUtility;
static const int pat_gevInverseIteration;
static const int pat_gevSeed;
static const int pat_gevOne;
static const int pat_gevMinimumMu;
static const int pat_gevSummaryParameters;
static const int pat_gevSummaryFile;
static const int pat_gevStopFileName;
static const int pat_gevCheckDerivatives;
static const int pat_gevBufferSize;
static const int pat_gevDataFileDisplayStep;
static const int pat_gevTtestThreshold;
static const int pat_gevGlobal;
static const int pat_gevAnalGrad;
static const int pat_gevAnalHess;
static const int pat_gevCheapF;
static const int pat_gevFactSec;
static const int pat_gevTermCode;
static const int pat_gevTypx;
static const int pat_gevTypF;
static const int pat_gevFDigits;
static const int pat_gevGradTol;
static const int pat_gevMaxStep;
static const int pat_gevItnLimit;
static const int pat_gevDelta;
static const int pat_gevAlgo;
static const int pat_gevScreenPrintLevel;
static const int pat_gevLogFilePrintLevel;
static const int pat_gevGeneratedGroups;
static const int pat_gevGeneratedData;
static const int pat_gevGeneratedAttr;
static const int pat_gevGeneratedAlt;
static const int pat_gevSubSampleLevel;
static const int pat_gevSubSampleBasis;
static const int pat_gevComputeLastHessian;
static const int pat_gevEigenvalueThreshold;
static const int pat_gevNonParamPlotRes;
static const int pat_gevNonParamPlotMaxY;
static const int pat_gevNonParamPlotXSizeCm;
static const int pat_gevNonParamPlotYSizeCm;
static const int pat_gevNonParamPlotMinXSizeCm;
static const int pat_gevNonParamPlotMinYSizeCm;
static const int pat_svdMaxIter;
static const int pat_HieLoWSection;
static const int pat_hieMultinomial;
static const int pat_hieTruncStructUtil;
static const int pat_hieUpdateHessien;
static const int pat_hieDateInLog;
static const int pat_LogitKernelFortranSection;
static const int pat_bolducMaxAlts;
static const int pat_bolducMaxFact;
static const int pat_bolducMaxNVar;
static const int pat_NewtonLikeSection;
static const int pat_StepSecondIndividual;
static const int pat_NLgWeight;
static const int pat_NLhWeight;
static const int pat_TointSteihaugSection;
static const int pat_TSFractionGradientRequired;
static const int pat_TSExpTheta;
static const int pat_cfsqpSection;
static const int pat_cfsqpMode;
static const int pat_cfsqpIprint;
static const int pat_cfsqpMaxIter;
static const int pat_cfsqpEps;
static const int pat_cfsqpEpsEqn;
static const int pat_cfsqpUdelta;
static const int pat_dfoSection;
static const int pat_dfoAddToLWRK;
static const int pat_dfoAddToLIWRK;
static const int pat_dfoMaxFunEval;
static const int pat_donlp2Section;
static const int pat_donlp2Epsx;
static const int pat_donlp2Delmin;
static const int pat_donlp2Smallw;
static const int pat_donlp2Epsdif;
static const int pat_donlp2NReset;
static const int pat_solvoptSection;
static const int pat_solvoptMaxIter;
static const int pat_solvoptDisplay;
static const int pat_solvoptErrorArgument;
static const int pat_solvoptErrorFunction;
static const int patEQUAL;
static const int patOB;
static const int patCB;
static const int patINT;
static const int patREAL;
static const int patTIME;
static const int patNAME;
static const int patSTRING;
static const int patPAIR;


#line 182 "/usr/local/lib/bison.h"
 /* decl const */
#else
enum YY_patBisonParam_ENUM_TOKEN { YY_patBisonParam_NULL_TOKEN=0

/* #line 185 "/usr/local/lib/bison.h" */
#line 798 "patParserParam.yy.tab.h"
	,pat_BasicTrustRegionSection=258
	,pat_BTRMaxGcpIter=259
	,pat_BTRArmijoBeta1=260
	,pat_BTRArmijoBeta2=261
	,pat_BTRStartDraws=262
	,pat_BTRIncreaseDraws=263
	,pat_BTREta1=264
	,pat_BTREta2=265
	,pat_BTRGamma1=266
	,pat_BTRGamma2=267
	,pat_BTRInitRadius=268
	,pat_BTRIncreaseTRRadius=269
	,pat_BTRUnfeasibleCGIterations=270
	,pat_BTRForceExactHessianIfMnl=271
	,pat_BTRExactHessian=272
	,pat_BTRCheapHessian=273
	,pat_BTRQuasiNewtonUpdate=274
	,pat_BTRInitQuasiNewtonWithTrueHessian=275
	,pat_BTRInitQuasiNewtonWithBHHH=276
	,pat_BTRMaxIter=277
	,pat_BTRTypf=278
	,pat_BTRTolerance=279
	,pat_BTRMaxTRRadius=280
	,pat_BTRMinTRRadius=281
	,pat_BTRUsePreconditioner=282
	,pat_BTRSingularityThreshold=283
	,pat_BTRKappaEpp=284
	,pat_BTRKappaLbs=285
	,pat_BTRKappaUbs=286
	,pat_BTRKappaFrd=287
	,pat_BTRSignificantDigits=288
	,pat_CondTrustRegionSection=289
	,pat_CTRAETA0=290
	,pat_CTRAETA1=291
	,pat_CTRAETA2=292
	,pat_CTRAGAMMA1=293
	,pat_CTRAGAMMA2=294
	,pat_CTRAEPSILONC=295
	,pat_CTRAALPHA=296
	,pat_CTRAMU=297
	,pat_CTRAMAXNBRFUNCTEVAL=298
	,pat_CTRAMAXLENGTH=299
	,pat_CTRAMAXDATA=300
	,pat_CTRANBROFBESTPTS=301
	,pat_CTRAPOWER=302
	,pat_CTRAMAXRAD=303
	,pat_CTRAMINRAD=304
	,pat_CTRAUPPERBOUND=305
	,pat_CTRALOWERBOUND=306
	,pat_CTRAGAMMA3=307
	,pat_CTRAGAMMA4=308
	,pat_CTRACOEFVALID=309
	,pat_CTRACOEFGEN=310
	,pat_CTRAEPSERROR=311
	,pat_CTRAEPSPOINT=312
	,pat_CTRACOEFNORM=313
	,pat_CTRAMINSTEP=314
	,pat_CTRAMINPIVOTVALUE=315
	,pat_CTRAGOODPIVOTVALUE=316
	,pat_CTRAFINEPS=317
	,pat_CTRAFINEPSREL=318
	,pat_CTRACHECKEPS=319
	,pat_CTRACHECKTESTEPS=320
	,pat_CTRACHECKTESTEPSREL=321
	,pat_CTRAVALMINGAUSS=322
	,pat_CTRAFACTOFPOND=323
	,pat_ConjugateGradientSection=324
	,pat_Precond=325
	,pat_Epsilon=326
	,pat_CondLimit=327
	,pat_PrecResidu=328
	,pat_MaxCGIter=329
	,pat_TolSchnabelEskow=330
	,pat_DefaultValuesSection=331
	,pat_MaxIter=332
	,pat_InitStep=333
	,pat_MinStep=334
	,pat_MaxEval=335
	,pat_NbrRun=336
	,pat_MaxStep=337
	,pat_AlphaProba=338
	,pat_StepReduc=339
	,pat_StepIncr=340
	,pat_ExpectedImprovement=341
	,pat_AllowPremUnsucc=342
	,pat_PrematureStart=343
	,pat_PrematureStep=344
	,pat_MaxUnsuccIter=345
	,pat_NormWeight=346
	,pat_FilesSection=347
	,pat_InputDirectory=348
	,pat_OutputDirectory=349
	,pat_TmpDirectory=350
	,pat_FunctionEvalExec=351
	,pat_jonSimulator=352
	,pat_CandidateFile=353
	,pat_ResultFile=354
	,pat_OutsifFile=355
	,pat_LogFile=356
	,pat_ProblemsFile=357
	,pat_MITSIMorigin=358
	,pat_MITSIMinformation=359
	,pat_MITSIMtravelTime=360
	,pat_MITSIMexec=361
	,pat_Formule1Section=362
	,pat_AugmentationStep=363
	,pat_ReductionStep=364
	,pat_SubSpaceMaxIter=365
	,pat_SubSpaceConsecutiveFailure=366
	,pat_WarmUpnbre=367
	,pat_GEVSection=368
	,pat_gevInputDirectory=369
	,pat_gevOutputDirectory=370
	,pat_gevWorkingDirectory=371
	,pat_gevSignificantDigitsParameters=372
	,pat_gevDecimalDigitsTTest=373
	,pat_gevDecimalDigitsStats=374
	,pat_gevForceScientificNotation=375
	,pat_gevSingularValueThreshold=376
	,pat_gevPrintVarCovarAsList=377
	,pat_gevPrintVarCovarAsMatrix=378
	,pat_gevPrintPValue=379
	,pat_gevNumberOfThreads=380
	,pat_gevSaveIntermediateResults=381
	,pat_gevVarCovarFromBHHH=382
	,pat_gevDebugDataFirstRow=383
	,pat_gevDebugDataLastRow=384
	,pat_gevStoreDataOnFile=385
	,pat_gevBinaryDataFile=386
	,pat_gevDumpDrawsOnFile=387
	,pat_gevReadDrawsFromFile=388
	,pat_gevGenerateActualSample=389
	,pat_gevOutputActualSample=390
	,pat_gevNormalDrawsFile=391
	,pat_gevRectangularDrawsFile=392
	,pat_gevRandomDistrib=393
	,pat_gevMaxPrimeNumber=394
	,pat_gevWarningSign=395
	,pat_gevWarningLowDraws=396
	,pat_gevMissingValue=397
	,pat_gevGenerateFilesForDenis=398
	,pat_gevGenerateGnuplotFile=399
	,pat_gevGeneratePythonFile=400
	,pat_gevPythonFileWithEstimatedParam=401
	,pat_gevFileForDenis=402
	,pat_gevAutomaticScalingOfLinearUtility=403
	,pat_gevInverseIteration=404
	,pat_gevSeed=405
	,pat_gevOne=406
	,pat_gevMinimumMu=407
	,pat_gevSummaryParameters=408
	,pat_gevSummaryFile=409
	,pat_gevStopFileName=410
	,pat_gevCheckDerivatives=411
	,pat_gevBufferSize=412
	,pat_gevDataFileDisplayStep=413
	,pat_gevTtestThreshold=414
	,pat_gevGlobal=415
	,pat_gevAnalGrad=416
	,pat_gevAnalHess=417
	,pat_gevCheapF=418
	,pat_gevFactSec=419
	,pat_gevTermCode=420
	,pat_gevTypx=421
	,pat_gevTypF=422
	,pat_gevFDigits=423
	,pat_gevGradTol=424
	,pat_gevMaxStep=425
	,pat_gevItnLimit=426
	,pat_gevDelta=427
	,pat_gevAlgo=428
	,pat_gevScreenPrintLevel=429
	,pat_gevLogFilePrintLevel=430
	,pat_gevGeneratedGroups=431
	,pat_gevGeneratedData=432
	,pat_gevGeneratedAttr=433
	,pat_gevGeneratedAlt=434
	,pat_gevSubSampleLevel=435
	,pat_gevSubSampleBasis=436
	,pat_gevComputeLastHessian=437
	,pat_gevEigenvalueThreshold=438
	,pat_gevNonParamPlotRes=439
	,pat_gevNonParamPlotMaxY=440
	,pat_gevNonParamPlotXSizeCm=441
	,pat_gevNonParamPlotYSizeCm=442
	,pat_gevNonParamPlotMinXSizeCm=443
	,pat_gevNonParamPlotMinYSizeCm=444
	,pat_svdMaxIter=445
	,pat_HieLoWSection=446
	,pat_hieMultinomial=447
	,pat_hieTruncStructUtil=448
	,pat_hieUpdateHessien=449
	,pat_hieDateInLog=450
	,pat_LogitKernelFortranSection=451
	,pat_bolducMaxAlts=452
	,pat_bolducMaxFact=453
	,pat_bolducMaxNVar=454
	,pat_NewtonLikeSection=455
	,pat_StepSecondIndividual=456
	,pat_NLgWeight=457
	,pat_NLhWeight=458
	,pat_TointSteihaugSection=459
	,pat_TSFractionGradientRequired=460
	,pat_TSExpTheta=461
	,pat_cfsqpSection=462
	,pat_cfsqpMode=463
	,pat_cfsqpIprint=464
	,pat_cfsqpMaxIter=465
	,pat_cfsqpEps=466
	,pat_cfsqpEpsEqn=467
	,pat_cfsqpUdelta=468
	,pat_dfoSection=469
	,pat_dfoAddToLWRK=470
	,pat_dfoAddToLIWRK=471
	,pat_dfoMaxFunEval=472
	,pat_donlp2Section=473
	,pat_donlp2Epsx=474
	,pat_donlp2Delmin=475
	,pat_donlp2Smallw=476
	,pat_donlp2Epsdif=477
	,pat_donlp2NReset=478
	,pat_solvoptSection=479
	,pat_solvoptMaxIter=480
	,pat_solvoptDisplay=481
	,pat_solvoptErrorArgument=482
	,pat_solvoptErrorFunction=483
	,patEQUAL=484
	,patOB=485
	,patCB=486
	,patINT=487
	,patREAL=488
	,patTIME=489
	,patNAME=490
	,patSTRING=491
	,patPAIR=492


#line 185 "/usr/local/lib/bison.h"
 /* enum token */
     }; /* end of enum declaration */
#endif
public:
 int YY_patBisonParam_PARSE(YY_patBisonParam_PARSE_PARAM);
 virtual void YY_patBisonParam_ERROR(char *msg) YY_patBisonParam_ERROR_BODY;
#ifdef YY_patBisonParam_PURE
#ifdef YY_patBisonParam_LSP_NEEDED
 virtual int  YY_patBisonParam_LEX(YY_patBisonParam_STYPE *YY_patBisonParam_LVAL,YY_patBisonParam_LTYPE *YY_patBisonParam_LLOC) YY_patBisonParam_LEX_BODY;
#else
 virtual int  YY_patBisonParam_LEX(YY_patBisonParam_STYPE *YY_patBisonParam_LVAL) YY_patBisonParam_LEX_BODY;
#endif
#else
 virtual int YY_patBisonParam_LEX() YY_patBisonParam_LEX_BODY;
 YY_patBisonParam_STYPE YY_patBisonParam_LVAL;
#ifdef YY_patBisonParam_LSP_NEEDED
 YY_patBisonParam_LTYPE YY_patBisonParam_LLOC;
#endif
 int YY_patBisonParam_NERRS;
 int YY_patBisonParam_CHAR;
#endif
#if YY_patBisonParam_DEBUG != 0
public:
 int YY_patBisonParam_DEBUG_FLAG;	/*  nonzero means print parse trace	*/
#endif
public:
 YY_patBisonParam_CLASS(YY_patBisonParam_CONSTRUCTOR_PARAM);
public:
 YY_patBisonParam_MEMBERS 
};
/* other declare folow */
#endif


#if YY_patBisonParam_COMPATIBILITY != 0
/* backward compatibility */
#ifndef YYSTYPE
#define YYSTYPE YY_patBisonParam_STYPE
#endif

#ifndef YYLTYPE
#define YYLTYPE YY_patBisonParam_LTYPE
#endif
#ifndef YYDEBUG
#ifdef YY_patBisonParam_DEBUG 
#define YYDEBUG YY_patBisonParam_DEBUG
#endif
#endif

#endif
/* END */

/* #line 236 "/usr/local/lib/bison.h" */
#line 1090 "patParserParam.yy.tab.h"

#line 1556 "patParserParam.yy"

//--------------------------------------------------------------------
// Following pieces of code will be verbosely copied into the parser.
//--------------------------------------------------------------------

class patParserParam: public patBisonParam {

public:
                                    // ctor with filename argument

  patParserParam( const string& fname_ ) :	
    patBisonParam( fname_.c_str() ) {}
  
                                    // dtor
  virtual ~patParserParam() {}
                                    // Utility functions

  string filename() const { return scanner.filename(); }

  void yyerror( char* msg ) {
    cout << "Call to yyerror" << endl << endl ;
    cerr << *msg;
    cerr << " (" << filename() << ":" << scanner.lineno() << ") ";
    cerr << "Field: <" << scanner.YYText() << ">" << endl;
    exit( 1 );
  }

  int yylex() { return scanner.yylex(); }

  patBoolean parse( patParameters *p ) {
     if ( pParameters ) {
       cerr << "\nError:: cannot parse <";
       cerr << filename() << "> twice" << endl;
       return( patFALSE );
     }
     else {
       ostringstream os;
       os << "Parsing <"
	  << filename() << ">";
       pParameters = p;
       yyparse();
       return( patTRUE );
     }
   }
};

 #endif
