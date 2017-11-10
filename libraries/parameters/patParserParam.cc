#define YY_patBisonParam_h_included

/*  A Bison++ parser, made from patParserParam.yy  */

 /* with Bison++ version bison++ Version 1.21-8, adapted from GNU bison by coetmeur@icdc.fr
  */


#line 1 "/usr/local/lib/bison.cc"
/* -*-C-*-  Note some compilers choke on comments on `#line' lines.  */
/* Skeleton output parser for bison,
   Copyright (C) 1984, 1989, 1990 Bob Corbett and Richard Stallman

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 1, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.  */

/* HEADER SECTION */
#if defined( _MSDOS ) || defined(MSDOS) || defined(__MSDOS__) 
#define __MSDOS_AND_ALIKE
#endif
#if defined(_WINDOWS) && defined(_MSC_VER)
#define __HAVE_NO_ALLOCA
#define __MSDOS_AND_ALIKE
#endif

#ifndef alloca
#if defined( __GNUC__)
#define alloca __builtin_alloca

#elif (!defined (__STDC__) && defined (sparc)) || defined (__sparc__) || defined (__sparc)  || defined (__sgi)
#include <alloca.h>

#elif defined (__MSDOS_AND_ALIKE)
#include <malloc.h>
#ifndef __TURBOC__
/* MS C runtime lib */
#define alloca _alloca
#endif

#elif defined(_AIX)
#include <malloc.h>
#pragma alloca

#elif defined(__hpux)
#ifdef __cplusplus
extern "C" {
void *alloca (unsigned int);
};
#else /* not __cplusplus */
void *alloca ();
#endif /* not __cplusplus */

#endif /* not _AIX  not MSDOS, or __TURBOC__ or _AIX, not sparc.  */
#endif /* alloca not defined.  */
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
#ifndef __STDC__
#define const
#endif
#endif
#include <stdio.h>
#define YYBISON 1  

/* #line 73 "/usr/local/lib/bison.cc" */
#line 85 "patParserParam.yy.tab.c"
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

#line 73 "/usr/local/lib/bison.cc"
/* %{ and %header{ and %union, during decl */
#define YY_patBisonParam_BISON 1
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
#endif
#endif
#ifdef YYSTYPE
#ifndef YY_patBisonParam_STYPE 
#define YY_patBisonParam_STYPE YYSTYPE
#endif
#endif
#ifdef YYDEBUG
#ifndef YY_patBisonParam_DEBUG
#define  YY_patBisonParam_DEBUG YYDEBUG
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

/* #line 117 "/usr/local/lib/bison.cc" */
#line 233 "patParserParam.yy.tab.c"

#line 117 "/usr/local/lib/bison.cc"
/*  YY_patBisonParam_PURE */
#endif

/* section apres lecture def, avant lecture grammaire S2 */

/* #line 121 "/usr/local/lib/bison.cc" */
#line 242 "patParserParam.yy.tab.c"

#line 121 "/usr/local/lib/bison.cc"
/* prefix */
#ifndef YY_patBisonParam_DEBUG

/* #line 123 "/usr/local/lib/bison.cc" */
#line 249 "patParserParam.yy.tab.c"

#line 123 "/usr/local/lib/bison.cc"
/* YY_patBisonParam_DEBUG */
#endif


#ifndef YY_patBisonParam_LSP_NEEDED

/* #line 128 "/usr/local/lib/bison.cc" */
#line 259 "patParserParam.yy.tab.c"

#line 128 "/usr/local/lib/bison.cc"
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
      /* We used to use `unsigned long' as YY_patBisonParam_STYPE on MSDOS,
	 but it seems better to be consistent.
	 Most programs should declare their own type anyway.  */

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
#if YY_patBisonParam_COMPATIBILITY != 0
/* backward compatibility */
#ifdef YY_patBisonParam_LTYPE
#ifndef YYLTYPE
#define YYLTYPE YY_patBisonParam_LTYPE
#else
/* WARNING obsolete !!! user defined YYLTYPE not reported into generated header */
#endif
#endif
#ifndef YYSTYPE
#define YYSTYPE YY_patBisonParam_STYPE
#else
/* WARNING obsolete !!! user defined YYSTYPE not reported into generated header */
#endif
#ifdef YY_patBisonParam_PURE
#ifndef YYPURE
#define YYPURE YY_patBisonParam_PURE
#endif
#endif
#ifdef YY_patBisonParam_DEBUG
#ifndef YYDEBUG
#define YYDEBUG YY_patBisonParam_DEBUG 
#endif
#endif
#ifndef YY_patBisonParam_ERROR_VERBOSE
#ifdef YYERROR_VERBOSE
#define YY_patBisonParam_ERROR_VERBOSE YYERROR_VERBOSE
#endif
#endif
#ifndef YY_patBisonParam_LSP_NEEDED
#ifdef YYLSP_NEEDED
#define YY_patBisonParam_LSP_NEEDED YYLSP_NEEDED
#endif
#endif
#endif
#ifndef YY_USE_CLASS
/* TOKEN C */

/* #line 236 "/usr/local/lib/bison.cc" */
#line 372 "patParserParam.yy.tab.c"
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


#line 236 "/usr/local/lib/bison.cc"
 /* #defines tokens */
#else
/* CLASS */
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
#ifndef YY_patBisonParam_CONSTRUCTOR_CODE
#define YY_patBisonParam_CONSTRUCTOR_CODE
#endif
#ifndef YY_patBisonParam_CONSTRUCTOR_INIT
#define YY_patBisonParam_CONSTRUCTOR_INIT
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

/* #line 280 "/usr/local/lib/bison.cc" */
#line 657 "patParserParam.yy.tab.c"
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


#line 280 "/usr/local/lib/bison.cc"
 /* decl const */
#else
enum YY_patBisonParam_ENUM_TOKEN { YY_patBisonParam_NULL_TOKEN=0

/* #line 283 "/usr/local/lib/bison.cc" */
#line 901 "patParserParam.yy.tab.c"
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


#line 283 "/usr/local/lib/bison.cc"
 /* enum token */
     }; /* end of enum declaration */
#endif
public:
 int YY_patBisonParam_PARSE (YY_patBisonParam_PARSE_PARAM);
 virtual void YY_patBisonParam_ERROR(char *msg) YY_patBisonParam_ERROR_BODY;
#ifdef YY_patBisonParam_PURE
#ifdef YY_patBisonParam_LSP_NEEDED
 virtual int  YY_patBisonParam_LEX (YY_patBisonParam_STYPE *YY_patBisonParam_LVAL,YY_patBisonParam_LTYPE *YY_patBisonParam_LLOC) YY_patBisonParam_LEX_BODY;
#else
 virtual int  YY_patBisonParam_LEX (YY_patBisonParam_STYPE *YY_patBisonParam_LVAL) YY_patBisonParam_LEX_BODY;
#endif
#else
 virtual int YY_patBisonParam_LEX() YY_patBisonParam_LEX_BODY;
 YY_patBisonParam_STYPE YY_patBisonParam_LVAL;
#ifdef YY_patBisonParam_LSP_NEEDED
 YY_patBisonParam_LTYPE YY_patBisonParam_LLOC;
#endif
 int   YY_patBisonParam_NERRS;
 int    YY_patBisonParam_CHAR;
#endif
#if YY_patBisonParam_DEBUG != 0
 int YY_patBisonParam_DEBUG_FLAG;   /*  nonzero means print parse trace     */
#endif
public:
 YY_patBisonParam_CLASS(YY_patBisonParam_CONSTRUCTOR_PARAM);
public:
 YY_patBisonParam_MEMBERS 
};
/* other declare folow */
#if YY_patBisonParam_USE_CONST_TOKEN != 0

/* #line 314 "/usr/local/lib/bison.cc" */
#line 1173 "patParserParam.yy.tab.c"
const int YY_patBisonParam_CLASS::pat_BasicTrustRegionSection=258;
const int YY_patBisonParam_CLASS::pat_BTRMaxGcpIter=259;
const int YY_patBisonParam_CLASS::pat_BTRArmijoBeta1=260;
const int YY_patBisonParam_CLASS::pat_BTRArmijoBeta2=261;
const int YY_patBisonParam_CLASS::pat_BTRStartDraws=262;
const int YY_patBisonParam_CLASS::pat_BTRIncreaseDraws=263;
const int YY_patBisonParam_CLASS::pat_BTREta1=264;
const int YY_patBisonParam_CLASS::pat_BTREta2=265;
const int YY_patBisonParam_CLASS::pat_BTRGamma1=266;
const int YY_patBisonParam_CLASS::pat_BTRGamma2=267;
const int YY_patBisonParam_CLASS::pat_BTRInitRadius=268;
const int YY_patBisonParam_CLASS::pat_BTRIncreaseTRRadius=269;
const int YY_patBisonParam_CLASS::pat_BTRUnfeasibleCGIterations=270;
const int YY_patBisonParam_CLASS::pat_BTRForceExactHessianIfMnl=271;
const int YY_patBisonParam_CLASS::pat_BTRExactHessian=272;
const int YY_patBisonParam_CLASS::pat_BTRCheapHessian=273;
const int YY_patBisonParam_CLASS::pat_BTRQuasiNewtonUpdate=274;
const int YY_patBisonParam_CLASS::pat_BTRInitQuasiNewtonWithTrueHessian=275;
const int YY_patBisonParam_CLASS::pat_BTRInitQuasiNewtonWithBHHH=276;
const int YY_patBisonParam_CLASS::pat_BTRMaxIter=277;
const int YY_patBisonParam_CLASS::pat_BTRTypf=278;
const int YY_patBisonParam_CLASS::pat_BTRTolerance=279;
const int YY_patBisonParam_CLASS::pat_BTRMaxTRRadius=280;
const int YY_patBisonParam_CLASS::pat_BTRMinTRRadius=281;
const int YY_patBisonParam_CLASS::pat_BTRUsePreconditioner=282;
const int YY_patBisonParam_CLASS::pat_BTRSingularityThreshold=283;
const int YY_patBisonParam_CLASS::pat_BTRKappaEpp=284;
const int YY_patBisonParam_CLASS::pat_BTRKappaLbs=285;
const int YY_patBisonParam_CLASS::pat_BTRKappaUbs=286;
const int YY_patBisonParam_CLASS::pat_BTRKappaFrd=287;
const int YY_patBisonParam_CLASS::pat_BTRSignificantDigits=288;
const int YY_patBisonParam_CLASS::pat_CondTrustRegionSection=289;
const int YY_patBisonParam_CLASS::pat_CTRAETA0=290;
const int YY_patBisonParam_CLASS::pat_CTRAETA1=291;
const int YY_patBisonParam_CLASS::pat_CTRAETA2=292;
const int YY_patBisonParam_CLASS::pat_CTRAGAMMA1=293;
const int YY_patBisonParam_CLASS::pat_CTRAGAMMA2=294;
const int YY_patBisonParam_CLASS::pat_CTRAEPSILONC=295;
const int YY_patBisonParam_CLASS::pat_CTRAALPHA=296;
const int YY_patBisonParam_CLASS::pat_CTRAMU=297;
const int YY_patBisonParam_CLASS::pat_CTRAMAXNBRFUNCTEVAL=298;
const int YY_patBisonParam_CLASS::pat_CTRAMAXLENGTH=299;
const int YY_patBisonParam_CLASS::pat_CTRAMAXDATA=300;
const int YY_patBisonParam_CLASS::pat_CTRANBROFBESTPTS=301;
const int YY_patBisonParam_CLASS::pat_CTRAPOWER=302;
const int YY_patBisonParam_CLASS::pat_CTRAMAXRAD=303;
const int YY_patBisonParam_CLASS::pat_CTRAMINRAD=304;
const int YY_patBisonParam_CLASS::pat_CTRAUPPERBOUND=305;
const int YY_patBisonParam_CLASS::pat_CTRALOWERBOUND=306;
const int YY_patBisonParam_CLASS::pat_CTRAGAMMA3=307;
const int YY_patBisonParam_CLASS::pat_CTRAGAMMA4=308;
const int YY_patBisonParam_CLASS::pat_CTRACOEFVALID=309;
const int YY_patBisonParam_CLASS::pat_CTRACOEFGEN=310;
const int YY_patBisonParam_CLASS::pat_CTRAEPSERROR=311;
const int YY_patBisonParam_CLASS::pat_CTRAEPSPOINT=312;
const int YY_patBisonParam_CLASS::pat_CTRACOEFNORM=313;
const int YY_patBisonParam_CLASS::pat_CTRAMINSTEP=314;
const int YY_patBisonParam_CLASS::pat_CTRAMINPIVOTVALUE=315;
const int YY_patBisonParam_CLASS::pat_CTRAGOODPIVOTVALUE=316;
const int YY_patBisonParam_CLASS::pat_CTRAFINEPS=317;
const int YY_patBisonParam_CLASS::pat_CTRAFINEPSREL=318;
const int YY_patBisonParam_CLASS::pat_CTRACHECKEPS=319;
const int YY_patBisonParam_CLASS::pat_CTRACHECKTESTEPS=320;
const int YY_patBisonParam_CLASS::pat_CTRACHECKTESTEPSREL=321;
const int YY_patBisonParam_CLASS::pat_CTRAVALMINGAUSS=322;
const int YY_patBisonParam_CLASS::pat_CTRAFACTOFPOND=323;
const int YY_patBisonParam_CLASS::pat_ConjugateGradientSection=324;
const int YY_patBisonParam_CLASS::pat_Precond=325;
const int YY_patBisonParam_CLASS::pat_Epsilon=326;
const int YY_patBisonParam_CLASS::pat_CondLimit=327;
const int YY_patBisonParam_CLASS::pat_PrecResidu=328;
const int YY_patBisonParam_CLASS::pat_MaxCGIter=329;
const int YY_patBisonParam_CLASS::pat_TolSchnabelEskow=330;
const int YY_patBisonParam_CLASS::pat_DefaultValuesSection=331;
const int YY_patBisonParam_CLASS::pat_MaxIter=332;
const int YY_patBisonParam_CLASS::pat_InitStep=333;
const int YY_patBisonParam_CLASS::pat_MinStep=334;
const int YY_patBisonParam_CLASS::pat_MaxEval=335;
const int YY_patBisonParam_CLASS::pat_NbrRun=336;
const int YY_patBisonParam_CLASS::pat_MaxStep=337;
const int YY_patBisonParam_CLASS::pat_AlphaProba=338;
const int YY_patBisonParam_CLASS::pat_StepReduc=339;
const int YY_patBisonParam_CLASS::pat_StepIncr=340;
const int YY_patBisonParam_CLASS::pat_ExpectedImprovement=341;
const int YY_patBisonParam_CLASS::pat_AllowPremUnsucc=342;
const int YY_patBisonParam_CLASS::pat_PrematureStart=343;
const int YY_patBisonParam_CLASS::pat_PrematureStep=344;
const int YY_patBisonParam_CLASS::pat_MaxUnsuccIter=345;
const int YY_patBisonParam_CLASS::pat_NormWeight=346;
const int YY_patBisonParam_CLASS::pat_FilesSection=347;
const int YY_patBisonParam_CLASS::pat_InputDirectory=348;
const int YY_patBisonParam_CLASS::pat_OutputDirectory=349;
const int YY_patBisonParam_CLASS::pat_TmpDirectory=350;
const int YY_patBisonParam_CLASS::pat_FunctionEvalExec=351;
const int YY_patBisonParam_CLASS::pat_jonSimulator=352;
const int YY_patBisonParam_CLASS::pat_CandidateFile=353;
const int YY_patBisonParam_CLASS::pat_ResultFile=354;
const int YY_patBisonParam_CLASS::pat_OutsifFile=355;
const int YY_patBisonParam_CLASS::pat_LogFile=356;
const int YY_patBisonParam_CLASS::pat_ProblemsFile=357;
const int YY_patBisonParam_CLASS::pat_MITSIMorigin=358;
const int YY_patBisonParam_CLASS::pat_MITSIMinformation=359;
const int YY_patBisonParam_CLASS::pat_MITSIMtravelTime=360;
const int YY_patBisonParam_CLASS::pat_MITSIMexec=361;
const int YY_patBisonParam_CLASS::pat_Formule1Section=362;
const int YY_patBisonParam_CLASS::pat_AugmentationStep=363;
const int YY_patBisonParam_CLASS::pat_ReductionStep=364;
const int YY_patBisonParam_CLASS::pat_SubSpaceMaxIter=365;
const int YY_patBisonParam_CLASS::pat_SubSpaceConsecutiveFailure=366;
const int YY_patBisonParam_CLASS::pat_WarmUpnbre=367;
const int YY_patBisonParam_CLASS::pat_GEVSection=368;
const int YY_patBisonParam_CLASS::pat_gevInputDirectory=369;
const int YY_patBisonParam_CLASS::pat_gevOutputDirectory=370;
const int YY_patBisonParam_CLASS::pat_gevWorkingDirectory=371;
const int YY_patBisonParam_CLASS::pat_gevSignificantDigitsParameters=372;
const int YY_patBisonParam_CLASS::pat_gevDecimalDigitsTTest=373;
const int YY_patBisonParam_CLASS::pat_gevDecimalDigitsStats=374;
const int YY_patBisonParam_CLASS::pat_gevForceScientificNotation=375;
const int YY_patBisonParam_CLASS::pat_gevSingularValueThreshold=376;
const int YY_patBisonParam_CLASS::pat_gevPrintVarCovarAsList=377;
const int YY_patBisonParam_CLASS::pat_gevPrintVarCovarAsMatrix=378;
const int YY_patBisonParam_CLASS::pat_gevPrintPValue=379;
const int YY_patBisonParam_CLASS::pat_gevNumberOfThreads=380;
const int YY_patBisonParam_CLASS::pat_gevSaveIntermediateResults=381;
const int YY_patBisonParam_CLASS::pat_gevVarCovarFromBHHH=382;
const int YY_patBisonParam_CLASS::pat_gevDebugDataFirstRow=383;
const int YY_patBisonParam_CLASS::pat_gevDebugDataLastRow=384;
const int YY_patBisonParam_CLASS::pat_gevStoreDataOnFile=385;
const int YY_patBisonParam_CLASS::pat_gevBinaryDataFile=386;
const int YY_patBisonParam_CLASS::pat_gevDumpDrawsOnFile=387;
const int YY_patBisonParam_CLASS::pat_gevReadDrawsFromFile=388;
const int YY_patBisonParam_CLASS::pat_gevGenerateActualSample=389;
const int YY_patBisonParam_CLASS::pat_gevOutputActualSample=390;
const int YY_patBisonParam_CLASS::pat_gevNormalDrawsFile=391;
const int YY_patBisonParam_CLASS::pat_gevRectangularDrawsFile=392;
const int YY_patBisonParam_CLASS::pat_gevRandomDistrib=393;
const int YY_patBisonParam_CLASS::pat_gevMaxPrimeNumber=394;
const int YY_patBisonParam_CLASS::pat_gevWarningSign=395;
const int YY_patBisonParam_CLASS::pat_gevWarningLowDraws=396;
const int YY_patBisonParam_CLASS::pat_gevMissingValue=397;
const int YY_patBisonParam_CLASS::pat_gevGenerateFilesForDenis=398;
const int YY_patBisonParam_CLASS::pat_gevGenerateGnuplotFile=399;
const int YY_patBisonParam_CLASS::pat_gevGeneratePythonFile=400;
const int YY_patBisonParam_CLASS::pat_gevPythonFileWithEstimatedParam=401;
const int YY_patBisonParam_CLASS::pat_gevFileForDenis=402;
const int YY_patBisonParam_CLASS::pat_gevAutomaticScalingOfLinearUtility=403;
const int YY_patBisonParam_CLASS::pat_gevInverseIteration=404;
const int YY_patBisonParam_CLASS::pat_gevSeed=405;
const int YY_patBisonParam_CLASS::pat_gevOne=406;
const int YY_patBisonParam_CLASS::pat_gevMinimumMu=407;
const int YY_patBisonParam_CLASS::pat_gevSummaryParameters=408;
const int YY_patBisonParam_CLASS::pat_gevSummaryFile=409;
const int YY_patBisonParam_CLASS::pat_gevStopFileName=410;
const int YY_patBisonParam_CLASS::pat_gevCheckDerivatives=411;
const int YY_patBisonParam_CLASS::pat_gevBufferSize=412;
const int YY_patBisonParam_CLASS::pat_gevDataFileDisplayStep=413;
const int YY_patBisonParam_CLASS::pat_gevTtestThreshold=414;
const int YY_patBisonParam_CLASS::pat_gevGlobal=415;
const int YY_patBisonParam_CLASS::pat_gevAnalGrad=416;
const int YY_patBisonParam_CLASS::pat_gevAnalHess=417;
const int YY_patBisonParam_CLASS::pat_gevCheapF=418;
const int YY_patBisonParam_CLASS::pat_gevFactSec=419;
const int YY_patBisonParam_CLASS::pat_gevTermCode=420;
const int YY_patBisonParam_CLASS::pat_gevTypx=421;
const int YY_patBisonParam_CLASS::pat_gevTypF=422;
const int YY_patBisonParam_CLASS::pat_gevFDigits=423;
const int YY_patBisonParam_CLASS::pat_gevGradTol=424;
const int YY_patBisonParam_CLASS::pat_gevMaxStep=425;
const int YY_patBisonParam_CLASS::pat_gevItnLimit=426;
const int YY_patBisonParam_CLASS::pat_gevDelta=427;
const int YY_patBisonParam_CLASS::pat_gevAlgo=428;
const int YY_patBisonParam_CLASS::pat_gevScreenPrintLevel=429;
const int YY_patBisonParam_CLASS::pat_gevLogFilePrintLevel=430;
const int YY_patBisonParam_CLASS::pat_gevGeneratedGroups=431;
const int YY_patBisonParam_CLASS::pat_gevGeneratedData=432;
const int YY_patBisonParam_CLASS::pat_gevGeneratedAttr=433;
const int YY_patBisonParam_CLASS::pat_gevGeneratedAlt=434;
const int YY_patBisonParam_CLASS::pat_gevSubSampleLevel=435;
const int YY_patBisonParam_CLASS::pat_gevSubSampleBasis=436;
const int YY_patBisonParam_CLASS::pat_gevComputeLastHessian=437;
const int YY_patBisonParam_CLASS::pat_gevEigenvalueThreshold=438;
const int YY_patBisonParam_CLASS::pat_gevNonParamPlotRes=439;
const int YY_patBisonParam_CLASS::pat_gevNonParamPlotMaxY=440;
const int YY_patBisonParam_CLASS::pat_gevNonParamPlotXSizeCm=441;
const int YY_patBisonParam_CLASS::pat_gevNonParamPlotYSizeCm=442;
const int YY_patBisonParam_CLASS::pat_gevNonParamPlotMinXSizeCm=443;
const int YY_patBisonParam_CLASS::pat_gevNonParamPlotMinYSizeCm=444;
const int YY_patBisonParam_CLASS::pat_svdMaxIter=445;
const int YY_patBisonParam_CLASS::pat_HieLoWSection=446;
const int YY_patBisonParam_CLASS::pat_hieMultinomial=447;
const int YY_patBisonParam_CLASS::pat_hieTruncStructUtil=448;
const int YY_patBisonParam_CLASS::pat_hieUpdateHessien=449;
const int YY_patBisonParam_CLASS::pat_hieDateInLog=450;
const int YY_patBisonParam_CLASS::pat_LogitKernelFortranSection=451;
const int YY_patBisonParam_CLASS::pat_bolducMaxAlts=452;
const int YY_patBisonParam_CLASS::pat_bolducMaxFact=453;
const int YY_patBisonParam_CLASS::pat_bolducMaxNVar=454;
const int YY_patBisonParam_CLASS::pat_NewtonLikeSection=455;
const int YY_patBisonParam_CLASS::pat_StepSecondIndividual=456;
const int YY_patBisonParam_CLASS::pat_NLgWeight=457;
const int YY_patBisonParam_CLASS::pat_NLhWeight=458;
const int YY_patBisonParam_CLASS::pat_TointSteihaugSection=459;
const int YY_patBisonParam_CLASS::pat_TSFractionGradientRequired=460;
const int YY_patBisonParam_CLASS::pat_TSExpTheta=461;
const int YY_patBisonParam_CLASS::pat_cfsqpSection=462;
const int YY_patBisonParam_CLASS::pat_cfsqpMode=463;
const int YY_patBisonParam_CLASS::pat_cfsqpIprint=464;
const int YY_patBisonParam_CLASS::pat_cfsqpMaxIter=465;
const int YY_patBisonParam_CLASS::pat_cfsqpEps=466;
const int YY_patBisonParam_CLASS::pat_cfsqpEpsEqn=467;
const int YY_patBisonParam_CLASS::pat_cfsqpUdelta=468;
const int YY_patBisonParam_CLASS::pat_dfoSection=469;
const int YY_patBisonParam_CLASS::pat_dfoAddToLWRK=470;
const int YY_patBisonParam_CLASS::pat_dfoAddToLIWRK=471;
const int YY_patBisonParam_CLASS::pat_dfoMaxFunEval=472;
const int YY_patBisonParam_CLASS::pat_donlp2Section=473;
const int YY_patBisonParam_CLASS::pat_donlp2Epsx=474;
const int YY_patBisonParam_CLASS::pat_donlp2Delmin=475;
const int YY_patBisonParam_CLASS::pat_donlp2Smallw=476;
const int YY_patBisonParam_CLASS::pat_donlp2Epsdif=477;
const int YY_patBisonParam_CLASS::pat_donlp2NReset=478;
const int YY_patBisonParam_CLASS::pat_solvoptSection=479;
const int YY_patBisonParam_CLASS::pat_solvoptMaxIter=480;
const int YY_patBisonParam_CLASS::pat_solvoptDisplay=481;
const int YY_patBisonParam_CLASS::pat_solvoptErrorArgument=482;
const int YY_patBisonParam_CLASS::pat_solvoptErrorFunction=483;
const int YY_patBisonParam_CLASS::patEQUAL=484;
const int YY_patBisonParam_CLASS::patOB=485;
const int YY_patBisonParam_CLASS::patCB=486;
const int YY_patBisonParam_CLASS::patINT=487;
const int YY_patBisonParam_CLASS::patREAL=488;
const int YY_patBisonParam_CLASS::patTIME=489;
const int YY_patBisonParam_CLASS::patNAME=490;
const int YY_patBisonParam_CLASS::patSTRING=491;
const int YY_patBisonParam_CLASS::patPAIR=492;


#line 314 "/usr/local/lib/bison.cc"
 /* const YY_patBisonParam_CLASS::token */
#endif
/*apres const  */
YY_patBisonParam_CLASS::YY_patBisonParam_CLASS(YY_patBisonParam_CONSTRUCTOR_PARAM) YY_patBisonParam_CONSTRUCTOR_INIT
{
#if YY_patBisonParam_DEBUG != 0
YY_patBisonParam_DEBUG_FLAG=0;
#endif
YY_patBisonParam_CONSTRUCTOR_CODE;
};
#endif

/* #line 325 "/usr/local/lib/bison.cc" */
#line 1425 "patParserParam.yy.tab.c"


#define	YYFINAL		717
#define	YYFLAG		-32768
#define	YYNTBASE	238

#define YYTRANSLATE(x) ((unsigned)(x) <= 492 ? yytranslate[x] : 289)

static const short yytranslate[] = {     0,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     1,     2,     3,     4,     5,
     6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
    16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
    26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
    36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
    46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
    56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
    66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
    76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
    86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
    96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
   106,   107,   108,   109,   110,   111,   112,   113,   114,   115,
   116,   117,   118,   119,   120,   121,   122,   123,   124,   125,
   126,   127,   128,   129,   130,   131,   132,   133,   134,   135,
   136,   137,   138,   139,   140,   141,   142,   143,   144,   145,
   146,   147,   148,   149,   150,   151,   152,   153,   154,   155,
   156,   157,   158,   159,   160,   161,   162,   163,   164,   165,
   166,   167,   168,   169,   170,   171,   172,   173,   174,   175,
   176,   177,   178,   179,   180,   181,   182,   183,   184,   185,
   186,   187,   188,   189,   190,   191,   192,   193,   194,   195,
   196,   197,   198,   199,   200,   201,   202,   203,   204,   205,
   206,   207,   208,   209,   210,   211,   212,   213,   214,   215,
   216,   217,   218,   219,   220,   221,   222,   223,   224,   225,
   226,   227,   228,   229,   230,   231,   232,   233,   234,   235,
   236,   237
};

#if YY_patBisonParam_DEBUG != 0
static const short yyprhs[] = {     0,
     0,     2,     4,     7,     9,    11,    13,    15,    17,    19,
    21,    23,    25,    27,    29,    31,    33,    35,    37,    40,
    42,    45,    49,    53,    57,    61,    65,    69,    73,    77,
    81,    85,    89,    93,    97,   101,   105,   109,   113,   117,
   121,   125,   129,   133,   137,   141,   145,   149,   153,   157,
   161,   165,   168,   170,   173,   177,   181,   185,   189,   193,
   197,   201,   205,   209,   213,   217,   221,   225,   229,   233,
   237,   241,   245,   249,   253,   257,   261,   265,   269,   273,
   277,   281,   285,   289,   293,   297,   301,   305,   309,   312,
   314,   317,   321,   325,   329,   333,   337,   341,   344,   346,
   349,   353,   357,   361,   365,   369,   373,   377,   381,   385,
   389,   393,   397,   401,   405,   409,   412,   414,   417,   421,
   425,   429,   433,   437,   441,   445,   449,   453,   457,   461,
   465,   469,   473,   476,   478,   481,   485,   489,   493,   497,
   501,   504,   506,   509,   513,   517,   521,   525,   529,   533,
   537,   541,   545,   549,   553,   557,   561,   565,   569,   573,
   577,   581,   585,   589,   593,   597,   601,   605,   609,   613,
   617,   621,   625,   629,   633,   637,   641,   645,   649,   653,
   657,   661,   665,   669,   673,   677,   681,   685,   689,   693,
   697,   701,   705,   709,   713,   717,   721,   725,   729,   733,
   737,   741,   745,   749,   753,   757,   761,   765,   769,   773,
   777,   781,   785,   789,   793,   797,   801,   805,   809,   813,
   817,   820,   822,   825,   829,   833,   837,   841,   844,   846,
   849,   853,   857,   861,   864,   866,   869,   873,   877,   881,
   884,   886,   889,   893,   897,   900,   902,   905,   909,   913,
   917,   921,   925,   929,   932,   934,   937,   941,   945,   949,
   952,   954,   957,   961,   965,   969,   973,   977,   980,   982,
   985,   989,   993,   997,  1001,  1003,  1005
};

static const short yyrhs[] = {   239,
     0,   240,     0,   239,   240,     0,   241,     0,   244,     0,
   247,     0,   250,     0,   253,     0,   256,     0,   259,     0,
   262,     0,   265,     0,   268,     0,   271,     0,   274,     0,
   277,     0,   280,     0,   283,     0,     3,   242,     0,   243,
     0,   242,   243,     0,     4,   229,   288,     0,     5,   229,
   287,     0,     6,   229,   287,     0,     7,   229,   288,     0,
     8,   229,   287,     0,     9,   229,   287,     0,    10,   229,
   287,     0,    11,   229,   287,     0,    12,   229,   287,     0,
    13,   229,   287,     0,    14,   229,   287,     0,    15,   229,
   288,     0,    16,   229,   288,     0,    17,   229,   288,     0,
    18,   229,   288,     0,    19,   229,   288,     0,    20,   229,
   288,     0,    21,   229,   288,     0,    22,   229,   288,     0,
    23,   229,   287,     0,    24,   229,   287,     0,    25,   229,
   287,     0,    26,   229,   287,     0,    27,   229,   288,     0,
    28,   229,   287,     0,    29,   229,   287,     0,    30,   229,
   287,     0,    31,   229,   287,     0,    32,   229,   287,     0,
    33,   229,   288,     0,    34,   245,     0,   246,     0,   245,
   246,     0,    35,   229,   287,     0,    36,   229,   287,     0,
    37,   229,   287,     0,    38,   229,   287,     0,    39,   229,
   287,     0,    40,   229,   287,     0,    41,   229,   287,     0,
    42,   229,   288,     0,    43,   229,   288,     0,    44,   229,
   288,     0,    45,   229,   288,     0,    46,   229,   288,     0,
    47,   229,   287,     0,    48,   229,   288,     0,    49,   229,
   287,     0,    50,   229,   287,     0,    51,   229,   287,     0,
    52,   229,   287,     0,    53,   229,   287,     0,    54,   229,
   287,     0,    55,   229,   287,     0,    56,   229,   287,     0,
    57,   229,   287,     0,    58,   229,   287,     0,    59,   229,
   287,     0,    60,   229,   287,     0,    61,   229,   287,     0,
    62,   229,   287,     0,    63,   229,   287,     0,    64,   229,
   287,     0,    65,   229,   287,     0,    66,   229,   287,     0,
    67,   229,   287,     0,    68,   229,   287,     0,    69,   248,
     0,   249,     0,   248,   249,     0,    70,   229,   288,     0,
    71,   229,   287,     0,    72,   229,   287,     0,    73,   229,
   287,     0,    74,   229,   288,     0,    75,   229,   287,     0,
    76,   251,     0,   252,     0,   251,   252,     0,    77,   229,
   288,     0,    78,   229,   287,     0,    79,   229,   287,     0,
    80,   229,   288,     0,    81,   229,   288,     0,    82,   229,
   287,     0,    83,   229,   287,     0,    84,   229,   287,     0,
    85,   229,   287,     0,    86,   229,   287,     0,    87,   229,
   288,     0,    88,   229,   287,     0,    89,   229,   287,     0,
    90,   229,   288,     0,    91,   229,   287,     0,    92,   254,
     0,   255,     0,   254,   255,     0,    93,   229,   286,     0,
    94,   229,   286,     0,    95,   229,   286,     0,    96,   229,
   286,     0,    97,   229,   286,     0,    98,   229,   286,     0,
    99,   229,   286,     0,   100,   229,   286,     0,   101,   229,
   286,     0,   102,   229,   286,     0,   103,   229,   286,     0,
   104,   229,   286,     0,   105,   229,   286,     0,   106,   229,
   286,     0,   107,   257,     0,   258,     0,   257,   258,     0,
   108,   229,   287,     0,   109,   229,   287,     0,   110,   229,
   288,     0,   111,   229,   288,     0,   112,   229,   288,     0,
   113,   260,     0,   261,     0,   260,   261,     0,   114,   229,
   286,     0,   115,   229,   286,     0,   116,   229,   286,     0,
   117,   229,   288,     0,   118,   229,   288,     0,   119,   229,
   288,     0,   120,   229,   288,     0,   121,   229,   287,     0,
   122,   229,   288,     0,   123,   229,   288,     0,   124,   229,
   288,     0,   125,   229,   288,     0,   126,   229,   288,     0,
   127,   229,   288,     0,   128,   229,   288,     0,   129,   229,
   288,     0,   130,   229,   288,     0,   131,   229,   286,     0,
   132,   229,   288,     0,   133,   229,   288,     0,   134,   229,
   288,     0,   135,   229,   286,     0,   136,   229,   286,     0,
   137,   229,   286,     0,   138,   229,   286,     0,   139,   229,
   288,     0,   140,   229,   286,     0,   141,   229,   288,     0,
   142,   229,   287,     0,   143,   229,   288,     0,   144,   229,
   288,     0,   145,   229,   288,     0,   146,   229,   288,     0,
   147,   229,   286,     0,   148,   229,   288,     0,   149,   229,
   288,     0,   150,   229,   288,     0,   151,   229,   286,     0,
   152,   229,   287,     0,   153,   229,   286,     0,   154,   229,
   286,     0,   155,   229,   286,     0,   156,   229,   288,     0,
   157,   229,   288,     0,   158,   229,   288,     0,   159,   229,
   287,     0,   160,   229,   288,     0,   161,   229,   288,     0,
   162,   229,   288,     0,   163,   229,   288,     0,   164,   229,
   288,     0,   165,   229,   288,     0,   166,   229,   288,     0,
   167,   229,   287,     0,   168,   229,   288,     0,   169,   229,
   287,     0,   170,   229,   287,     0,   171,   229,   288,     0,
   172,   229,   287,     0,   173,   229,   286,     0,   174,   229,
   288,     0,   175,   229,   288,     0,   176,   229,   288,     0,
   177,   229,   288,     0,   178,   229,   288,     0,   179,   229,
   288,     0,   180,   229,   288,     0,   181,   229,   288,     0,
   182,   229,   288,     0,   183,   229,   287,     0,   184,   229,
   288,     0,   185,   229,   287,     0,   186,   229,   288,     0,
   187,   229,   288,     0,   188,   229,   287,     0,   189,   229,
   287,     0,   190,   229,   288,     0,   191,   263,     0,   264,
     0,   263,   264,     0,   192,   229,   288,     0,   193,   229,
   288,     0,   194,   229,   288,     0,   195,   229,   288,     0,
   196,   266,     0,   267,     0,   266,   267,     0,   197,   229,
   288,     0,   198,   229,   288,     0,   199,   229,   288,     0,
   200,   269,     0,   270,     0,   269,   270,     0,   201,   229,
   287,     0,   202,   229,   287,     0,   203,   229,   287,     0,
   204,   272,     0,   273,     0,   272,   273,     0,   205,   229,
   287,     0,   206,   229,   287,     0,   207,   275,     0,   276,
     0,   275,   276,     0,   208,   229,   288,     0,   209,   229,
   288,     0,   210,   229,   288,     0,   211,   229,   287,     0,
   212,   229,   287,     0,   213,   229,   287,     0,   214,   278,
     0,   279,     0,   278,   279,     0,   215,   229,   288,     0,
   216,   229,   288,     0,   217,   229,   288,     0,   218,   281,
     0,   282,     0,   281,   282,     0,   219,   229,   287,     0,
   220,   229,   287,     0,   221,   229,   287,     0,   222,   229,
   287,     0,   223,   229,   288,     0,   224,   284,     0,   285,
     0,   284,   285,     0,   225,   229,   288,     0,   226,   229,
   288,     0,   227,   229,   287,     0,   228,   229,   287,     0,
   236,     0,   233,     0,   232,     0
};

#endif

#if YY_patBisonParam_DEBUG != 0
static const short yyrline[] = { 0,
   367,   371,   372,   374,   375,   376,   377,   378,   379,   380,
   381,   382,   383,   384,   385,   386,   387,   388,   390,   391,
   392,   394,   398,   403,   408,   413,   418,   423,   428,   433,
   438,   443,   448,   453,   458,   463,   469,   474,   479,   484,
   489,   494,   499,   504,   509,   514,   519,   524,   529,   534,
   539,   545,   546,   547,   549,   553,   558,   563,   568,   573,
   578,   583,   588,   593,   598,   603,   608,   613,   618,   623,
   628,   633,   638,   643,   648,   653,   658,   663,   668,   673,
   678,   683,   688,   693,   698,   703,   708,   713,   719,   720,
   721,   723,   727,   732,   737,   742,   747,   753,   754,   755,
   757,   761,   766,   771,   776,   781,   786,   791,   796,   801,
   806,   811,   816,   821,   826,   832,   833,   834,   836,   841,
   847,   853,   859,   865,   871,   877,   883,   889,   895,   901,
   907,   913,   920,   921,   922,   924,   928,   933,   938,   943,
   949,   950,   951,   953,   958,   964,   970,   975,   980,   985,
   990,   995,  1000,  1005,  1010,  1015,  1020,  1025,  1030,  1035,
  1040,  1046,  1051,  1056,  1061,  1067,  1073,  1079,  1085,  1090,
  1096,  1101,  1106,  1111,  1116,  1121,  1126,  1132,  1137,  1142,
  1147,  1153,  1158,  1164,  1170,  1176,  1181,  1186,  1191,  1196,
  1201,  1206,  1211,  1216,  1221,  1226,  1231,  1236,  1241,  1246,
  1251,  1256,  1261,  1267,  1272,  1277,  1282,  1287,  1292,  1297,
  1302,  1307,  1312,  1317,  1322,  1327,  1332,  1337,  1342,  1347,
  1353,  1354,  1355,  1357,  1361,  1366,  1371,  1377,  1378,  1379,
  1381,  1385,  1390,  1396,  1397,  1398,  1400,  1404,  1409,  1415,
  1416,  1417,  1419,  1423,  1429,  1430,  1431,  1433,  1437,  1442,
  1447,  1452,  1457,  1463,  1464,  1465,  1467,  1471,  1476,  1482,
  1483,  1484,  1486,  1490,  1495,  1500,  1505,  1511,  1512,  1513,
  1515,  1519,  1524,  1529,  1536,  1541,  1546
};

static const char * const yytname[] = {   "$","error","$illegal.","pat_BasicTrustRegionSection",
"pat_BTRMaxGcpIter","pat_BTRArmijoBeta1","pat_BTRArmijoBeta2","pat_BTRStartDraws",
"pat_BTRIncreaseDraws","pat_BTREta1","pat_BTREta2","pat_BTRGamma1","pat_BTRGamma2",
"pat_BTRInitRadius","pat_BTRIncreaseTRRadius","pat_BTRUnfeasibleCGIterations",
"pat_BTRForceExactHessianIfMnl","pat_BTRExactHessian","pat_BTRCheapHessian",
"pat_BTRQuasiNewtonUpdate","pat_BTRInitQuasiNewtonWithTrueHessian","pat_BTRInitQuasiNewtonWithBHHH",
"pat_BTRMaxIter","pat_BTRTypf","pat_BTRTolerance","pat_BTRMaxTRRadius","pat_BTRMinTRRadius",
"pat_BTRUsePreconditioner","pat_BTRSingularityThreshold","pat_BTRKappaEpp","pat_BTRKappaLbs",
"pat_BTRKappaUbs","pat_BTRKappaFrd","pat_BTRSignificantDigits","pat_CondTrustRegionSection",
"pat_CTRAETA0","pat_CTRAETA1","pat_CTRAETA2","pat_CTRAGAMMA1","pat_CTRAGAMMA2",
"pat_CTRAEPSILONC","pat_CTRAALPHA","pat_CTRAMU","pat_CTRAMAXNBRFUNCTEVAL","pat_CTRAMAXLENGTH",
"pat_CTRAMAXDATA","pat_CTRANBROFBESTPTS","pat_CTRAPOWER","pat_CTRAMAXRAD","pat_CTRAMINRAD",
"pat_CTRAUPPERBOUND","pat_CTRALOWERBOUND","pat_CTRAGAMMA3","pat_CTRAGAMMA4",
"pat_CTRACOEFVALID","pat_CTRACOEFGEN","pat_CTRAEPSERROR","pat_CTRAEPSPOINT",
"pat_CTRACOEFNORM","pat_CTRAMINSTEP","pat_CTRAMINPIVOTVALUE","pat_CTRAGOODPIVOTVALUE",
"pat_CTRAFINEPS","pat_CTRAFINEPSREL","pat_CTRACHECKEPS","pat_CTRACHECKTESTEPS",
"pat_CTRACHECKTESTEPSREL","pat_CTRAVALMINGAUSS","pat_CTRAFACTOFPOND","pat_ConjugateGradientSection",
"pat_Precond","pat_Epsilon","pat_CondLimit","pat_PrecResidu","pat_MaxCGIter",
"pat_TolSchnabelEskow","pat_DefaultValuesSection","pat_MaxIter","pat_InitStep",
"pat_MinStep","pat_MaxEval","pat_NbrRun","pat_MaxStep","pat_AlphaProba","pat_StepReduc",
"pat_StepIncr","pat_ExpectedImprovement","pat_AllowPremUnsucc","pat_PrematureStart",
"pat_PrematureStep","pat_MaxUnsuccIter","pat_NormWeight","pat_FilesSection",
"pat_InputDirectory","pat_OutputDirectory","pat_TmpDirectory","pat_FunctionEvalExec",
"pat_jonSimulator","pat_CandidateFile","pat_ResultFile","pat_OutsifFile","pat_LogFile",
"pat_ProblemsFile","pat_MITSIMorigin","pat_MITSIMinformation","pat_MITSIMtravelTime",
"pat_MITSIMexec","pat_Formule1Section","pat_AugmentationStep","pat_ReductionStep",
"pat_SubSpaceMaxIter","pat_SubSpaceConsecutiveFailure","pat_WarmUpnbre","pat_GEVSection",
"pat_gevInputDirectory","pat_gevOutputDirectory","pat_gevWorkingDirectory","pat_gevSignificantDigitsParameters",
"pat_gevDecimalDigitsTTest","pat_gevDecimalDigitsStats","pat_gevForceScientificNotation",
"pat_gevSingularValueThreshold","pat_gevPrintVarCovarAsList","pat_gevPrintVarCovarAsMatrix",
"pat_gevPrintPValue","pat_gevNumberOfThreads","pat_gevSaveIntermediateResults",
"pat_gevVarCovarFromBHHH","pat_gevDebugDataFirstRow","pat_gevDebugDataLastRow",
"pat_gevStoreDataOnFile","pat_gevBinaryDataFile","pat_gevDumpDrawsOnFile","pat_gevReadDrawsFromFile",
"pat_gevGenerateActualSample","pat_gevOutputActualSample","pat_gevNormalDrawsFile",
"pat_gevRectangularDrawsFile","pat_gevRandomDistrib","pat_gevMaxPrimeNumber",
"pat_gevWarningSign","pat_gevWarningLowDraws","pat_gevMissingValue","pat_gevGenerateFilesForDenis",
"pat_gevGenerateGnuplotFile","pat_gevGeneratePythonFile","pat_gevPythonFileWithEstimatedParam",
"pat_gevFileForDenis","pat_gevAutomaticScalingOfLinearUtility","pat_gevInverseIteration",
"pat_gevSeed","pat_gevOne","pat_gevMinimumMu","pat_gevSummaryParameters","pat_gevSummaryFile",
"pat_gevStopFileName","pat_gevCheckDerivatives","pat_gevBufferSize","pat_gevDataFileDisplayStep",
"pat_gevTtestThreshold","pat_gevGlobal","pat_gevAnalGrad","pat_gevAnalHess",
"pat_gevCheapF","pat_gevFactSec","pat_gevTermCode","pat_gevTypx","pat_gevTypF",
"pat_gevFDigits","pat_gevGradTol","pat_gevMaxStep","pat_gevItnLimit","pat_gevDelta",
"pat_gevAlgo","pat_gevScreenPrintLevel","pat_gevLogFilePrintLevel","pat_gevGeneratedGroups",
"pat_gevGeneratedData","pat_gevGeneratedAttr","pat_gevGeneratedAlt","pat_gevSubSampleLevel",
"pat_gevSubSampleBasis","pat_gevComputeLastHessian","pat_gevEigenvalueThreshold",
"pat_gevNonParamPlotRes","pat_gevNonParamPlotMaxY","pat_gevNonParamPlotXSizeCm",
"pat_gevNonParamPlotYSizeCm","pat_gevNonParamPlotMinXSizeCm","pat_gevNonParamPlotMinYSizeCm",
"pat_svdMaxIter","pat_HieLoWSection","pat_hieMultinomial","pat_hieTruncStructUtil",
"pat_hieUpdateHessien","pat_hieDateInLog","pat_LogitKernelFortranSection","pat_bolducMaxAlts",
"pat_bolducMaxFact","pat_bolducMaxNVar","pat_NewtonLikeSection","pat_StepSecondIndividual",
"pat_NLgWeight","pat_NLhWeight","pat_TointSteihaugSection","pat_TSFractionGradientRequired",
"pat_TSExpTheta","pat_cfsqpSection","pat_cfsqpMode","pat_cfsqpIprint","pat_cfsqpMaxIter",
"pat_cfsqpEps","pat_cfsqpEpsEqn","pat_cfsqpUdelta","pat_dfoSection","pat_dfoAddToLWRK",
"pat_dfoAddToLIWRK","pat_dfoMaxFunEval","pat_donlp2Section","pat_donlp2Epsx",
"pat_donlp2Delmin","pat_donlp2Smallw","pat_donlp2Epsdif","pat_donlp2NReset",
"pat_solvoptSection","pat_solvoptMaxIter","pat_solvoptDisplay","pat_solvoptErrorArgument",
"pat_solvoptErrorFunction","patEQUAL","patOB","patCB","patINT","patREAL","patTIME",
"patNAME","patSTRING","patPAIR","everything","sections","section","yBasicTrustRegion",
"yBasicTrustRegion_entries","yBasicTrustRegion_entry","yCondTrustRegion","yCondTrustRegion_entries",
"yCondTrustRegion_entry","yConjugateGradient","yConjugateGradient_entries","yConjugateGradient_entry",
"yDefaultValues","yDefaultValues_entries","yDefaultValues_entry","yFiles","yFiles_entries",
"yFiles_entry","yFormule1","yFormule1_entries","yFormule1_entry","yGEV","yGEV_entries",
"yGEV_entry","yHieLoW","yHieLoW_entries","yHieLoW_entry","yLogitKernelFortran",
"yLogitKernelFortran_entries","yLogitKernelFortran_entry","yNewtonLike","yNewtonLike_entries",
"yNewtonLike_entry","yTointSteihaug","yTointSteihaug_entries","yTointSteihaug_entry",
"ycfsqp","ycfsqp_entries","ycfsqp_entry","ydfo","ydfo_entries","ydfo_entry",
"ydonlp2","ydonlp2_entries","ydonlp2_entry","ysolvopt","ysolvopt_entries","ysolvopt_entry",
"stringParam","floatParam","longParam",""
};
#endif

static const short yyr1[] = {     0,
   238,   239,   239,   240,   240,   240,   240,   240,   240,   240,
   240,   240,   240,   240,   240,   240,   240,   240,   241,   242,
   242,   243,   243,   243,   243,   243,   243,   243,   243,   243,
   243,   243,   243,   243,   243,   243,   243,   243,   243,   243,
   243,   243,   243,   243,   243,   243,   243,   243,   243,   243,
   243,   244,   245,   245,   246,   246,   246,   246,   246,   246,
   246,   246,   246,   246,   246,   246,   246,   246,   246,   246,
   246,   246,   246,   246,   246,   246,   246,   246,   246,   246,
   246,   246,   246,   246,   246,   246,   246,   246,   247,   248,
   248,   249,   249,   249,   249,   249,   249,   250,   251,   251,
   252,   252,   252,   252,   252,   252,   252,   252,   252,   252,
   252,   252,   252,   252,   252,   253,   254,   254,   255,   255,
   255,   255,   255,   255,   255,   255,   255,   255,   255,   255,
   255,   255,   256,   257,   257,   258,   258,   258,   258,   258,
   259,   260,   260,   261,   261,   261,   261,   261,   261,   261,
   261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
   261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
   261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
   261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
   261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
   261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
   261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
   262,   263,   263,   264,   264,   264,   264,   265,   266,   266,
   267,   267,   267,   268,   269,   269,   270,   270,   270,   271,
   272,   272,   273,   273,   274,   275,   275,   276,   276,   276,
   276,   276,   276,   277,   278,   278,   279,   279,   279,   280,
   281,   281,   282,   282,   282,   282,   282,   283,   284,   284,
   285,   285,   285,   285,   286,   287,   288
};

static const short yyr2[] = {     0,
     1,     1,     2,     1,     1,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     1,     2,     1,
     2,     3,     3,     3,     3,     3,     3,     3,     3,     3,
     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
     3,     2,     1,     2,     3,     3,     3,     3,     3,     3,
     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
     3,     3,     3,     3,     3,     3,     3,     3,     2,     1,
     2,     3,     3,     3,     3,     3,     3,     2,     1,     2,
     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
     3,     3,     3,     3,     3,     2,     1,     2,     3,     3,
     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
     3,     3,     2,     1,     2,     3,     3,     3,     3,     3,
     2,     1,     2,     3,     3,     3,     3,     3,     3,     3,
     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
     2,     1,     2,     3,     3,     3,     3,     2,     1,     2,
     3,     3,     3,     2,     1,     2,     3,     3,     3,     2,
     1,     2,     3,     3,     2,     1,     2,     3,     3,     3,
     3,     3,     3,     2,     1,     2,     3,     3,     3,     2,
     1,     2,     3,     3,     3,     3,     3,     2,     1,     2,
     3,     3,     3,     3,     1,     1,     1
};

static const short yydefact[] = {     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     1,     2,     4,     5,     6,
     7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
    17,    18,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,    19,    20,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,    52,    53,
     0,     0,     0,     0,     0,     0,    89,    90,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,    98,    99,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,   116,
   117,     0,     0,     0,     0,     0,   133,   134,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,   141,   142,     0,     0,     0,
     0,   221,   222,     0,     0,     0,   228,   229,     0,     0,
     0,   234,   235,     0,     0,   240,   241,     0,     0,     0,
     0,     0,     0,   245,   246,     0,     0,     0,   254,   255,
     0,     0,     0,     0,     0,   260,   261,     0,     0,     0,
     0,   268,   269,     3,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,    21,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,    54,
     0,     0,     0,     0,     0,     0,    91,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   100,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,   118,     0,     0,
     0,     0,     0,   135,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,   143,     0,     0,     0,     0,   223,     0,     0,     0,
   230,     0,     0,     0,   236,     0,     0,   242,     0,     0,
     0,     0,     0,     0,   247,     0,     0,     0,   256,     0,
     0,     0,     0,     0,   262,     0,     0,     0,     0,   270,
   277,    22,   276,    23,    24,    25,    26,    27,    28,    29,
    30,    31,    32,    33,    34,    35,    36,    37,    38,    39,
    40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
    50,    51,    55,    56,    57,    58,    59,    60,    61,    62,
    63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
    73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
    83,    84,    85,    86,    87,    88,    92,    93,    94,    95,
    96,    97,   101,   102,   103,   104,   105,   106,   107,   108,
   109,   110,   111,   112,   113,   114,   115,   275,   119,   120,
   121,   122,   123,   124,   125,   126,   127,   128,   129,   130,
   131,   132,   136,   137,   138,   139,   140,   144,   145,   146,
   147,   148,   149,   150,   151,   152,   153,   154,   155,   156,
   157,   158,   159,   160,   161,   162,   163,   164,   165,   166,
   167,   168,   169,   170,   171,   172,   173,   174,   175,   176,
   177,   178,   179,   180,   181,   182,   183,   184,   185,   186,
   187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
   197,   198,   199,   200,   201,   202,   203,   204,   205,   206,
   207,   208,   209,   210,   211,   212,   213,   214,   215,   216,
   217,   218,   219,   220,   224,   225,   226,   227,   231,   232,
   233,   237,   238,   239,   243,   244,   248,   249,   250,   251,
   252,   253,   257,   258,   259,   263,   264,   265,   266,   267,
   271,   272,   273,   274,     0,     0,     0
};

static const short yydefgoto[] = {   715,
    16,    17,    18,    63,    64,    19,    99,   100,    20,   107,
   108,    21,   124,   125,    22,   140,   141,    23,   147,   148,
    24,   226,   227,    25,   232,   233,    26,   237,   238,    27,
   242,   243,    28,   246,   247,    29,   254,   255,    30,   259,
   260,    31,   266,   267,    32,   272,   273,   589,   504,   502
};

static const short yypact[] = {    36,
   660,   595,   -61,     9,   202,     8,   439,   -63,  -161,   -68,
  -190,  -102,   -65,   -98,   -88,    36,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,  -228,  -208,  -202,  -201,  -189,  -187,  -166,  -165,
  -158,  -155,  -154,  -148,  -145,  -128,  -125,  -116,  -115,  -103,
   -87,   -85,   -84,   -82,   -81,   -75,   -74,   -73,   -72,   -71,
   -70,   -69,   660,-32768,   -67,   -64,   -62,   -60,   -59,   -58,
   -57,   -56,   -55,   -54,   -53,   -51,   -49,   -48,   -45,   -44,
   -43,   -42,   -41,   -40,   -39,   -38,   -37,   -36,   -35,   -31,
   -28,   -27,   -26,   -21,   -20,   -19,   -18,   -17,   595,-32768,
   -12,   -11,   -10,    -9,    -6,    -5,   -61,-32768,    -4,    -3,
    -1,     0,     1,     2,     4,     5,     6,    10,    12,    13,
    15,    16,    17,     9,-32768,    18,    19,    20,    22,    23,
    26,    27,    28,    29,    30,    40,    41,    42,    43,   202,
-32768,    45,    46,    47,    48,    49,     8,-32768,    51,    52,
    53,    54,    55,    56,    57,    58,    64,    80,    81,    82,
    83,    84,    85,    86,    88,    89,    90,    92,    93,    95,
    96,    99,   100,   101,   102,   103,   105,   106,   109,   110,
   111,   112,   113,   114,   115,   116,   117,   118,   119,   120,
   121,   122,   123,   124,   125,   126,   130,   131,   132,   133,
   138,   148,   152,   153,   154,   155,   157,   159,   164,   168,
   169,   170,   171,   172,   176,   184,   186,   187,   189,   190,
   200,   205,   206,   208,   213,   439,-32768,   217,   218,   219,
   220,   -63,-32768,   221,   222,   223,  -161,-32768,   224,   228,
   229,   -68,-32768,   230,   231,  -190,-32768,   235,   236,   237,
   238,   239,   241,  -102,-32768,   257,   258,   259,   -65,-32768,
   260,   261,   262,   263,   267,   -98,-32768,   268,   269,   270,
   271,   -88,-32768,-32768,  -164,  -106,  -106,  -164,  -106,  -106,
  -106,  -106,  -106,  -106,  -106,  -164,  -164,  -164,  -164,  -164,
  -164,  -164,  -164,  -106,  -106,  -106,  -106,  -164,  -106,  -106,
  -106,  -106,  -106,  -164,-32768,  -106,  -106,  -106,  -106,  -106,
  -106,  -106,  -164,  -164,  -164,  -164,  -164,  -106,  -164,  -106,
  -106,  -106,  -106,  -106,  -106,  -106,  -106,  -106,  -106,  -106,
  -106,  -106,  -106,  -106,  -106,  -106,  -106,  -106,  -106,-32768,
  -164,  -106,  -106,  -106,  -164,  -106,-32768,  -164,  -106,  -106,
  -164,  -164,  -106,  -106,  -106,  -106,  -106,  -164,  -106,  -106,
  -164,  -106,-32768,   -95,   -95,   -95,   -95,   -95,   -95,   -95,
   -95,   -95,   -95,   -95,   -95,   -95,   -95,-32768,  -106,  -106,
  -164,  -164,  -164,-32768,   -95,   -95,   -95,  -164,  -164,  -164,
  -164,  -106,  -164,  -164,  -164,  -164,  -164,  -164,  -164,  -164,
  -164,   -95,  -164,  -164,  -164,   -95,   -95,   -95,   -95,  -164,
   -95,  -164,  -106,  -164,  -164,  -164,  -164,   -95,  -164,  -164,
  -164,   -95,  -106,   -95,   -95,   -95,  -164,  -164,  -164,  -106,
  -164,  -164,  -164,  -164,  -164,  -164,  -164,  -106,  -164,  -106,
  -106,  -164,  -106,   -95,  -164,  -164,  -164,  -164,  -164,  -164,
  -164,  -164,  -164,  -106,  -164,  -106,  -164,  -164,  -106,  -106,
  -164,-32768,  -164,  -164,  -164,  -164,-32768,  -164,  -164,  -164,
-32768,  -106,  -106,  -106,-32768,  -106,  -106,-32768,  -164,  -164,
  -164,  -106,  -106,  -106,-32768,  -164,  -164,  -164,-32768,  -106,
  -106,  -106,  -106,  -164,-32768,  -164,  -164,  -106,  -106,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,   204,   337,-32768
};

static const short yypgoto[] = {-32768,
-32768,   415,-32768,-32768,   174,-32768,-32768,    69,-32768,-32768,
   394,-32768,-32768,   378,-32768,-32768,    98,-32768,-32768,   356,
-32768,-32768,   278,-32768,-32768,   273,-32768,-32768,   272,-32768,
-32768,   264,-32768,-32768,   265,-32768,-32768,   253,-32768,-32768,
   249,-32768,-32768,   246,-32768,-32768,   248,   108,  -277,   -25
};


#define	YYLAST		693


static const short yytable[] = {   505,
   275,   507,   508,   509,   510,   511,   512,   513,   101,   102,
   103,   104,   105,   106,   244,   245,   522,   523,   524,   525,
   276,   527,   528,   529,   530,   531,   277,   278,   533,   534,
   535,   536,   537,   538,   539,   234,   235,   236,     1,   279,
   545,   280,   547,   548,   549,   550,   551,   552,   553,   554,
   555,   556,   557,   558,   559,   560,   561,   562,   563,   564,
   565,   566,   281,   282,   568,   569,   570,   501,   572,     2,
   283,   574,   575,   284,   285,   578,   579,   580,   581,   582,
   286,   584,   585,   287,   587,   109,   110,   111,   112,   113,
   114,   115,   116,   117,   118,   119,   120,   121,   122,   123,
   288,   603,   604,   289,     3,   248,   249,   250,   251,   252,
   253,     4,   290,   291,   615,   142,   143,   144,   145,   146,
   261,   262,   263,   264,   265,   292,   503,     5,   228,   229,
   230,   231,   239,   240,   241,   636,   268,   269,   270,   271,
   588,   293,     6,   294,   295,   646,   296,   297,     7,   256,
   257,   258,   653,   298,   299,   300,   301,   302,   303,   304,
   661,   306,   663,   664,   307,   666,   308,   340,   309,   310,
   311,   312,   313,   314,   315,   316,   677,   317,   679,   318,
   319,   682,   683,   320,   321,   322,   323,   324,   325,   326,
   327,   328,   329,   330,   692,   693,   694,   331,   695,   696,
   332,   333,   334,   716,   700,   701,   702,   335,   336,   337,
   338,   339,   706,   707,   708,   709,   341,   342,   343,   344,
   713,   714,   345,   346,   348,   349,     8,   350,   351,   352,
   353,     9,   354,   355,   356,    10,   305,   378,   357,    11,
   358,   359,    12,   360,   361,   362,   364,   365,   366,    13,
   367,   368,   506,    14,   369,   370,   371,   372,   373,    15,
   514,   515,   516,   517,   518,   519,   520,   521,   374,   375,
   376,   377,   526,   379,   380,   381,   382,   383,   532,   385,
   386,   387,   388,   389,   390,   391,   392,   540,   541,   542,
   543,   544,   393,   546,   126,   127,   128,   129,   130,   131,
   132,   133,   134,   135,   136,   137,   138,   139,   394,   395,
   396,   397,   398,   399,   400,   567,   401,   402,   403,   571,
   404,   405,   573,   406,   407,   576,   577,   408,   409,   410,
   411,   412,   583,   413,   414,   586,   717,   415,   416,   417,
   418,   419,   420,   421,   422,   423,   424,   425,   426,   427,
   428,   429,   430,   431,   432,   605,   606,   607,   433,   434,
   435,   436,   611,   612,   613,   614,   437,   616,   617,   618,
   619,   620,   621,   622,   623,   624,   438,   626,   627,   628,
   439,   440,   441,   442,   633,   443,   635,   444,   637,   638,
   639,   640,   445,   642,   643,   644,   446,   447,   448,   449,
   450,   650,   651,   652,   451,   654,   655,   656,   657,   658,
   659,   660,   452,   662,   453,   454,   665,   455,   456,   668,
   669,   670,   671,   672,   673,   674,   675,   676,   457,   678,
   274,   680,   681,   458,   459,   684,   460,   685,   686,   687,
   688,   461,   689,   690,   691,   463,   464,   465,   466,   468,
   469,   470,   472,   697,   698,   699,   473,   474,   476,   477,
   703,   704,   705,   479,   480,   481,   482,   483,   710,   484,
   711,   712,   590,   591,   592,   593,   594,   595,   596,   597,
   598,   599,   600,   601,   602,   486,   487,   488,   490,   491,
   492,   493,   608,   609,   610,   494,   496,   497,   498,   499,
   347,   363,   384,   462,   467,   475,   485,   489,   471,   625,
   478,   495,     0,   629,   630,   631,   632,     0,   634,   500,
     0,     0,     0,     0,     0,   641,     0,     0,     0,   645,
     0,   647,   648,   649,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,   667,   149,   150,   151,   152,   153,   154,   155,   156,
   157,   158,   159,   160,   161,   162,   163,   164,   165,   166,
   167,   168,   169,   170,   171,   172,   173,   174,   175,   176,
   177,   178,   179,   180,   181,   182,   183,   184,   185,   186,
   187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
   197,   198,   199,   200,   201,   202,   203,   204,   205,   206,
   207,   208,   209,   210,   211,   212,   213,   214,   215,   216,
   217,   218,   219,   220,   221,   222,   223,   224,   225,    65,
    66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
    76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
    86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
    96,    97,    98,    33,    34,    35,    36,    37,    38,    39,
    40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
    50,    51,    52,    53,    54,    55,    56,    57,    58,    59,
    60,    61,    62
};

static const short yycheck[] = {   277,
   229,   279,   280,   281,   282,   283,   284,   285,    70,    71,
    72,    73,    74,    75,   205,   206,   294,   295,   296,   297,
   229,   299,   300,   301,   302,   303,   229,   229,   306,   307,
   308,   309,   310,   311,   312,   197,   198,   199,     3,   229,
   318,   229,   320,   321,   322,   323,   324,   325,   326,   327,
   328,   329,   330,   331,   332,   333,   334,   335,   336,   337,
   338,   339,   229,   229,   342,   343,   344,   232,   346,    34,
   229,   349,   350,   229,   229,   353,   354,   355,   356,   357,
   229,   359,   360,   229,   362,    77,    78,    79,    80,    81,
    82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
   229,   379,   380,   229,    69,   208,   209,   210,   211,   212,
   213,    76,   229,   229,   392,   108,   109,   110,   111,   112,
   219,   220,   221,   222,   223,   229,   233,    92,   192,   193,
   194,   195,   201,   202,   203,   413,   225,   226,   227,   228,
   236,   229,   107,   229,   229,   423,   229,   229,   113,   215,
   216,   217,   430,   229,   229,   229,   229,   229,   229,   229,
   438,   229,   440,   441,   229,   443,   229,    99,   229,   229,
   229,   229,   229,   229,   229,   229,   454,   229,   456,   229,
   229,   459,   460,   229,   229,   229,   229,   229,   229,   229,
   229,   229,   229,   229,   472,   473,   474,   229,   476,   477,
   229,   229,   229,     0,   482,   483,   484,   229,   229,   229,
   229,   229,   490,   491,   492,   493,   229,   229,   229,   229,
   498,   499,   229,   229,   229,   229,   191,   229,   229,   229,
   229,   196,   229,   229,   229,   200,    63,   140,   229,   204,
   229,   229,   207,   229,   229,   229,   229,   229,   229,   214,
   229,   229,   278,   218,   229,   229,   229,   229,   229,   224,
   286,   287,   288,   289,   290,   291,   292,   293,   229,   229,
   229,   229,   298,   229,   229,   229,   229,   229,   304,   229,
   229,   229,   229,   229,   229,   229,   229,   313,   314,   315,
   316,   317,   229,   319,    93,    94,    95,    96,    97,    98,
    99,   100,   101,   102,   103,   104,   105,   106,   229,   229,
   229,   229,   229,   229,   229,   341,   229,   229,   229,   345,
   229,   229,   348,   229,   229,   351,   352,   229,   229,   229,
   229,   229,   358,   229,   229,   361,     0,   229,   229,   229,
   229,   229,   229,   229,   229,   229,   229,   229,   229,   229,
   229,   229,   229,   229,   229,   381,   382,   383,   229,   229,
   229,   229,   388,   389,   390,   391,   229,   393,   394,   395,
   396,   397,   398,   399,   400,   401,   229,   403,   404,   405,
   229,   229,   229,   229,   410,   229,   412,   229,   414,   415,
   416,   417,   229,   419,   420,   421,   229,   229,   229,   229,
   229,   427,   428,   429,   229,   431,   432,   433,   434,   435,
   436,   437,   229,   439,   229,   229,   442,   229,   229,   445,
   446,   447,   448,   449,   450,   451,   452,   453,   229,   455,
    16,   457,   458,   229,   229,   461,   229,   463,   464,   465,
   466,   229,   468,   469,   470,   229,   229,   229,   229,   229,
   229,   229,   229,   479,   480,   481,   229,   229,   229,   229,
   486,   487,   488,   229,   229,   229,   229,   229,   494,   229,
   496,   497,   365,   366,   367,   368,   369,   370,   371,   372,
   373,   374,   375,   376,   377,   229,   229,   229,   229,   229,
   229,   229,   385,   386,   387,   229,   229,   229,   229,   229,
   107,   124,   147,   226,   232,   242,   254,   259,   237,   402,
   246,   266,    -1,   406,   407,   408,   409,    -1,   411,   272,
    -1,    -1,    -1,    -1,    -1,   418,    -1,    -1,    -1,   422,
    -1,   424,   425,   426,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,   444,   114,   115,   116,   117,   118,   119,   120,   121,
   122,   123,   124,   125,   126,   127,   128,   129,   130,   131,
   132,   133,   134,   135,   136,   137,   138,   139,   140,   141,
   142,   143,   144,   145,   146,   147,   148,   149,   150,   151,
   152,   153,   154,   155,   156,   157,   158,   159,   160,   161,
   162,   163,   164,   165,   166,   167,   168,   169,   170,   171,
   172,   173,   174,   175,   176,   177,   178,   179,   180,   181,
   182,   183,   184,   185,   186,   187,   188,   189,   190,    35,
    36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
    46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
    56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
    66,    67,    68,     4,     5,     6,     7,     8,     9,    10,
    11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
    21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
    31,    32,    33
};

#line 325 "/usr/local/lib/bison.cc"
 /* fattrs + tables */

/* parser code folow  */


/* This is the parser code that is written into each bison parser
  when the %semantic_parser declaration is not specified in the grammar.
  It was written by Richard Stallman by simplifying the hairy parser
  used when %semantic_parser is specified.  */

/* Note: dollar marks section change
   the next  is replaced by the list of actions, each action
   as one case of the switch.  */ 

#if YY_patBisonParam_USE_GOTO != 0
/* 
 SUPRESSION OF GOTO : on some C++ compiler (sun c++)
  the goto is strictly forbidden if any constructor/destructor
  is used in the whole function (very stupid isn't it ?)
 so goto are to be replaced with a 'while/switch/case construct'
 here are the macro to keep some apparent compatibility
*/
#define YYGOTO(lb) {yy_gotostate=lb;continue;}
#define YYBEGINGOTO  enum yy_labels yy_gotostate=yygotostart; \
                     for(;;) switch(yy_gotostate) { case yygotostart: {
#define YYLABEL(lb) } case lb: {
#define YYENDGOTO } } 
#define YYBEGINDECLARELABEL enum yy_labels {yygotostart
#define YYDECLARELABEL(lb) ,lb
#define YYENDDECLARELABEL  };
#else
/* macro to keep goto */
#define YYGOTO(lb) goto lb
#define YYBEGINGOTO 
#define YYLABEL(lb) lb:
#define YYENDGOTO
#define YYBEGINDECLARELABEL 
#define YYDECLARELABEL(lb)
#define YYENDDECLARELABEL 
#endif
/* LABEL DECLARATION */
YYBEGINDECLARELABEL
  YYDECLARELABEL(yynewstate)
  YYDECLARELABEL(yybackup)
/* YYDECLARELABEL(yyresume) */
  YYDECLARELABEL(yydefault)
  YYDECLARELABEL(yyreduce)
  YYDECLARELABEL(yyerrlab)   /* here on detecting error */
  YYDECLARELABEL(yyerrlab1)   /* here on error raised explicitly by an action */
  YYDECLARELABEL(yyerrdefault)  /* current state does not do anything special for the error token. */
  YYDECLARELABEL(yyerrpop)   /* pop the current state because it cannot handle the error token */
  YYDECLARELABEL(yyerrhandle)  
YYENDDECLARELABEL
/* ALLOCA SIMULATION */
/* __HAVE_NO_ALLOCA */
#ifdef __HAVE_NO_ALLOCA
int __alloca_free_ptr(char *ptr,char *ref)
{if(ptr!=ref) free(ptr);
 return 0;}

#define __ALLOCA_alloca(size) malloc(size)
#define __ALLOCA_free(ptr,ref) __alloca_free_ptr((char *)ptr,(char *)ref)

#ifdef YY_patBisonParam_LSP_NEEDED
#define __ALLOCA_return(num) \
            return( __ALLOCA_free(yyss,yyssa)+\
		    __ALLOCA_free(yyvs,yyvsa)+\
		    __ALLOCA_free(yyls,yylsa)+\
		   (num))
#else
#define __ALLOCA_return(num) \
            return( __ALLOCA_free(yyss,yyssa)+\
		    __ALLOCA_free(yyvs,yyvsa)+\
		   (num))
#endif
#else
#define __ALLOCA_return(num) return(num)
#define __ALLOCA_alloca(size) alloca(size)
#define __ALLOCA_free(ptr,ref) 
#endif

/* ENDALLOCA SIMULATION */

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (YY_patBisonParam_CHAR = YYEMPTY)
#define YYEMPTY         -2
#define YYEOF           0
#define YYACCEPT        __ALLOCA_return(0)
#define YYABORT         __ALLOCA_return(1)
#define YYERROR         YYGOTO(yyerrlab1)
/* Like YYERROR except do call yyerror.
   This remains here temporarily to ease the
   transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */
#define YYFAIL          YYGOTO(yyerrlab)
#define YYRECOVERING()  (!!yyerrstatus)
#define YYBACKUP(token, value) \
do                                                              \
  if (YY_patBisonParam_CHAR == YYEMPTY && yylen == 1)                               \
    { YY_patBisonParam_CHAR = (token), YY_patBisonParam_LVAL = (value);                 \
      yychar1 = YYTRANSLATE (YY_patBisonParam_CHAR);                                \
      YYPOPSTACK;                                               \
      YYGOTO(yybackup);                                            \
    }                                                           \
  else                                                          \
    { YY_patBisonParam_ERROR ("syntax error: cannot back up"); YYERROR; }   \
while (0)

#define YYTERROR        1
#define YYERRCODE       256

#ifndef YY_patBisonParam_PURE
/* UNPURE */
#define YYLEX           YY_patBisonParam_LEX()
#ifndef YY_USE_CLASS
/* If nonreentrant, and not class , generate the variables here */
int     YY_patBisonParam_CHAR;                      /*  the lookahead symbol        */
YY_patBisonParam_STYPE      YY_patBisonParam_LVAL;              /*  the semantic value of the */
				/*  lookahead symbol    */
int YY_patBisonParam_NERRS;                 /*  number of parse errors so far */
#ifdef YY_patBisonParam_LSP_NEEDED
YY_patBisonParam_LTYPE YY_patBisonParam_LLOC;   /*  location data for the lookahead     */
			/*  symbol                              */
#endif
#endif


#else
/* PURE */
#ifdef YY_patBisonParam_LSP_NEEDED
#define YYLEX           YY_patBisonParam_LEX(&YY_patBisonParam_LVAL, &YY_patBisonParam_LLOC)
#else
#define YYLEX           YY_patBisonParam_LEX(&YY_patBisonParam_LVAL)
#endif
#endif
#ifndef YY_USE_CLASS
#if YY_patBisonParam_DEBUG != 0
int YY_patBisonParam_DEBUG_FLAG;                    /*  nonzero means print parse trace     */
/* Since this is uninitialized, it does not stop multiple parsers
   from coexisting.  */
#endif
#endif



/*  YYINITDEPTH indicates the initial size of the parser's stacks       */

#ifndef YYINITDEPTH
#define YYINITDEPTH 200
#endif

/*  YYMAXDEPTH is the maximum size the stacks can grow to
    (effective only if the built-in stack extension method is used).  */

#if YYMAXDEPTH == 0
#undef YYMAXDEPTH
#endif

#ifndef YYMAXDEPTH
#define YYMAXDEPTH 10000
#endif


#if __GNUC__ > 1                /* GNU C and GNU C++ define this.  */
#define __yy_bcopy(FROM,TO,COUNT)       __builtin_memcpy(TO,FROM,COUNT)
#else                           /* not GNU C or C++ */

/* This is the most reliable way to avoid incompatibilities
   in available built-in functions on various systems.  */

#ifdef __cplusplus
static void __yy_bcopy (char *from, char *to, int count)
#else
#ifdef __STDC__
static void __yy_bcopy (char *from, char *to, int count)
#else
static void __yy_bcopy (from, to, count)
     char *from;
     char *to;
     int count;
#endif
#endif
{
   char *f = from;
   char *t = to;
   int i = count;

  while (i-- > 0)
    *t++ = *f++;
}
#endif

int
#ifdef YY_USE_CLASS
 YY_patBisonParam_CLASS::
#endif
     YY_patBisonParam_PARSE(YY_patBisonParam_PARSE_PARAM)
#ifndef __STDC__
#ifndef __cplusplus
#ifndef YY_USE_CLASS
/* parameter definition without protypes */
YY_patBisonParam_PARSE_PARAM_DEF
#endif
#endif
#endif
{
   int yystate;
   int yyn;
   short *yyssp;
   YY_patBisonParam_STYPE *yyvsp;
  int yyerrstatus;      /*  number of tokens to shift before error messages enabled */
  int yychar1=0;          /*  lookahead token as an internal (translated) token number */

  short yyssa[YYINITDEPTH];     /*  the state stack                     */
  YY_patBisonParam_STYPE yyvsa[YYINITDEPTH];        /*  the semantic value stack            */

  short *yyss = yyssa;          /*  refer to the stacks thru separate pointers */
  YY_patBisonParam_STYPE *yyvs = yyvsa;     /*  to allow yyoverflow to reallocate them elsewhere */

#ifdef YY_patBisonParam_LSP_NEEDED
  YY_patBisonParam_LTYPE yylsa[YYINITDEPTH];        /*  the location stack                  */
  YY_patBisonParam_LTYPE *yyls = yylsa;
  YY_patBisonParam_LTYPE *yylsp;

#define YYPOPSTACK   (yyvsp--, yyssp--, yylsp--)
#else
#define YYPOPSTACK   (yyvsp--, yyssp--)
#endif

  int yystacksize = YYINITDEPTH;

#ifdef YY_patBisonParam_PURE
  int YY_patBisonParam_CHAR;
  YY_patBisonParam_STYPE YY_patBisonParam_LVAL;
  int YY_patBisonParam_NERRS;
#ifdef YY_patBisonParam_LSP_NEEDED
  YY_patBisonParam_LTYPE YY_patBisonParam_LLOC;
#endif
#endif

  YY_patBisonParam_STYPE yyval;             /*  the variable used to return         */
				/*  semantic values from the action     */
				/*  routines                            */

  int yylen;
/* start loop, in which YYGOTO may be used. */
YYBEGINGOTO

#if YY_patBisonParam_DEBUG != 0
  if (YY_patBisonParam_DEBUG_FLAG)
    fprintf(stderr, "Starting parse\n");
#endif
  yystate = 0;
  yyerrstatus = 0;
  YY_patBisonParam_NERRS = 0;
  YY_patBisonParam_CHAR = YYEMPTY;          /* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss - 1;
  yyvsp = yyvs;
#ifdef YY_patBisonParam_LSP_NEEDED
  yylsp = yyls;
#endif

/* Push a new state, which is found in  yystate  .  */
/* In all cases, when you get here, the value and location stacks
   have just been pushed. so pushing a state here evens the stacks.  */
YYLABEL(yynewstate)

  *++yyssp = yystate;

  if (yyssp >= yyss + yystacksize - 1)
    {
      /* Give user a chance to reallocate the stack */
      /* Use copies of these so that the &'s don't force the real ones into memory. */
      YY_patBisonParam_STYPE *yyvs1 = yyvs;
      short *yyss1 = yyss;
#ifdef YY_patBisonParam_LSP_NEEDED
      YY_patBisonParam_LTYPE *yyls1 = yyls;
#endif

      /* Get the current used size of the three stacks, in elements.  */
      int size = yyssp - yyss + 1;

#ifdef yyoverflow
      /* Each stack pointer address is followed by the size of
	 the data in use in that stack, in bytes.  */
#ifdef YY_patBisonParam_LSP_NEEDED
      /* This used to be a conditional around just the two extra args,
	 but that might be undefined if yyoverflow is a macro.  */
      yyoverflow("parser stack overflow",
		 &yyss1, size * sizeof (*yyssp),
		 &yyvs1, size * sizeof (*yyvsp),
		 &yyls1, size * sizeof (*yylsp),
		 &yystacksize);
#else
      yyoverflow("parser stack overflow",
		 &yyss1, size * sizeof (*yyssp),
		 &yyvs1, size * sizeof (*yyvsp),
		 &yystacksize);
#endif

      yyss = yyss1; yyvs = yyvs1;
#ifdef YY_patBisonParam_LSP_NEEDED
      yyls = yyls1;
#endif
#else /* no yyoverflow */
      /* Extend the stack our own way.  */
      if (yystacksize >= YYMAXDEPTH)
	{
	  YY_patBisonParam_ERROR("parser stack overflow");
	  __ALLOCA_return(2);
	}
      yystacksize *= 2;
      if (yystacksize > YYMAXDEPTH)
	yystacksize = YYMAXDEPTH;
      yyss = (short *) __ALLOCA_alloca (yystacksize * sizeof (*yyssp));
      __yy_bcopy ((char *)yyss1, (char *)yyss, size * sizeof (*yyssp));
      __ALLOCA_free(yyss1,yyssa);
      yyvs = (YY_patBisonParam_STYPE *) __ALLOCA_alloca (yystacksize * sizeof (*yyvsp));
      __yy_bcopy ((char *)yyvs1, (char *)yyvs, size * sizeof (*yyvsp));
      __ALLOCA_free(yyvs1,yyvsa);
#ifdef YY_patBisonParam_LSP_NEEDED
      yyls = (YY_patBisonParam_LTYPE *) __ALLOCA_alloca (yystacksize * sizeof (*yylsp));
      __yy_bcopy ((char *)yyls1, (char *)yyls, size * sizeof (*yylsp));
      __ALLOCA_free(yyls1,yylsa);
#endif
#endif /* no yyoverflow */

      yyssp = yyss + size - 1;
      yyvsp = yyvs + size - 1;
#ifdef YY_patBisonParam_LSP_NEEDED
      yylsp = yyls + size - 1;
#endif

#if YY_patBisonParam_DEBUG != 0
      if (YY_patBisonParam_DEBUG_FLAG)
	fprintf(stderr, "Stack size increased to %d\n", yystacksize);
#endif

      if (yyssp >= yyss + yystacksize - 1)
	YYABORT;
    }

#if YY_patBisonParam_DEBUG != 0
  if (YY_patBisonParam_DEBUG_FLAG)
    fprintf(stderr, "Entering state %d\n", yystate);
#endif

  YYGOTO(yybackup);
YYLABEL(yybackup)

/* Do appropriate processing given the current state.  */
/* Read a lookahead token if we need one and don't already have one.  */
/* YYLABEL(yyresume) */

  /* First try to decide what to do without reference to lookahead token.  */

  yyn = yypact[yystate];
  if (yyn == YYFLAG)
    YYGOTO(yydefault);

  /* Not known => get a lookahead token if don't already have one.  */

  /* yychar is either YYEMPTY or YYEOF
     or a valid token in external form.  */

  if (YY_patBisonParam_CHAR == YYEMPTY)
    {
#if YY_patBisonParam_DEBUG != 0
      if (YY_patBisonParam_DEBUG_FLAG)
	fprintf(stderr, "Reading a token: ");
#endif
      YY_patBisonParam_CHAR = YYLEX;
    }

  /* Convert token to internal form (in yychar1) for indexing tables with */

  if (YY_patBisonParam_CHAR <= 0)           /* This means end of input. */
    {
      yychar1 = 0;
      YY_patBisonParam_CHAR = YYEOF;                /* Don't call YYLEX any more */

#if YY_patBisonParam_DEBUG != 0
      if (YY_patBisonParam_DEBUG_FLAG)
	fprintf(stderr, "Now at end of input.\n");
#endif
    }
  else
    {
      yychar1 = YYTRANSLATE(YY_patBisonParam_CHAR);

#if YY_patBisonParam_DEBUG != 0
      if (YY_patBisonParam_DEBUG_FLAG)
	{
	  fprintf (stderr, "Next token is %d (%s", YY_patBisonParam_CHAR, yytname[yychar1]);
	  /* Give the individual parser a way to print the precise meaning
	     of a token, for further debugging info.  */
#ifdef YYPRINT
	  YYPRINT (stderr, YY_patBisonParam_CHAR, YY_patBisonParam_LVAL);
#endif
	  fprintf (stderr, ")\n");
	}
#endif
    }

  yyn += yychar1;
  if (yyn < 0 || yyn > YYLAST || yycheck[yyn] != yychar1)
    YYGOTO(yydefault);

  yyn = yytable[yyn];

  /* yyn is what to do for this token type in this state.
     Negative => reduce, -yyn is rule number.
     Positive => shift, yyn is new state.
       New state is final state => don't bother to shift,
       just return success.
     0, or most negative number => error.  */

  if (yyn < 0)
    {
      if (yyn == YYFLAG)
	YYGOTO(yyerrlab);
      yyn = -yyn;
      YYGOTO(yyreduce);
    }
  else if (yyn == 0)
    YYGOTO(yyerrlab);

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Shift the lookahead token.  */

#if YY_patBisonParam_DEBUG != 0
  if (YY_patBisonParam_DEBUG_FLAG)
    fprintf(stderr, "Shifting token %d (%s), ", YY_patBisonParam_CHAR, yytname[yychar1]);
#endif

  /* Discard the token being shifted unless it is eof.  */
  if (YY_patBisonParam_CHAR != YYEOF)
    YY_patBisonParam_CHAR = YYEMPTY;

  *++yyvsp = YY_patBisonParam_LVAL;
#ifdef YY_patBisonParam_LSP_NEEDED
  *++yylsp = YY_patBisonParam_LLOC;
#endif

  /* count tokens shifted since error; after three, turn off error status.  */
  if (yyerrstatus) yyerrstatus--;

  yystate = yyn;
  YYGOTO(yynewstate);

/* Do the default action for the current state.  */
YYLABEL(yydefault)

  yyn = yydefact[yystate];
  if (yyn == 0)
    YYGOTO(yyerrlab);

/* Do a reduction.  yyn is the number of a rule to reduce with.  */
YYLABEL(yyreduce)
  yylen = yyr2[yyn];
  if (yylen > 0)
    yyval = yyvsp[1-yylen]; /* implement default value of the action */

#if YY_patBisonParam_DEBUG != 0
  if (YY_patBisonParam_DEBUG_FLAG)
    {
      int i;

      fprintf (stderr, "Reducing via rule %d (line %d), ",
	       yyn, yyrline[yyn]);

      /* Print the symbols being reduced, and their result.  */
      for (i = yyprhs[yyn]; yyrhs[i] > 0; i++)
	fprintf (stderr, "%s ", yytname[yyrhs[i]]);
      fprintf (stderr, " -> %s\n", yytname[yyr1[yyn]]);
    }
#endif


/* #line 811 "/usr/local/lib/bison.cc" */
#line 2596 "patParserParam.yy.tab.c"

  switch (yyn) {

case 1:
#line 367 "patParserParam.yy"
{
             ;
    break;}
case 22:
#line 394 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRMaxGcpIter(yyvsp[0].itype) ;
;
    break;}
case 23:
#line 399 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRArmijoBeta1(yyvsp[0].ftype) ;
;
    break;}
case 24:
#line 404 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRArmijoBeta2(yyvsp[0].ftype) ;
;
    break;}
case 25:
#line 409 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRStartDraws(yyvsp[0].itype) ;
;
    break;}
case 26:
#line 414 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRIncreaseDraws(yyvsp[0].ftype) ;
;
    break;}
case 27:
#line 419 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTREta1(yyvsp[0].ftype) ;
;
    break;}
case 28:
#line 424 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTREta2(yyvsp[0].ftype) ;
;
    break;}
case 29:
#line 429 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRGamma1(yyvsp[0].ftype) ;
;
    break;}
case 30:
#line 434 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRGamma2(yyvsp[0].ftype) ;
;
    break;}
case 31:
#line 439 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRInitRadius(yyvsp[0].ftype) ;
;
    break;}
case 32:
#line 444 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRIncreaseTRRadius(yyvsp[0].ftype) ;
;
    break;}
case 33:
#line 449 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRUnfeasibleCGIterations(yyvsp[0].itype) ;
;
    break;}
case 34:
#line 454 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRForceExactHessianIfMnl(yyvsp[0].itype) ;
;
    break;}
case 35:
#line 459 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRExactHessian(yyvsp[0].itype) ;
;
    break;}
case 36:
#line 464 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  DEBUG_MESSAGE("Set cheapHessian to " << yyvsp[0].itype) ;
  pParameters->setBTRCheapHessian(yyvsp[0].itype) ;
;
    break;}
case 37:
#line 470 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRQuasiNewtonUpdate(yyvsp[0].itype) ;
;
    break;}
case 38:
#line 475 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRInitQuasiNewtonWithTrueHessian(yyvsp[0].itype) ;
;
    break;}
case 39:
#line 480 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRInitQuasiNewtonWithBHHH(yyvsp[0].itype) ;
;
    break;}
case 40:
#line 485 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRMaxIter(yyvsp[0].itype) ;
;
    break;}
case 41:
#line 490 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRTypf(yyvsp[0].ftype) ;
;
    break;}
case 42:
#line 495 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRTolerance(yyvsp[0].ftype) ;
;
    break;}
case 43:
#line 500 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRMaxTRRadius(yyvsp[0].ftype) ;
;
    break;}
case 44:
#line 505 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRMinTRRadius(yyvsp[0].ftype) ;
;
    break;}
case 45:
#line 510 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRUsePreconditioner(yyvsp[0].itype) ;
;
    break;}
case 46:
#line 515 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRSingularityThreshold(yyvsp[0].ftype) ;
;
    break;}
case 47:
#line 520 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRKappaEpp(yyvsp[0].ftype) ;
;
    break;}
case 48:
#line 525 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRKappaLbs(yyvsp[0].ftype) ;
;
    break;}
case 49:
#line 530 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRKappaUbs(yyvsp[0].ftype) ;
;
    break;}
case 50:
#line 535 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRKappaFrd(yyvsp[0].ftype) ;
;
    break;}
case 51:
#line 540 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setBTRSignificantDigits(yyvsp[0].itype) ;
;
    break;}
case 55:
#line 549 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAETA0(yyvsp[0].ftype) ;
;
    break;}
case 56:
#line 554 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAETA1(yyvsp[0].ftype) ;
;
    break;}
case 57:
#line 559 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAETA2(yyvsp[0].ftype) ;
;
    break;}
case 58:
#line 564 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAGAMMA1(yyvsp[0].ftype) ;
;
    break;}
case 59:
#line 569 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAGAMMA2(yyvsp[0].ftype) ;
;
    break;}
case 60:
#line 574 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAEPSILONC(yyvsp[0].ftype) ;
;
    break;}
case 61:
#line 579 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAALPHA(yyvsp[0].ftype) ;
;
    break;}
case 62:
#line 584 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAMU(yyvsp[0].itype) ;
;
    break;}
case 63:
#line 589 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAMAXNBRFUNCTEVAL(yyvsp[0].itype) ;
;
    break;}
case 64:
#line 594 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAMAXLENGTH(yyvsp[0].itype) ;
;
    break;}
case 65:
#line 599 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAMAXDATA(yyvsp[0].itype) ;
;
    break;}
case 66:
#line 604 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRANBROFBESTPTS(yyvsp[0].itype) ;
;
    break;}
case 67:
#line 609 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAPOWER(yyvsp[0].ftype) ;
;
    break;}
case 68:
#line 614 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAMAXRAD(yyvsp[0].itype) ;
;
    break;}
case 69:
#line 619 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAMINRAD(yyvsp[0].ftype) ;
;
    break;}
case 70:
#line 624 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAUPPERBOUND(yyvsp[0].ftype) ;
;
    break;}
case 71:
#line 629 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRALOWERBOUND(yyvsp[0].ftype) ;
;
    break;}
case 72:
#line 634 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAGAMMA3(yyvsp[0].ftype) ;
;
    break;}
case 73:
#line 639 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAGAMMA4(yyvsp[0].ftype) ;
;
    break;}
case 74:
#line 644 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRACOEFVALID(yyvsp[0].ftype) ;
;
    break;}
case 75:
#line 649 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRACOEFGEN(yyvsp[0].ftype) ;
;
    break;}
case 76:
#line 654 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAEPSERROR(yyvsp[0].ftype) ;
;
    break;}
case 77:
#line 659 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAEPSPOINT(yyvsp[0].ftype) ;
;
    break;}
case 78:
#line 664 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRACOEFNORM(yyvsp[0].ftype) ;
;
    break;}
case 79:
#line 669 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAMINSTEP(yyvsp[0].ftype) ;
;
    break;}
case 80:
#line 674 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAMINPIVOTVALUE(yyvsp[0].ftype) ;
;
    break;}
case 81:
#line 679 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAGOODPIVOTVALUE(yyvsp[0].ftype) ;
;
    break;}
case 82:
#line 684 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAFINEPS(yyvsp[0].ftype) ;
;
    break;}
case 83:
#line 689 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAFINEPSREL(yyvsp[0].ftype) ;
;
    break;}
case 84:
#line 694 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRACHECKEPS(yyvsp[0].ftype) ;
;
    break;}
case 85:
#line 699 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRACHECKTESTEPS(yyvsp[0].ftype) ;
;
    break;}
case 86:
#line 704 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRACHECKTESTEPSREL(yyvsp[0].ftype) ;
;
    break;}
case 87:
#line 709 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAVALMINGAUSS(yyvsp[0].ftype) ;
;
    break;}
case 88:
#line 714 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCTRAFACTOFPOND(yyvsp[0].ftype) ;
;
    break;}
case 92:
#line 723 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setPrecond(yyvsp[0].itype) ;
;
    break;}
case 93:
#line 728 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setEpsilon(yyvsp[0].ftype) ;
;
    break;}
case 94:
#line 733 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCondLimit(yyvsp[0].ftype) ;
;
    break;}
case 95:
#line 738 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setPrecResidu(yyvsp[0].ftype) ;
;
    break;}
case 96:
#line 743 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setMaxCGIter(yyvsp[0].itype) ;
;
    break;}
case 97:
#line 748 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setTolSchnabelEskow(yyvsp[0].ftype) ;
;
    break;}
case 101:
#line 757 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setMaxIter(yyvsp[0].itype) ;
;
    break;}
case 102:
#line 762 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setInitStep(yyvsp[0].ftype) ;
;
    break;}
case 103:
#line 767 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setMinStep(yyvsp[0].ftype) ;
;
    break;}
case 104:
#line 772 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setMaxEval(yyvsp[0].itype) ;
;
    break;}
case 105:
#line 777 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setNbrRun(yyvsp[0].itype) ;
;
    break;}
case 106:
#line 782 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setMaxStep(yyvsp[0].ftype) ;
;
    break;}
case 107:
#line 787 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setAlphaProba(yyvsp[0].ftype) ;
;
    break;}
case 108:
#line 792 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setStepReduc(yyvsp[0].ftype) ;
;
    break;}
case 109:
#line 797 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setStepIncr(yyvsp[0].ftype) ;
;
    break;}
case 110:
#line 802 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setExpectedImprovement(yyvsp[0].ftype) ;
;
    break;}
case 111:
#line 807 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setAllowPremUnsucc(yyvsp[0].itype) ;
;
    break;}
case 112:
#line 812 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setPrematureStart(yyvsp[0].ftype) ;
;
    break;}
case 113:
#line 817 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setPrematureStep(yyvsp[0].ftype) ;
;
    break;}
case 114:
#line 822 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setMaxUnsuccIter(yyvsp[0].itype) ;
;
    break;}
case 115:
#line 827 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setNormWeight(yyvsp[0].ftype) ;
;
    break;}
case 119:
#line 836 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setInputDirectory(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 120:
#line 842 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setOutputDirectory(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 121:
#line 848 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setTmpDirectory(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 122:
#line 854 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setFunctionEvalExec(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 123:
#line 860 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setjonSimulator(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 124:
#line 866 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setCandidateFile(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 125:
#line 872 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setResultFile(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 126:
#line 878 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setOutsifFile(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 127:
#line 884 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setLogFile(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 128:
#line 890 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setProblemsFile(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 129:
#line 896 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setMITSIMorigin(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 130:
#line 902 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setMITSIMinformation(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 131:
#line 908 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setMITSIMtravelTime(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 132:
#line 914 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setMITSIMexec(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 136:
#line 924 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setAugmentationStep(yyvsp[0].ftype) ;
;
    break;}
case 137:
#line 929 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setReductionStep(yyvsp[0].ftype) ;
;
    break;}
case 138:
#line 934 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setSubSpaceMaxIter(yyvsp[0].itype) ;
;
    break;}
case 139:
#line 939 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setSubSpaceConsecutiveFailure(yyvsp[0].itype) ;
;
    break;}
case 140:
#line 944 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setWarmUpnbre(yyvsp[0].itype) ;
;
    break;}
case 144:
#line 953 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevInputDirectory(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 145:
#line 959 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevOutputDirectory(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 146:
#line 965 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevWorkingDirectory(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 147:
#line 971 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevSignificantDigitsParameters(yyvsp[0].itype) ;
;
    break;}
case 148:
#line 976 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevDecimalDigitsTTest(yyvsp[0].itype) ;
;
    break;}
case 149:
#line 981 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevDecimalDigitsStats(yyvsp[0].itype) ;
;
    break;}
case 150:
#line 986 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevForceScientificNotation(yyvsp[0].itype) ;
;
    break;}
case 151:
#line 991 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevSingularValueThreshold(yyvsp[0].ftype) ;
;
    break;}
case 152:
#line 996 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevPrintVarCovarAsList(yyvsp[0].itype) ;
;
    break;}
case 153:
#line 1001 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevPrintVarCovarAsMatrix(yyvsp[0].itype) ;
;
    break;}
case 154:
#line 1006 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevPrintPValue(yyvsp[0].itype) ;
;
    break;}
case 155:
#line 1011 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevNumberOfThreads(yyvsp[0].itype) ;
;
    break;}
case 156:
#line 1016 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevSaveIntermediateResults(yyvsp[0].itype) ;
;
    break;}
case 157:
#line 1021 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevVarCovarFromBHHH(yyvsp[0].itype) ;
;
    break;}
case 158:
#line 1026 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevDebugDataFirstRow(yyvsp[0].itype) ;
;
    break;}
case 159:
#line 1031 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevDebugDataLastRow(yyvsp[0].itype) ;
;
    break;}
case 160:
#line 1036 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevStoreDataOnFile(yyvsp[0].itype) ;
;
    break;}
case 161:
#line 1041 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevBinaryDataFile(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 162:
#line 1047 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevDumpDrawsOnFile(yyvsp[0].itype) ;
;
    break;}
case 163:
#line 1052 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevReadDrawsFromFile(yyvsp[0].itype) ;
;
    break;}
case 164:
#line 1057 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevGenerateActualSample(yyvsp[0].itype) ;
;
    break;}
case 165:
#line 1062 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevOutputActualSample(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 166:
#line 1068 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevNormalDrawsFile(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 167:
#line 1074 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevRectangularDrawsFile(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 168:
#line 1080 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevRandomDistrib(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 169:
#line 1086 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevMaxPrimeNumber(yyvsp[0].itype) ;
;
    break;}
case 170:
#line 1091 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevWarningSign(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 171:
#line 1097 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevWarningLowDraws(yyvsp[0].itype) ;
;
    break;}
case 172:
#line 1102 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevMissingValue(yyvsp[0].ftype) ;
;
    break;}
case 173:
#line 1107 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevGenerateFilesForDenis(yyvsp[0].itype) ;
;
    break;}
case 174:
#line 1112 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevGenerateGnuplotFile(yyvsp[0].itype) ;
;
    break;}
case 175:
#line 1117 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevGeneratePythonFile(yyvsp[0].itype) ;
;
    break;}
case 176:
#line 1122 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevPythonFileWithEstimatedParam(yyvsp[0].itype) ;
;
    break;}
case 177:
#line 1127 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevFileForDenis(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 178:
#line 1133 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevAutomaticScalingOfLinearUtility(yyvsp[0].itype) ;
;
    break;}
case 179:
#line 1138 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevInverseIteration(yyvsp[0].itype) ;
;
    break;}
case 180:
#line 1143 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevSeed(yyvsp[0].itype) ;
;
    break;}
case 181:
#line 1148 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevOne(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 182:
#line 1154 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevMinimumMu(yyvsp[0].ftype) ;
;
    break;}
case 183:
#line 1159 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevSummaryParameters(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 184:
#line 1165 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevSummaryFile(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 185:
#line 1171 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevStopFileName(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 186:
#line 1177 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevCheckDerivatives(yyvsp[0].itype) ;
;
    break;}
case 187:
#line 1182 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevBufferSize(yyvsp[0].itype) ;
;
    break;}
case 188:
#line 1187 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevDataFileDisplayStep(yyvsp[0].itype) ;
;
    break;}
case 189:
#line 1192 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevTtestThreshold(yyvsp[0].ftype) ;
;
    break;}
case 190:
#line 1197 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevGlobal(yyvsp[0].itype) ;
;
    break;}
case 191:
#line 1202 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevAnalGrad(yyvsp[0].itype) ;
;
    break;}
case 192:
#line 1207 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevAnalHess(yyvsp[0].itype) ;
;
    break;}
case 193:
#line 1212 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevCheapF(yyvsp[0].itype) ;
;
    break;}
case 194:
#line 1217 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevFactSec(yyvsp[0].itype) ;
;
    break;}
case 195:
#line 1222 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevTermCode(yyvsp[0].itype) ;
;
    break;}
case 196:
#line 1227 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevTypx(yyvsp[0].itype) ;
;
    break;}
case 197:
#line 1232 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevTypF(yyvsp[0].ftype) ;
;
    break;}
case 198:
#line 1237 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevFDigits(yyvsp[0].itype) ;
;
    break;}
case 199:
#line 1242 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevGradTol(yyvsp[0].ftype) ;
;
    break;}
case 200:
#line 1247 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevMaxStep(yyvsp[0].ftype) ;
;
    break;}
case 201:
#line 1252 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevItnLimit(yyvsp[0].itype) ;
;
    break;}
case 202:
#line 1257 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevDelta(yyvsp[0].ftype) ;
;
    break;}
case 203:
#line 1262 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevAlgo(*yyvsp[0].stype) ;
  DELETE_PTR(yyvsp[0].stype) ;
;
    break;}
case 204:
#line 1268 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevScreenPrintLevel(yyvsp[0].itype) ;
;
    break;}
case 205:
#line 1273 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevLogFilePrintLevel(yyvsp[0].itype) ;
;
    break;}
case 206:
#line 1278 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevGeneratedGroups(yyvsp[0].itype) ;
;
    break;}
case 207:
#line 1283 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevGeneratedData(yyvsp[0].itype) ;
;
    break;}
case 208:
#line 1288 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevGeneratedAttr(yyvsp[0].itype) ;
;
    break;}
case 209:
#line 1293 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevGeneratedAlt(yyvsp[0].itype) ;
;
    break;}
case 210:
#line 1298 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevSubSampleLevel(yyvsp[0].itype) ;
;
    break;}
case 211:
#line 1303 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevSubSampleBasis(yyvsp[0].itype) ;
;
    break;}
case 212:
#line 1308 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevComputeLastHessian(yyvsp[0].itype) ;
;
    break;}
case 213:
#line 1313 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevEigenvalueThreshold(yyvsp[0].ftype) ;
;
    break;}
case 214:
#line 1318 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevNonParamPlotRes(yyvsp[0].itype) ;
;
    break;}
case 215:
#line 1323 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevNonParamPlotMaxY(yyvsp[0].ftype) ;
;
    break;}
case 216:
#line 1328 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevNonParamPlotXSizeCm(yyvsp[0].itype) ;
;
    break;}
case 217:
#line 1333 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevNonParamPlotYSizeCm(yyvsp[0].itype) ;
;
    break;}
case 218:
#line 1338 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevNonParamPlotMinXSizeCm(yyvsp[0].ftype) ;
;
    break;}
case 219:
#line 1343 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setgevNonParamPlotMinYSizeCm(yyvsp[0].ftype) ;
;
    break;}
case 220:
#line 1348 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setsvdMaxIter(yyvsp[0].itype) ;
;
    break;}
case 224:
#line 1357 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->sethieMultinomial(yyvsp[0].itype) ;
;
    break;}
case 225:
#line 1362 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->sethieTruncStructUtil(yyvsp[0].itype) ;
;
    break;}
case 226:
#line 1367 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->sethieUpdateHessien(yyvsp[0].itype) ;
;
    break;}
case 227:
#line 1372 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->sethieDateInLog(yyvsp[0].itype) ;
;
    break;}
case 231:
#line 1381 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setbolducMaxAlts(yyvsp[0].itype) ;
;
    break;}
case 232:
#line 1386 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setbolducMaxFact(yyvsp[0].itype) ;
;
    break;}
case 233:
#line 1391 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setbolducMaxNVar(yyvsp[0].itype) ;
;
    break;}
case 237:
#line 1400 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setStepSecondIndividual(yyvsp[0].ftype) ;
;
    break;}
case 238:
#line 1405 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setNLgWeight(yyvsp[0].ftype) ;
;
    break;}
case 239:
#line 1410 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setNLhWeight(yyvsp[0].ftype) ;
;
    break;}
case 243:
#line 1419 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setTSFractionGradientRequired(yyvsp[0].ftype) ;
;
    break;}
case 244:
#line 1424 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setTSExpTheta(yyvsp[0].ftype) ;
;
    break;}
case 248:
#line 1433 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setcfsqpMode(yyvsp[0].itype) ;
;
    break;}
case 249:
#line 1438 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setcfsqpIprint(yyvsp[0].itype) ;
;
    break;}
case 250:
#line 1443 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setcfsqpMaxIter(yyvsp[0].itype) ;
;
    break;}
case 251:
#line 1448 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setcfsqpEps(yyvsp[0].ftype) ;
;
    break;}
case 252:
#line 1453 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setcfsqpEpsEqn(yyvsp[0].ftype) ;
;
    break;}
case 253:
#line 1458 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setcfsqpUdelta(yyvsp[0].ftype) ;
;
    break;}
case 257:
#line 1467 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setdfoAddToLWRK(yyvsp[0].itype) ;
;
    break;}
case 258:
#line 1472 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setdfoAddToLIWRK(yyvsp[0].itype) ;
;
    break;}
case 259:
#line 1477 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setdfoMaxFunEval(yyvsp[0].itype) ;
;
    break;}
case 263:
#line 1486 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setdonlp2Epsx(yyvsp[0].ftype) ;
;
    break;}
case 264:
#line 1491 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setdonlp2Delmin(yyvsp[0].ftype) ;
;
    break;}
case 265:
#line 1496 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setdonlp2Smallw(yyvsp[0].ftype) ;
;
    break;}
case 266:
#line 1501 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setdonlp2Epsdif(yyvsp[0].ftype) ;
;
    break;}
case 267:
#line 1506 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setdonlp2NReset(yyvsp[0].itype) ;
;
    break;}
case 271:
#line 1515 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setsolvoptMaxIter(yyvsp[0].itype) ;
;
    break;}
case 272:
#line 1520 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setsolvoptDisplay(yyvsp[0].itype) ;
;
    break;}
case 273:
#line 1525 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setsolvoptErrorArgument(yyvsp[0].ftype) ;
;
    break;}
case 274:
#line 1530 "patParserParam.yy"
{
  assert (pParameters != NULL) ;
  pParameters->setsolvoptErrorFunction(yyvsp[0].ftype) ;
;
    break;}
case 275:
#line 1536 "patParserParam.yy"
{
  string* str = new string((scanner.removeDelimeters()));
  yyval.stype = str ;
;
    break;}
case 276:
#line 1541 "patParserParam.yy"
{
  yyval.ftype = atof( scanner.value().c_str() );
;
    break;}
case 277:
#line 1546 "patParserParam.yy"
{
  yyval.itype = atoi( scanner.value().c_str() );
;
    break;}
}

#line 811 "/usr/local/lib/bison.cc"
   /* the action file gets copied in in place of this dollarsign  */
  yyvsp -= yylen;
  yyssp -= yylen;
#ifdef YY_patBisonParam_LSP_NEEDED
  yylsp -= yylen;
#endif

#if YY_patBisonParam_DEBUG != 0
  if (YY_patBisonParam_DEBUG_FLAG)
    {
      short *ssp1 = yyss - 1;
      fprintf (stderr, "state stack now");
      while (ssp1 != yyssp)
	fprintf (stderr, " %d", *++ssp1);
      fprintf (stderr, "\n");
    }
#endif

  *++yyvsp = yyval;

#ifdef YY_patBisonParam_LSP_NEEDED
  yylsp++;
  if (yylen == 0)
    {
      yylsp->first_line = YY_patBisonParam_LLOC.first_line;
      yylsp->first_column = YY_patBisonParam_LLOC.first_column;
      yylsp->last_line = (yylsp-1)->last_line;
      yylsp->last_column = (yylsp-1)->last_column;
      yylsp->text = 0;
    }
  else
    {
      yylsp->last_line = (yylsp+yylen-1)->last_line;
      yylsp->last_column = (yylsp+yylen-1)->last_column;
    }
#endif

  /* Now "shift" the result of the reduction.
     Determine what state that goes to,
     based on the state we popped back to
     and the rule number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTBASE] + *yyssp;
  if (yystate >= 0 && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTBASE];

  YYGOTO(yynewstate);

YYLABEL(yyerrlab)   /* here on detecting error */

  if (! yyerrstatus)
    /* If not already recovering from an error, report this error.  */
    {
      ++YY_patBisonParam_NERRS;

#ifdef YY_patBisonParam_ERROR_VERBOSE
      yyn = yypact[yystate];

      if (yyn > YYFLAG && yyn < YYLAST)
	{
	  int size = 0;
	  char *msg;
	  int x, count;

	  count = 0;
	  /* Start X at -yyn if nec to avoid negative indexes in yycheck.  */
	  for (x = (yyn < 0 ? -yyn : 0);
	       x < (sizeof(yytname) / sizeof(char *)); x++)
	    if (yycheck[x + yyn] == x)
	      size += strlen(yytname[x]) + 15, count++;
	  msg = (char *) malloc(size + 15);
	  if (msg != 0)
	    {
	      strcpy(msg, "parse error");

	      if (count < 5)
		{
		  count = 0;
		  for (x = (yyn < 0 ? -yyn : 0);
		       x < (sizeof(yytname) / sizeof(char *)); x++)
		    if (yycheck[x + yyn] == x)
		      {
			strcat(msg, count == 0 ? ", expecting `" : " or `");
			strcat(msg, yytname[x]);
			strcat(msg, "'");
			count++;
		      }
		}
	      YY_patBisonParam_ERROR(msg);
	      free(msg);
	    }
	  else
	    YY_patBisonParam_ERROR ("parse error; also virtual memory exceeded");
	}
      else
#endif /* YY_patBisonParam_ERROR_VERBOSE */
	YY_patBisonParam_ERROR("parse error");
    }

  YYGOTO(yyerrlab1);
YYLABEL(yyerrlab1)   /* here on error raised explicitly by an action */

  if (yyerrstatus == 3)
    {
      /* if just tried and failed to reuse lookahead token after an error, discard it.  */

      /* return failure if at end of input */
      if (YY_patBisonParam_CHAR == YYEOF)
	YYABORT;

#if YY_patBisonParam_DEBUG != 0
      if (YY_patBisonParam_DEBUG_FLAG)
	fprintf(stderr, "Discarding token %d (%s).\n", YY_patBisonParam_CHAR, yytname[yychar1]);
#endif

      YY_patBisonParam_CHAR = YYEMPTY;
    }

  /* Else will try to reuse lookahead token
     after shifting the error token.  */

  yyerrstatus = 3;              /* Each real token shifted decrements this */

  YYGOTO(yyerrhandle);

YYLABEL(yyerrdefault)  /* current state does not do anything special for the error token. */

#if 0
  /* This is wrong; only states that explicitly want error tokens
     should shift them.  */
  yyn = yydefact[yystate];  /* If its default is to accept any token, ok.  Otherwise pop it.*/
  if (yyn) YYGOTO(yydefault);
#endif

YYLABEL(yyerrpop)   /* pop the current state because it cannot handle the error token */

  if (yyssp == yyss) YYABORT;
  yyvsp--;
  yystate = *--yyssp;
#ifdef YY_patBisonParam_LSP_NEEDED
  yylsp--;
#endif

#if YY_patBisonParam_DEBUG != 0
  if (YY_patBisonParam_DEBUG_FLAG)
    {
      short *ssp1 = yyss - 1;
      fprintf (stderr, "Error: state stack now");
      while (ssp1 != yyssp)
	fprintf (stderr, " %d", *++ssp1);
      fprintf (stderr, "\n");
    }
#endif

YYLABEL(yyerrhandle)

  yyn = yypact[yystate];
  if (yyn == YYFLAG)
    YYGOTO(yyerrdefault);

  yyn += YYTERROR;
  if (yyn < 0 || yyn > YYLAST || yycheck[yyn] != YYTERROR)
    YYGOTO(yyerrdefault);

  yyn = yytable[yyn];
  if (yyn < 0)
    {
      if (yyn == YYFLAG)
	YYGOTO(yyerrpop);
      yyn = -yyn;
      YYGOTO(yyreduce);
    }
  else if (yyn == 0)
    YYGOTO(yyerrpop);

  if (yyn == YYFINAL)
    YYACCEPT;

#if YY_patBisonParam_DEBUG != 0
  if (YY_patBisonParam_DEBUG_FLAG)
    fprintf(stderr, "Shifting error token, ");
#endif

  *++yyvsp = YY_patBisonParam_LVAL;
#ifdef YY_patBisonParam_LSP_NEEDED
  *++yylsp = YY_patBisonParam_LLOC;
#endif

  yystate = yyn;
  YYGOTO(yynewstate);
/* end loop, in which YYGOTO may be used. */
  YYENDGOTO
}

/* END */

/* #line 1010 "/usr/local/lib/bison.cc" */
#line 4335 "patParserParam.yy.tab.c"
#line 1554 "patParserParam.yy"



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

 

