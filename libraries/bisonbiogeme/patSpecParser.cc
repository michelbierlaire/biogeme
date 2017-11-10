#define YY_patBisonSpec_h_included

/*  A Bison++ parser, made from patSpecParser.yy  */

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
#line 85 "patSpecParser.yy.tab.c"
#define YY_patBisonSpec_ERROR_BODY  = 0
#define YY_patBisonSpec_LEX_BODY  = 0
#define YY_patBisonSpec_MEMBERS  patSpecScanner scanner; patModelSpec *pModel; virtual ~patBisonSpec() {};
#define YY_patBisonSpec_CONSTRUCTOR_PARAM  const patString& fname_
#define YY_patBisonSpec_CONSTRUCTOR_INIT  : scanner(fname_) , pModel(NULL)
#line 18 "patSpecParser.yy"

  
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




#line 152 "patSpecParser.yy"
typedef union {
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
} yy_patBisonSpec_stype;
#define YY_patBisonSpec_STYPE yy_patBisonSpec_stype

#line 73 "/usr/local/lib/bison.cc"
/* %{ and %header{ and %union, during decl */
#define YY_patBisonSpec_BISON 1
#ifndef YY_patBisonSpec_COMPATIBILITY
#ifndef YY_USE_CLASS
#define  YY_patBisonSpec_COMPATIBILITY 1
#else
#define  YY_patBisonSpec_COMPATIBILITY 0
#endif
#endif

#if YY_patBisonSpec_COMPATIBILITY != 0
/* backward compatibility */
#ifdef YYLTYPE
#ifndef YY_patBisonSpec_LTYPE
#define YY_patBisonSpec_LTYPE YYLTYPE
#endif
#endif
#ifdef YYSTYPE
#ifndef YY_patBisonSpec_STYPE 
#define YY_patBisonSpec_STYPE YYSTYPE
#endif
#endif
#ifdef YYDEBUG
#ifndef YY_patBisonSpec_DEBUG
#define  YY_patBisonSpec_DEBUG YYDEBUG
#endif
#endif
#ifdef YY_patBisonSpec_STYPE
#ifndef yystype
#define yystype YY_patBisonSpec_STYPE
#endif
#endif
/* use goto to be compatible */
#ifndef YY_patBisonSpec_USE_GOTO
#define YY_patBisonSpec_USE_GOTO 1
#endif
#endif

/* use no goto to be clean in C++ */
#ifndef YY_patBisonSpec_USE_GOTO
#define YY_patBisonSpec_USE_GOTO 0
#endif

#ifndef YY_patBisonSpec_PURE

/* #line 117 "/usr/local/lib/bison.cc" */
#line 293 "patSpecParser.yy.tab.c"

#line 117 "/usr/local/lib/bison.cc"
/*  YY_patBisonSpec_PURE */
#endif

/* section apres lecture def, avant lecture grammaire S2 */

/* #line 121 "/usr/local/lib/bison.cc" */
#line 302 "patSpecParser.yy.tab.c"

#line 121 "/usr/local/lib/bison.cc"
/* prefix */
#ifndef YY_patBisonSpec_DEBUG

/* #line 123 "/usr/local/lib/bison.cc" */
#line 309 "patSpecParser.yy.tab.c"

#line 123 "/usr/local/lib/bison.cc"
/* YY_patBisonSpec_DEBUG */
#endif


#ifndef YY_patBisonSpec_LSP_NEEDED

/* #line 128 "/usr/local/lib/bison.cc" */
#line 319 "patSpecParser.yy.tab.c"

#line 128 "/usr/local/lib/bison.cc"
 /* YY_patBisonSpec_LSP_NEEDED*/
#endif



/* DEFAULT LTYPE*/
#ifdef YY_patBisonSpec_LSP_NEEDED
#ifndef YY_patBisonSpec_LTYPE
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

#define YY_patBisonSpec_LTYPE yyltype
#endif
#endif
/* DEFAULT STYPE*/
      /* We used to use `unsigned long' as YY_patBisonSpec_STYPE on MSDOS,
	 but it seems better to be consistent.
	 Most programs should declare their own type anyway.  */

#ifndef YY_patBisonSpec_STYPE
#define YY_patBisonSpec_STYPE int
#endif
/* DEFAULT MISCELANEOUS */
#ifndef YY_patBisonSpec_PARSE
#define YY_patBisonSpec_PARSE yyparse
#endif
#ifndef YY_patBisonSpec_LEX
#define YY_patBisonSpec_LEX yylex
#endif
#ifndef YY_patBisonSpec_LVAL
#define YY_patBisonSpec_LVAL yylval
#endif
#ifndef YY_patBisonSpec_LLOC
#define YY_patBisonSpec_LLOC yylloc
#endif
#ifndef YY_patBisonSpec_CHAR
#define YY_patBisonSpec_CHAR yychar
#endif
#ifndef YY_patBisonSpec_NERRS
#define YY_patBisonSpec_NERRS yynerrs
#endif
#ifndef YY_patBisonSpec_DEBUG_FLAG
#define YY_patBisonSpec_DEBUG_FLAG yydebug
#endif
#ifndef YY_patBisonSpec_ERROR
#define YY_patBisonSpec_ERROR yyerror
#endif
#ifndef YY_patBisonSpec_PARSE_PARAM
#ifndef __STDC__
#ifndef __cplusplus
#ifndef YY_USE_CLASS
#define YY_patBisonSpec_PARSE_PARAM
#ifndef YY_patBisonSpec_PARSE_PARAM_DEF
#define YY_patBisonSpec_PARSE_PARAM_DEF
#endif
#endif
#endif
#endif
#ifndef YY_patBisonSpec_PARSE_PARAM
#define YY_patBisonSpec_PARSE_PARAM void
#endif
#endif
#if YY_patBisonSpec_COMPATIBILITY != 0
/* backward compatibility */
#ifdef YY_patBisonSpec_LTYPE
#ifndef YYLTYPE
#define YYLTYPE YY_patBisonSpec_LTYPE
#else
/* WARNING obsolete !!! user defined YYLTYPE not reported into generated header */
#endif
#endif
#ifndef YYSTYPE
#define YYSTYPE YY_patBisonSpec_STYPE
#else
/* WARNING obsolete !!! user defined YYSTYPE not reported into generated header */
#endif
#ifdef YY_patBisonSpec_PURE
#ifndef YYPURE
#define YYPURE YY_patBisonSpec_PURE
#endif
#endif
#ifdef YY_patBisonSpec_DEBUG
#ifndef YYDEBUG
#define YYDEBUG YY_patBisonSpec_DEBUG 
#endif
#endif
#ifndef YY_patBisonSpec_ERROR_VERBOSE
#ifdef YYERROR_VERBOSE
#define YY_patBisonSpec_ERROR_VERBOSE YYERROR_VERBOSE
#endif
#endif
#ifndef YY_patBisonSpec_LSP_NEEDED
#ifdef YYLSP_NEEDED
#define YY_patBisonSpec_LSP_NEEDED YYLSP_NEEDED
#endif
#endif
#endif
#ifndef YY_USE_CLASS
/* TOKEN C */

/* #line 236 "/usr/local/lib/bison.cc" */
#line 432 "patSpecParser.yy.tab.c"
#define	pat_gevDataFile	258
#define	pat_gevModelDescription	259
#define	pat_gevChoice	260
#define	pat_gevPanel	261
#define	pat_gevWeight	262
#define	pat_gevBeta	263
#define	pat_gevBoxCox	264
#define	pat_gevBoxTukey	265
#define	pat_gevLatex1	266
#define	pat_gevLatex2	267
#define	pat_gevMu	268
#define	pat_gevSampleEnum	269
#define	pat_gevGnuplot	270
#define	pat_gevUtilities	271
#define	pat_gevGeneralizedUtilities	272
#define	pat_gevDerivatives	273
#define	pat_gevParameterCovariances	274
#define	pat_gevExpr	275
#define	pat_gevGroup	276
#define	pat_gevExclude	277
#define	pat_gevScale	278
#define	pat_gevModel	279
#define	pat_gevNLNests	280
#define	pat_gevCNLAlpha	281
#define	pat_gevCNLNests	282
#define	pat_gevRatios	283
#define	pat_gevDraws	284
#define	pat_gevConstraintNestCoef	285
#define	pat_gevConstantProduct	286
#define	pat_gevNetworkGEVNodes	287
#define	pat_gevNetworkGEVLinks	288
#define	pat_gevLinearConstraints	289
#define	pat_gevNonLinearEqualityConstraints	290
#define	pat_gevNonLinearInequalityConstraints	291
#define	pat_gevLogitKernelSigmas	292
#define	pat_gevLogitKernelFactors	293
#define	pat_gevDiscreteDistributions	294
#define	pat_gevSelectionBias	295
#define	pat_gevSNP	296
#define	pat_gevAggregateLast	297
#define	pat_gevAggregateWeight	298
#define	pat_gevMassAtZero	299
#define	pat_gevOrdinalLogit	300
#define	pat_gevRegressionModels	301
#define	pat_gevDurationModel	302
#define	pat_gevZhengFosgerau	303
#define	pat_gevGeneralizedExtremeValue	304
#define	pat_gevIIATest	305
#define	pat_gevProbaStandardErrors	306
#define	pat_gevBP	307
#define	pat_gevOL	308
#define	pat_gevMNL	309
#define	pat_gevNL	310
#define	pat_gevCNL	311
#define	pat_gevNGEV	312
#define	pat_gevNONE	313
#define	pat_gevROOT	314
#define	pat_gevCOLUMNS	315
#define	pat_gevLOOP	316
#define	pat_gevDERIV	317
#define	pat_gevACQ	318
#define	pat_gevSIGMA_ACQ	319
#define	pat_gevLOG_ACQ	320
#define	pat_gevVAL	321
#define	pat_gevSIGMA_VAL	322
#define	pat_gevLOG_VAL	323
#define	pat_gevE	324
#define	pat_gevP	325
#define	patOR	326
#define	patAND	327
#define	patEQUAL	328
#define	patNOTEQUAL	329
#define	patLESS	330
#define	patLESSEQUAL	331
#define	patGREAT	332
#define	patGREATEQUAL	333
#define	patNOT	334
#define	patPLUS	335
#define	patMINUS	336
#define	patMULT	337
#define	patDIVIDE	338
#define	patMOD	339
#define	patPOWER	340
#define	patUNARYMINUS	341
#define	patMAX	342
#define	patMIN	343
#define	patSQRT	344
#define	patLOG	345
#define	patEXP	346
#define	patABS	347
#define	patINT	348
#define	patOPPAR	349
#define	patCLPAR	350
#define	patOPBRA	351
#define	patCLBRA	352
#define	patOPCUR	353
#define	patCLCUR	354
#define	patCOMMA	355
#define	patCOLON	356
#define	patINTEGER	357
#define	patREAL	358
#define	patTIME	359
#define	patNAME	360
#define	patSTRING	361
#define	patPAIR	362


#line 236 "/usr/local/lib/bison.cc"
 /* #defines tokens */
#else
/* CLASS */
#ifndef YY_patBisonSpec_CLASS
#define YY_patBisonSpec_CLASS patBisonSpec
#endif
#ifndef YY_patBisonSpec_INHERIT
#define YY_patBisonSpec_INHERIT
#endif
#ifndef YY_patBisonSpec_MEMBERS
#define YY_patBisonSpec_MEMBERS 
#endif
#ifndef YY_patBisonSpec_LEX_BODY
#define YY_patBisonSpec_LEX_BODY  
#endif
#ifndef YY_patBisonSpec_ERROR_BODY
#define YY_patBisonSpec_ERROR_BODY  
#endif
#ifndef YY_patBisonSpec_CONSTRUCTOR_PARAM
#define YY_patBisonSpec_CONSTRUCTOR_PARAM
#endif
#ifndef YY_patBisonSpec_CONSTRUCTOR_CODE
#define YY_patBisonSpec_CONSTRUCTOR_CODE
#endif
#ifndef YY_patBisonSpec_CONSTRUCTOR_INIT
#define YY_patBisonSpec_CONSTRUCTOR_INIT
#endif
/* choose between enum and const */
#ifndef YY_patBisonSpec_USE_CONST_TOKEN
#define YY_patBisonSpec_USE_CONST_TOKEN 0
/* yes enum is more compatible with flex,  */
/* so by default we use it */ 
#endif
#if YY_patBisonSpec_USE_CONST_TOKEN != 0
#ifndef YY_patBisonSpec_ENUM_TOKEN
#define YY_patBisonSpec_ENUM_TOKEN yy_patBisonSpec_enum_token
#endif
#endif

class YY_patBisonSpec_CLASS YY_patBisonSpec_INHERIT
{
public: 
#if YY_patBisonSpec_USE_CONST_TOKEN != 0
/* static const int token ... */

/* #line 280 "/usr/local/lib/bison.cc" */
#line 587 "patSpecParser.yy.tab.c"
static const int pat_gevDataFile;
static const int pat_gevModelDescription;
static const int pat_gevChoice;
static const int pat_gevPanel;
static const int pat_gevWeight;
static const int pat_gevBeta;
static const int pat_gevBoxCox;
static const int pat_gevBoxTukey;
static const int pat_gevLatex1;
static const int pat_gevLatex2;
static const int pat_gevMu;
static const int pat_gevSampleEnum;
static const int pat_gevGnuplot;
static const int pat_gevUtilities;
static const int pat_gevGeneralizedUtilities;
static const int pat_gevDerivatives;
static const int pat_gevParameterCovariances;
static const int pat_gevExpr;
static const int pat_gevGroup;
static const int pat_gevExclude;
static const int pat_gevScale;
static const int pat_gevModel;
static const int pat_gevNLNests;
static const int pat_gevCNLAlpha;
static const int pat_gevCNLNests;
static const int pat_gevRatios;
static const int pat_gevDraws;
static const int pat_gevConstraintNestCoef;
static const int pat_gevConstantProduct;
static const int pat_gevNetworkGEVNodes;
static const int pat_gevNetworkGEVLinks;
static const int pat_gevLinearConstraints;
static const int pat_gevNonLinearEqualityConstraints;
static const int pat_gevNonLinearInequalityConstraints;
static const int pat_gevLogitKernelSigmas;
static const int pat_gevLogitKernelFactors;
static const int pat_gevDiscreteDistributions;
static const int pat_gevSelectionBias;
static const int pat_gevSNP;
static const int pat_gevAggregateLast;
static const int pat_gevAggregateWeight;
static const int pat_gevMassAtZero;
static const int pat_gevOrdinalLogit;
static const int pat_gevRegressionModels;
static const int pat_gevDurationModel;
static const int pat_gevZhengFosgerau;
static const int pat_gevGeneralizedExtremeValue;
static const int pat_gevIIATest;
static const int pat_gevProbaStandardErrors;
static const int pat_gevBP;
static const int pat_gevOL;
static const int pat_gevMNL;
static const int pat_gevNL;
static const int pat_gevCNL;
static const int pat_gevNGEV;
static const int pat_gevNONE;
static const int pat_gevROOT;
static const int pat_gevCOLUMNS;
static const int pat_gevLOOP;
static const int pat_gevDERIV;
static const int pat_gevACQ;
static const int pat_gevSIGMA_ACQ;
static const int pat_gevLOG_ACQ;
static const int pat_gevVAL;
static const int pat_gevSIGMA_VAL;
static const int pat_gevLOG_VAL;
static const int pat_gevE;
static const int pat_gevP;
static const int patOR;
static const int patAND;
static const int patEQUAL;
static const int patNOTEQUAL;
static const int patLESS;
static const int patLESSEQUAL;
static const int patGREAT;
static const int patGREATEQUAL;
static const int patNOT;
static const int patPLUS;
static const int patMINUS;
static const int patMULT;
static const int patDIVIDE;
static const int patMOD;
static const int patPOWER;
static const int patUNARYMINUS;
static const int patMAX;
static const int patMIN;
static const int patSQRT;
static const int patLOG;
static const int patEXP;
static const int patABS;
static const int patINT;
static const int patOPPAR;
static const int patCLPAR;
static const int patOPBRA;
static const int patCLBRA;
static const int patOPCUR;
static const int patCLCUR;
static const int patCOMMA;
static const int patCOLON;
static const int patINTEGER;
static const int patREAL;
static const int patTIME;
static const int patNAME;
static const int patSTRING;
static const int patPAIR;


#line 280 "/usr/local/lib/bison.cc"
 /* decl const */
#else
enum YY_patBisonSpec_ENUM_TOKEN { YY_patBisonSpec_NULL_TOKEN=0

/* #line 283 "/usr/local/lib/bison.cc" */
#line 701 "patSpecParser.yy.tab.c"
	,pat_gevDataFile=258
	,pat_gevModelDescription=259
	,pat_gevChoice=260
	,pat_gevPanel=261
	,pat_gevWeight=262
	,pat_gevBeta=263
	,pat_gevBoxCox=264
	,pat_gevBoxTukey=265
	,pat_gevLatex1=266
	,pat_gevLatex2=267
	,pat_gevMu=268
	,pat_gevSampleEnum=269
	,pat_gevGnuplot=270
	,pat_gevUtilities=271
	,pat_gevGeneralizedUtilities=272
	,pat_gevDerivatives=273
	,pat_gevParameterCovariances=274
	,pat_gevExpr=275
	,pat_gevGroup=276
	,pat_gevExclude=277
	,pat_gevScale=278
	,pat_gevModel=279
	,pat_gevNLNests=280
	,pat_gevCNLAlpha=281
	,pat_gevCNLNests=282
	,pat_gevRatios=283
	,pat_gevDraws=284
	,pat_gevConstraintNestCoef=285
	,pat_gevConstantProduct=286
	,pat_gevNetworkGEVNodes=287
	,pat_gevNetworkGEVLinks=288
	,pat_gevLinearConstraints=289
	,pat_gevNonLinearEqualityConstraints=290
	,pat_gevNonLinearInequalityConstraints=291
	,pat_gevLogitKernelSigmas=292
	,pat_gevLogitKernelFactors=293
	,pat_gevDiscreteDistributions=294
	,pat_gevSelectionBias=295
	,pat_gevSNP=296
	,pat_gevAggregateLast=297
	,pat_gevAggregateWeight=298
	,pat_gevMassAtZero=299
	,pat_gevOrdinalLogit=300
	,pat_gevRegressionModels=301
	,pat_gevDurationModel=302
	,pat_gevZhengFosgerau=303
	,pat_gevGeneralizedExtremeValue=304
	,pat_gevIIATest=305
	,pat_gevProbaStandardErrors=306
	,pat_gevBP=307
	,pat_gevOL=308
	,pat_gevMNL=309
	,pat_gevNL=310
	,pat_gevCNL=311
	,pat_gevNGEV=312
	,pat_gevNONE=313
	,pat_gevROOT=314
	,pat_gevCOLUMNS=315
	,pat_gevLOOP=316
	,pat_gevDERIV=317
	,pat_gevACQ=318
	,pat_gevSIGMA_ACQ=319
	,pat_gevLOG_ACQ=320
	,pat_gevVAL=321
	,pat_gevSIGMA_VAL=322
	,pat_gevLOG_VAL=323
	,pat_gevE=324
	,pat_gevP=325
	,patOR=326
	,patAND=327
	,patEQUAL=328
	,patNOTEQUAL=329
	,patLESS=330
	,patLESSEQUAL=331
	,patGREAT=332
	,patGREATEQUAL=333
	,patNOT=334
	,patPLUS=335
	,patMINUS=336
	,patMULT=337
	,patDIVIDE=338
	,patMOD=339
	,patPOWER=340
	,patUNARYMINUS=341
	,patMAX=342
	,patMIN=343
	,patSQRT=344
	,patLOG=345
	,patEXP=346
	,patABS=347
	,patINT=348
	,patOPPAR=349
	,patCLPAR=350
	,patOPBRA=351
	,patCLBRA=352
	,patOPCUR=353
	,patCLCUR=354
	,patCOMMA=355
	,patCOLON=356
	,patINTEGER=357
	,patREAL=358
	,patTIME=359
	,patNAME=360
	,patSTRING=361
	,patPAIR=362


#line 283 "/usr/local/lib/bison.cc"
 /* enum token */
     }; /* end of enum declaration */
#endif
public:
 int YY_patBisonSpec_PARSE (YY_patBisonSpec_PARSE_PARAM);
 virtual void YY_patBisonSpec_ERROR(char *msg) YY_patBisonSpec_ERROR_BODY;
#ifdef YY_patBisonSpec_PURE
#ifdef YY_patBisonSpec_LSP_NEEDED
 virtual int  YY_patBisonSpec_LEX (YY_patBisonSpec_STYPE *YY_patBisonSpec_LVAL,YY_patBisonSpec_LTYPE *YY_patBisonSpec_LLOC) YY_patBisonSpec_LEX_BODY;
#else
 virtual int  YY_patBisonSpec_LEX (YY_patBisonSpec_STYPE *YY_patBisonSpec_LVAL) YY_patBisonSpec_LEX_BODY;
#endif
#else
 virtual int YY_patBisonSpec_LEX() YY_patBisonSpec_LEX_BODY;
 YY_patBisonSpec_STYPE YY_patBisonSpec_LVAL;
#ifdef YY_patBisonSpec_LSP_NEEDED
 YY_patBisonSpec_LTYPE YY_patBisonSpec_LLOC;
#endif
 int   YY_patBisonSpec_NERRS;
 int    YY_patBisonSpec_CHAR;
#endif
#if YY_patBisonSpec_DEBUG != 0
 int YY_patBisonSpec_DEBUG_FLAG;   /*  nonzero means print parse trace     */
#endif
public:
 YY_patBisonSpec_CLASS(YY_patBisonSpec_CONSTRUCTOR_PARAM);
public:
 YY_patBisonSpec_MEMBERS 
};
/* other declare folow */
#if YY_patBisonSpec_USE_CONST_TOKEN != 0

/* #line 314 "/usr/local/lib/bison.cc" */
#line 843 "patSpecParser.yy.tab.c"
const int YY_patBisonSpec_CLASS::pat_gevDataFile=258;
const int YY_patBisonSpec_CLASS::pat_gevModelDescription=259;
const int YY_patBisonSpec_CLASS::pat_gevChoice=260;
const int YY_patBisonSpec_CLASS::pat_gevPanel=261;
const int YY_patBisonSpec_CLASS::pat_gevWeight=262;
const int YY_patBisonSpec_CLASS::pat_gevBeta=263;
const int YY_patBisonSpec_CLASS::pat_gevBoxCox=264;
const int YY_patBisonSpec_CLASS::pat_gevBoxTukey=265;
const int YY_patBisonSpec_CLASS::pat_gevLatex1=266;
const int YY_patBisonSpec_CLASS::pat_gevLatex2=267;
const int YY_patBisonSpec_CLASS::pat_gevMu=268;
const int YY_patBisonSpec_CLASS::pat_gevSampleEnum=269;
const int YY_patBisonSpec_CLASS::pat_gevGnuplot=270;
const int YY_patBisonSpec_CLASS::pat_gevUtilities=271;
const int YY_patBisonSpec_CLASS::pat_gevGeneralizedUtilities=272;
const int YY_patBisonSpec_CLASS::pat_gevDerivatives=273;
const int YY_patBisonSpec_CLASS::pat_gevParameterCovariances=274;
const int YY_patBisonSpec_CLASS::pat_gevExpr=275;
const int YY_patBisonSpec_CLASS::pat_gevGroup=276;
const int YY_patBisonSpec_CLASS::pat_gevExclude=277;
const int YY_patBisonSpec_CLASS::pat_gevScale=278;
const int YY_patBisonSpec_CLASS::pat_gevModel=279;
const int YY_patBisonSpec_CLASS::pat_gevNLNests=280;
const int YY_patBisonSpec_CLASS::pat_gevCNLAlpha=281;
const int YY_patBisonSpec_CLASS::pat_gevCNLNests=282;
const int YY_patBisonSpec_CLASS::pat_gevRatios=283;
const int YY_patBisonSpec_CLASS::pat_gevDraws=284;
const int YY_patBisonSpec_CLASS::pat_gevConstraintNestCoef=285;
const int YY_patBisonSpec_CLASS::pat_gevConstantProduct=286;
const int YY_patBisonSpec_CLASS::pat_gevNetworkGEVNodes=287;
const int YY_patBisonSpec_CLASS::pat_gevNetworkGEVLinks=288;
const int YY_patBisonSpec_CLASS::pat_gevLinearConstraints=289;
const int YY_patBisonSpec_CLASS::pat_gevNonLinearEqualityConstraints=290;
const int YY_patBisonSpec_CLASS::pat_gevNonLinearInequalityConstraints=291;
const int YY_patBisonSpec_CLASS::pat_gevLogitKernelSigmas=292;
const int YY_patBisonSpec_CLASS::pat_gevLogitKernelFactors=293;
const int YY_patBisonSpec_CLASS::pat_gevDiscreteDistributions=294;
const int YY_patBisonSpec_CLASS::pat_gevSelectionBias=295;
const int YY_patBisonSpec_CLASS::pat_gevSNP=296;
const int YY_patBisonSpec_CLASS::pat_gevAggregateLast=297;
const int YY_patBisonSpec_CLASS::pat_gevAggregateWeight=298;
const int YY_patBisonSpec_CLASS::pat_gevMassAtZero=299;
const int YY_patBisonSpec_CLASS::pat_gevOrdinalLogit=300;
const int YY_patBisonSpec_CLASS::pat_gevRegressionModels=301;
const int YY_patBisonSpec_CLASS::pat_gevDurationModel=302;
const int YY_patBisonSpec_CLASS::pat_gevZhengFosgerau=303;
const int YY_patBisonSpec_CLASS::pat_gevGeneralizedExtremeValue=304;
const int YY_patBisonSpec_CLASS::pat_gevIIATest=305;
const int YY_patBisonSpec_CLASS::pat_gevProbaStandardErrors=306;
const int YY_patBisonSpec_CLASS::pat_gevBP=307;
const int YY_patBisonSpec_CLASS::pat_gevOL=308;
const int YY_patBisonSpec_CLASS::pat_gevMNL=309;
const int YY_patBisonSpec_CLASS::pat_gevNL=310;
const int YY_patBisonSpec_CLASS::pat_gevCNL=311;
const int YY_patBisonSpec_CLASS::pat_gevNGEV=312;
const int YY_patBisonSpec_CLASS::pat_gevNONE=313;
const int YY_patBisonSpec_CLASS::pat_gevROOT=314;
const int YY_patBisonSpec_CLASS::pat_gevCOLUMNS=315;
const int YY_patBisonSpec_CLASS::pat_gevLOOP=316;
const int YY_patBisonSpec_CLASS::pat_gevDERIV=317;
const int YY_patBisonSpec_CLASS::pat_gevACQ=318;
const int YY_patBisonSpec_CLASS::pat_gevSIGMA_ACQ=319;
const int YY_patBisonSpec_CLASS::pat_gevLOG_ACQ=320;
const int YY_patBisonSpec_CLASS::pat_gevVAL=321;
const int YY_patBisonSpec_CLASS::pat_gevSIGMA_VAL=322;
const int YY_patBisonSpec_CLASS::pat_gevLOG_VAL=323;
const int YY_patBisonSpec_CLASS::pat_gevE=324;
const int YY_patBisonSpec_CLASS::pat_gevP=325;
const int YY_patBisonSpec_CLASS::patOR=326;
const int YY_patBisonSpec_CLASS::patAND=327;
const int YY_patBisonSpec_CLASS::patEQUAL=328;
const int YY_patBisonSpec_CLASS::patNOTEQUAL=329;
const int YY_patBisonSpec_CLASS::patLESS=330;
const int YY_patBisonSpec_CLASS::patLESSEQUAL=331;
const int YY_patBisonSpec_CLASS::patGREAT=332;
const int YY_patBisonSpec_CLASS::patGREATEQUAL=333;
const int YY_patBisonSpec_CLASS::patNOT=334;
const int YY_patBisonSpec_CLASS::patPLUS=335;
const int YY_patBisonSpec_CLASS::patMINUS=336;
const int YY_patBisonSpec_CLASS::patMULT=337;
const int YY_patBisonSpec_CLASS::patDIVIDE=338;
const int YY_patBisonSpec_CLASS::patMOD=339;
const int YY_patBisonSpec_CLASS::patPOWER=340;
const int YY_patBisonSpec_CLASS::patUNARYMINUS=341;
const int YY_patBisonSpec_CLASS::patMAX=342;
const int YY_patBisonSpec_CLASS::patMIN=343;
const int YY_patBisonSpec_CLASS::patSQRT=344;
const int YY_patBisonSpec_CLASS::patLOG=345;
const int YY_patBisonSpec_CLASS::patEXP=346;
const int YY_patBisonSpec_CLASS::patABS=347;
const int YY_patBisonSpec_CLASS::patINT=348;
const int YY_patBisonSpec_CLASS::patOPPAR=349;
const int YY_patBisonSpec_CLASS::patCLPAR=350;
const int YY_patBisonSpec_CLASS::patOPBRA=351;
const int YY_patBisonSpec_CLASS::patCLBRA=352;
const int YY_patBisonSpec_CLASS::patOPCUR=353;
const int YY_patBisonSpec_CLASS::patCLCUR=354;
const int YY_patBisonSpec_CLASS::patCOMMA=355;
const int YY_patBisonSpec_CLASS::patCOLON=356;
const int YY_patBisonSpec_CLASS::patINTEGER=357;
const int YY_patBisonSpec_CLASS::patREAL=358;
const int YY_patBisonSpec_CLASS::patTIME=359;
const int YY_patBisonSpec_CLASS::patNAME=360;
const int YY_patBisonSpec_CLASS::patSTRING=361;
const int YY_patBisonSpec_CLASS::patPAIR=362;


#line 314 "/usr/local/lib/bison.cc"
 /* const YY_patBisonSpec_CLASS::token */
#endif
/*apres const  */
YY_patBisonSpec_CLASS::YY_patBisonSpec_CLASS(YY_patBisonSpec_CONSTRUCTOR_PARAM) YY_patBisonSpec_CONSTRUCTOR_INIT
{
#if YY_patBisonSpec_DEBUG != 0
YY_patBisonSpec_DEBUG_FLAG=0;
#endif
YY_patBisonSpec_CONSTRUCTOR_CODE;
};
#endif

/* #line 325 "/usr/local/lib/bison.cc" */
#line 965 "patSpecParser.yy.tab.c"


#define	YYFINAL		524
#define	YYFLAG		-32768
#define	YYNTBASE	108

#define YYTRANSLATE(x) ((unsigned)(x) <= 362 ? yytranslate[x] : 244)

static const char yytranslate[] = {     0,
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
   106,   107
};

#if YY_patBisonSpec_DEBUG != 0
static const short yyprhs[] = {     0,
     0,     2,     4,     7,     9,    11,    13,    15,    17,    19,
    21,    23,    25,    27,    29,    31,    33,    35,    37,    39,
    41,    43,    45,    47,    49,    51,    53,    55,    57,    59,
    61,    63,    65,    67,    69,    71,    73,    75,    77,    79,
    81,    83,    85,    87,    89,    91,    93,    95,    98,   102,
   105,   108,   111,   113,   115,   117,   120,   123,   126,   129,
   132,   135,   138,   141,   144,   147,   148,   150,   153,   156,
   158,   160,   163,   166,   169,   172,   175,   177,   180,   186,
   192,   197,   200,   203,   205,   207,   210,   217,   220,   222,
   224,   227,   230,   233,   235,   237,   240,   244,   247,   249,
   251,   254,   263,   272,   275,   278,   282,   286,   288,   291,
   293,   295,   297,   299,   303,   307,   311,   315,   318,   320,
   322,   325,   328,   331,   334,   336,   338,   341,   344,   347,
   349,   351,   354,   358,   361,   363,   365,   368,   372,   377,
   385,   388,   391,   394,   397,   399,   402,   404,   406,   409,
   415,   418,   420,   422,   424,   426,   428,   430,   433,   436,
   438,   441,   448,   450,   453,   456,   459,   461,   464,   470,
   473,   476,   478,   481,   488,   491,   493,   495,   498,   502,
   505,   508,   511,   514,   516,   518,   521,   523,   525,   527,
   530,   534,   538,   542,   544,   546,   548,   550,   553,   557,
   561,   563,   567,   569,   571,   577,   580,   582,   584,   587,
   593,   596,   598,   600,   603,   610,   613,   615,   617,   620,
   624,   627,   629,   631,   634,   638,   641,   643,   645,   648,
   650,   653,   656,   659,   661,   663,   666,   669,   672,   674,
   676,   679,   684,   686,   689,   694,   699,   702,   704,   707,
   712,   714,   716,   720,   724,   728,   730,   732,   734,   736,
   740,   742,   744,   746,   753,   758,   763,   766,   769,   774,
   779,   784,   789,   794,   798,   802,   806,   810,   814,   818,
   822,   826,   830,   834,   838,   842,   846,   850,   857,   864,
   868,   872,   874,   876,   878,   880,   882,   884,   886,   888,
   890
};

static const short yyrhs[] = {   109,
     0,   110,     0,   109,   110,     0,   111,     0,   113,     0,
   118,     0,   121,     0,   127,     0,   128,     0,   131,     0,
   133,     0,   132,     0,   134,     0,   158,     0,   161,     0,
   124,     0,   225,     0,   164,     0,   168,     0,   169,     0,
   170,     0,   173,     0,   175,     0,   179,     0,   182,     0,
   185,     0,   188,     0,   114,     0,   209,     0,   206,     0,
   200,     0,   203,     0,   189,     0,   190,     0,   191,     0,
   220,     0,   217,     0,   119,     0,   120,     0,   212,     0,
   155,     0,   148,     0,   147,     0,   143,     0,   146,     0,
   137,     0,   140,     0,     3,   112,     0,    60,    73,   243,
     0,     4,   123,     0,   115,    58,     0,   115,   116,     0,
    11,     0,    12,     0,   117,     0,   116,   117,     0,   241,
   240,     0,     5,   230,     0,    42,    58,     0,    42,   230,
     0,    43,    58,     0,    43,   230,     0,     6,    58,     0,
     6,   122,     0,   230,   123,     0,     0,   239,     0,   123,
   239,     0,    44,   125,     0,    58,     0,   126,     0,   125,
   126,     0,   239,   242,     0,     7,    58,     0,     7,   230,
     0,     8,   129,     0,   130,     0,   129,   130,     0,   241,
   238,   238,   238,   243,     0,    13,   238,   238,   238,   243,
     0,    15,   241,   238,   238,     0,    14,   243,     0,    19,
   135,     0,    58,     0,   136,     0,   135,   136,     0,   241,
   241,   238,   238,   238,   243,     0,    50,   138,     0,    58,
     0,   139,     0,   138,   139,     0,   241,   178,     0,    51,
   141,     0,    58,     0,   142,     0,   141,   142,     0,   241,
   241,   238,     0,    48,   144,     0,    58,     0,   145,     0,
   144,   145,     0,    70,    98,   241,    99,   238,   238,   238,
   240,     0,    69,    98,   230,    99,   238,   238,   238,   240,
     0,    49,   241,     0,    49,    58,     0,    47,   241,   241,
     0,    46,   241,   149,     0,   150,     0,   149,   150,     0,
   151,     0,   153,     0,   152,     0,   154,     0,    63,    73,
   228,     0,    64,    73,   241,     0,    66,    73,   228,     0,
    67,    73,   241,     0,    45,   156,     0,    58,     0,   157,
     0,   156,   157,     0,   243,    58,     0,   243,   241,     0,
    17,   159,     0,    58,     0,   160,     0,   159,   160,     0,
   243,   230,     0,    18,   162,     0,    58,     0,   163,     0,
   162,   163,     0,   243,   241,   230,     0,    20,   165,     0,
    58,     0,   166,     0,   165,   166,     0,   241,    73,   230,
     0,   167,   241,    73,   230,     0,    61,    98,   241,   243,
   243,   243,    99,     0,    21,    58,     0,    21,   230,     0,
    22,    58,     0,    22,   230,     0,    23,     0,    23,   171,
     0,    58,     0,   172,     0,   171,   172,     0,   243,   238,
   238,   238,   243,     0,    24,   174,     0,    53,     0,    52,
     0,    54,     0,    55,     0,    56,     0,    57,     0,    25,
    58,     0,    25,   176,     0,   177,     0,   176,   177,     0,
   241,   238,   238,   238,   243,   178,     0,   243,     0,   178,
   243,     0,    27,    58,     0,    27,   180,     0,   181,     0,
   180,   181,     0,   241,   238,   238,   238,   243,     0,    26,
    58,     0,    26,   183,     0,   184,     0,   183,   184,     0,
   241,   241,   238,   238,   238,   243,     0,    28,   186,     0,
    58,     0,   187,     0,   186,   187,     0,   241,   241,   241,
     0,    29,   243,     0,    34,   194,     0,    35,   192,     0,
    36,   192,     0,    58,     0,   193,     0,   192,   193,     0,
   230,     0,    58,     0,   195,     0,   194,   195,     0,   196,
    76,   238,     0,   196,    73,   238,     0,   196,    78,   238,
     0,    76,     0,    78,     0,    73,     0,   197,     0,    81,
   197,     0,   196,    80,   197,     0,   196,    81,   197,     0,
   198,     0,   238,    82,   198,     0,   241,     0,   199,     0,
    94,   241,   100,   241,    95,     0,    32,   201,     0,    58,
     0,   202,     0,   201,   202,     0,   241,   238,   238,   238,
   243,     0,    33,   204,     0,    58,     0,   205,     0,   204,
   205,     0,   241,   241,   238,   238,   238,   243,     0,    31,
   207,     0,    58,     0,   208,     0,   207,   208,     0,   241,
   241,   238,     0,    30,   210,     0,    58,     0,   211,     0,
   210,   211,     0,   241,    73,   241,     0,    41,   213,     0,
    58,     0,   214,     0,   241,   215,     0,   216,     0,   215,
   216,     0,   243,   241,     0,    40,   218,     0,    58,     0,
   219,     0,   218,   219,     0,   243,   241,     0,    39,   221,
     0,    58,     0,   222,     0,   221,   222,     0,   241,    75,
   223,    77,     0,   224,     0,   223,   224,     0,   241,    94,
   241,    95,     0,   231,    94,   241,    95,     0,    16,   226,
     0,   227,     0,   226,   227,     0,   243,   241,   241,   228,
     0,    58,     0,   229,     0,   228,    80,   229,     0,   241,
    82,   241,     0,   231,    82,   241,     0,   237,     0,   235,
     0,   236,     0,   231,     0,    94,   230,    95,     0,   232,
     0,   233,     0,   234,     0,    62,    94,   230,   100,   241,
    95,     0,   241,    96,   241,    97,     0,   241,    98,   241,
    99,     0,    81,   230,     0,    79,   230,     0,    89,    94,
   230,    95,     0,    90,    94,   230,    95,     0,    91,    94,
   230,    95,     0,    92,    94,   230,    95,     0,    93,    94,
   230,    95,     0,    94,   235,    95,     0,   230,    80,   230,
     0,   230,    81,   230,     0,   230,    82,   230,     0,   230,
    83,   230,     0,   230,    85,   230,     0,   230,    73,   230,
     0,   230,    74,   230,     0,   230,    71,   230,     0,   230,
    72,   230,     0,   230,    75,   230,     0,   230,    76,   230,
     0,   230,    77,   230,     0,   230,    78,   230,     0,    87,
    94,   230,   100,   230,    95,     0,    88,    94,   230,   100,
   230,    95,     0,   230,    84,   230,     0,    94,   236,    95,
     0,   238,     0,   241,     0,   242,     0,   243,     0,   240,
     0,   241,     0,   106,     0,   105,     0,   103,     0,   102,
     0
};

#endif

#if YY_patBisonSpec_DEBUG != 0
static const short yyrline[] = { 0,
   309,   314,   314,   317,   318,   319,   320,   321,   322,   323,
   324,   325,   326,   327,   328,   329,   330,   331,   332,   333,
   334,   335,   336,   337,   338,   339,   340,   341,   342,   343,
   344,   345,   346,   347,   348,   349,   350,   351,   352,   353,
   354,   355,   356,   357,   358,   359,   360,   366,   369,   373,
   380,   381,   385,   385,   387,   390,   394,   399,   406,   407,
   415,   416,   425,   428,   432,   440,   445,   449,   456,   458,
   459,   460,   462,   468,   471,   479,   481,   481,   483,   492,
   498,   505,   511,   513,   514,   515,   517,   526,   528,   529,
   530,   532,   538,   540,   541,   542,   544,   555,   557,   558,
   559,   561,   578,   593,   597,   599,   606,   611,   611,   613,
   613,   613,   613,   615,   622,   628,   633,   637,   639,   640,
   641,   643,   645,   651,   654,   655,   656,   658,   663,   666,
   667,   668,   670,   680,   682,   683,   684,   686,   697,   710,
   722,   731,   738,   744,   752,   752,   754,   754,   754,   756,
   762,   764,   769,   773,   777,   781,   785,   792,   794,   796,
   796,   798,   807,   811,   818,   821,   823,   823,   825,   833,
   836,   838,   838,   840,   852,   854,   854,   854,   856,   867,
   871,   877,   882,   887,   891,   897,   903,   908,   912,   918,
   924,   934,   942,   953,   953,   953,   955,   961,   969,   974,
   980,   987,   995,   998,  1002,  1010,  1011,  1011,  1011,  1012,
  1020,  1021,  1021,  1021,  1022,  1033,  1035,  1035,  1035,  1037,
  1057,  1059,  1059,  1059,  1061,  1074,  1076,  1076,  1078,  1084,
  1084,  1086,  1094,  1096,  1096,  1096,  1098,  1103,  1105,  1105,
  1105,  1107,  1119,  1123,  1129,  1133,  1142,  1144,  1144,  1146,
  1162,  1165,  1171,  1182,  1195,  1210,  1213,  1216,  1219,  1222,
  1225,  1230,  1233,  1237,  1247,  1265,  1285,  1293,  1301,  1308,
  1316,  1323,  1331,  1338,  1343,  1352,  1362,  1372,  1381,  1390,
  1400,  1410,  1420,  1430,  1440,  1450,  1460,  1470,  1480,  1490,
  1500,  1504,  1510,  1519,  1522,  1526,  1529,  1533,  1538,  1545,
  1550
};

static const char * const yytname[] = {   "$","error","$illegal.","pat_gevDataFile",
"pat_gevModelDescription","pat_gevChoice","pat_gevPanel","pat_gevWeight","pat_gevBeta",
"pat_gevBoxCox","pat_gevBoxTukey","pat_gevLatex1","pat_gevLatex2","pat_gevMu",
"pat_gevSampleEnum","pat_gevGnuplot","pat_gevUtilities","pat_gevGeneralizedUtilities",
"pat_gevDerivatives","pat_gevParameterCovariances","pat_gevExpr","pat_gevGroup",
"pat_gevExclude","pat_gevScale","pat_gevModel","pat_gevNLNests","pat_gevCNLAlpha",
"pat_gevCNLNests","pat_gevRatios","pat_gevDraws","pat_gevConstraintNestCoef",
"pat_gevConstantProduct","pat_gevNetworkGEVNodes","pat_gevNetworkGEVLinks","pat_gevLinearConstraints",
"pat_gevNonLinearEqualityConstraints","pat_gevNonLinearInequalityConstraints",
"pat_gevLogitKernelSigmas","pat_gevLogitKernelFactors","pat_gevDiscreteDistributions",
"pat_gevSelectionBias","pat_gevSNP","pat_gevAggregateLast","pat_gevAggregateWeight",
"pat_gevMassAtZero","pat_gevOrdinalLogit","pat_gevRegressionModels","pat_gevDurationModel",
"pat_gevZhengFosgerau","pat_gevGeneralizedExtremeValue","pat_gevIIATest","pat_gevProbaStandardErrors",
"pat_gevBP","pat_gevOL","pat_gevMNL","pat_gevNL","pat_gevCNL","pat_gevNGEV",
"pat_gevNONE","pat_gevROOT","pat_gevCOLUMNS","pat_gevLOOP","pat_gevDERIV","pat_gevACQ",
"pat_gevSIGMA_ACQ","pat_gevLOG_ACQ","pat_gevVAL","pat_gevSIGMA_VAL","pat_gevLOG_VAL",
"pat_gevE","pat_gevP","patOR","patAND","patEQUAL","patNOTEQUAL","patLESS","patLESSEQUAL",
"patGREAT","patGREATEQUAL","patNOT","patPLUS","patMINUS","patMULT","patDIVIDE",
"patMOD","patPOWER","patUNARYMINUS","patMAX","patMIN","patSQRT","patLOG","patEXP",
"patABS","patINT","patOPPAR","patCLPAR","patOPBRA","patCLBRA","patOPCUR","patCLCUR",
"patCOMMA","patCOLON","patINTEGER","patREAL","patTIME","patNAME","patSTRING",
"patPAIR","everything","sections","section","dataFileSec","dataFileColumns",
"modelDescSec","latexSec","latexHead","listOfLatexNames","latexName","choiceSec",
"aggregateLastSection","aggregateWeightSection","panelSec","panelDescription",
"listOfNames","massAtZeroSec","listOfMassAtZero","oneMassAtZero","weightSec",
"betaSec","betaList","oneBeta","muSec","gnuplotSec","sampleEnumSec","parameterCovarSec",
"listOfCovariances","covariance","IIATestSection","listOfIIATests","oneIIATest",
"probaStandardErrorsSection","listOfProbaStandardErrors","oneProbaStandardError",
"zhengFosgerauSection","listOfZheng","oneZheng","generalizedExtremeValueSection",
"durationModelSection","regressionModelSection","listOfRegModels","oneRegMode",
"acqRegressionModel","acqSigma","valRegressionModel","valSigma","ordinalLogitSection",
"listOfOrdinalLogit","ordinalLogit","generalizedUtilitiesSec","generalizedUtilities",
"generalizedUtility","derivativesSec","derivatives","oneDerivative","exprSec",
"listExpr","exprDef","aloop","groupSec","excludeSec","scaleSec","scaleList",
"oneScale","modelSec","modelType","nlnestsSec","nestList","oneNest","listId",
"cnlnestsSec","cnlNests","oneCnlNest","cnlalphaSec","cnlAlphas","oneAlpha","ratiosSec",
"ratiosList","oneRatio","drawsSec","linearConstraintsSec","nonLinearEqualityConstraintsSec",
"nonLinearInequalityConstraintsSec","nonLinearConstraintsList","oneNonLinearConstraint",
"constraintsList","oneConstraint","equation","eqTerm","parameter","pairParam",
"networkGevNodeSec","networkGevNodeList","oneNetworkGevNode","networkGevLinkSec",
"networkGevLinkList","oneNetworkGevLink","constantProductSec","constProdList",
"oneConstProd","constraintNestSec","constraintNestList","oneConstraintNest",
"SNPsection","defSnpTerms","nonEmptySnpTerms","listOfSnpTerms","oneSnpTerm",
"selectionBiasSec","listOfSelectionBias","oneSelectionBias","discreteDistSec",
"listOfDiscreteParameters","oneDiscreteParameter","listOfDiscreteTerms","oneDiscreteTerm",
"utilitiesSec","utilList","util","utilExpression","utilTerm","expression","any_random_expression",
"deriv_expression","random_expression","unirandom_expression","unary_expression",
"binary_expression","simple_expression","numberParam","anystringParam","stringParam",
"nameParam","floatParam","intParam","intParam"
};
#endif

static const short yyr1[] = {     0,
   108,   109,   109,   110,   110,   110,   110,   110,   110,   110,
   110,   110,   110,   110,   110,   110,   110,   110,   110,   110,
   110,   110,   110,   110,   110,   110,   110,   110,   110,   110,
   110,   110,   110,   110,   110,   110,   110,   110,   110,   110,
   110,   110,   110,   110,   110,   110,   110,   111,   112,   113,
   114,   114,   115,   115,   116,   116,   117,   118,   119,   119,
   120,   120,   121,   121,   122,   123,   123,   123,   124,   125,
   125,   125,   126,   127,   127,   128,   129,   129,   130,   131,
   132,   133,   134,   135,   135,   135,   136,   137,   138,   138,
   138,   139,   140,   141,   141,   141,   142,   143,   144,   144,
   144,   145,   145,   146,   146,   147,   148,   149,   149,   150,
   150,   150,   150,   151,   152,   153,   154,   155,   156,   156,
   156,   157,   157,   158,   159,   159,   159,   160,   161,   162,
   162,   162,   163,   164,   165,   165,   165,   166,   166,   167,
   168,   168,   169,   169,   170,   170,   171,   171,   171,   172,
   173,   174,   174,   174,   174,   174,   174,   175,   175,   176,
   176,   177,   178,   178,   179,   179,   180,   180,   181,   182,
   182,   183,   183,   184,   185,   186,   186,   186,   187,   188,
   189,   190,   191,   192,   192,   192,   193,   194,   194,   194,
   195,   195,   195,    -1,    -1,    -1,   196,   196,   196,   196,
   197,   197,   198,   198,   199,   200,   201,   201,   201,   202,
   203,   204,   204,   204,   205,   206,   207,   207,   207,   208,
   209,   210,   210,   210,   211,   212,   213,   213,   214,   215,
   215,   216,   217,   218,   218,   218,   219,   220,   221,   221,
   221,   222,   223,   223,   224,   224,   225,   226,   226,   227,
   228,   228,   228,   229,   229,   230,   230,   230,   230,   230,
   230,   231,   231,   232,   233,   234,   235,   235,   235,   235,
   235,   235,   235,   235,   236,   236,   236,   236,   236,   236,
   236,   236,   236,   236,   236,   236,   236,   236,   236,   236,
   236,   237,   237,   238,   238,   239,   239,   240,   241,   242,
   243
};

static const short yyr2[] = {     0,
     1,     1,     2,     1,     1,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     2,     3,     2,
     2,     2,     1,     1,     1,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     0,     1,     2,     2,     1,
     1,     2,     2,     2,     2,     2,     1,     2,     5,     5,
     4,     2,     2,     1,     1,     2,     6,     2,     1,     1,
     2,     2,     2,     1,     1,     2,     3,     2,     1,     1,
     2,     8,     8,     2,     2,     3,     3,     1,     2,     1,
     1,     1,     1,     3,     3,     3,     3,     2,     1,     1,
     2,     2,     2,     2,     1,     1,     2,     2,     2,     1,
     1,     2,     3,     2,     1,     1,     2,     3,     4,     7,
     2,     2,     2,     2,     1,     2,     1,     1,     2,     5,
     2,     1,     1,     1,     1,     1,     1,     2,     2,     1,
     2,     6,     1,     2,     2,     2,     1,     2,     5,     2,
     2,     1,     2,     6,     2,     1,     1,     2,     3,     2,
     2,     2,     2,     1,     1,     2,     1,     1,     1,     2,
     3,     3,     3,     1,     1,     1,     1,     2,     3,     3,
     1,     3,     1,     1,     5,     2,     1,     1,     2,     5,
     2,     1,     1,     2,     6,     2,     1,     1,     2,     3,
     2,     1,     1,     2,     3,     2,     1,     1,     2,     1,
     2,     2,     2,     1,     1,     2,     2,     2,     1,     1,
     2,     4,     1,     2,     4,     4,     2,     1,     2,     4,
     1,     1,     3,     3,     3,     1,     1,     1,     1,     3,
     1,     1,     1,     6,     4,     4,     2,     2,     4,     4,
     4,     4,     4,     3,     3,     3,     3,     3,     3,     3,
     3,     3,     3,     3,     3,     3,     3,     6,     6,     3,
     3,     1,     1,     1,     1,     1,     1,     1,     1,     1,
     1
};

static const short yydefact[] = {     0,
     0,    66,     0,     0,     0,     0,    53,    54,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,   145,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     1,     2,     4,     5,    28,
     0,     6,    38,    39,     7,    16,     8,     9,    10,    12,
    11,    13,    46,    47,    44,    45,    43,    42,    41,    14,
    15,    18,    19,    20,    21,    22,    23,    24,    25,    26,
    27,    33,    34,    35,    31,    32,    30,    29,    40,    37,
    36,    17,     0,    48,   299,   298,    50,    67,   296,   297,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,   301,   300,    58,   259,   261,   262,   263,   257,   258,
   256,   292,   293,   294,   295,    63,    64,    66,    74,    75,
    76,    77,     0,     0,    82,     0,   247,   248,     0,   125,
   124,   126,     0,   130,   129,   131,     0,    84,    83,    85,
     0,   135,     0,   134,   136,     0,     0,   141,   142,   143,
   144,   147,   146,   148,     0,   153,   152,   154,   155,   156,
   157,   151,   158,   159,   160,     0,   170,   171,   172,     0,
   165,   166,   167,     0,   176,   175,   177,     0,   180,   222,
   221,   223,     0,   217,   216,   218,     0,   207,   206,   208,
     0,   212,   211,   213,     0,   188,     0,     0,   181,   189,
     0,   197,   201,   204,     0,   203,   184,   182,   185,   187,
   183,   239,   238,   240,     0,   234,   233,   235,     0,   227,
   226,   228,     0,    59,    60,    61,    62,    70,    69,    71,
     0,   119,   118,   120,     0,     0,     0,    99,     0,     0,
    98,   100,   105,   104,    89,    88,    90,     0,    94,    93,
    95,     0,     3,    51,    52,    55,     0,     0,    68,     0,
   268,   267,     0,     0,     0,     0,     0,     0,     0,     0,
   257,   258,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,    65,    78,
     0,     0,     0,   249,     0,   127,   128,   132,     0,    86,
     0,     0,   137,     0,     0,   149,     0,   161,     0,   173,
     0,   168,     0,   178,     0,   224,     0,   219,     0,   209,
     0,   214,     0,   198,     0,   190,     0,     0,     0,     0,
     0,     0,   186,   241,     0,   236,   237,   229,   230,     0,
    72,    73,   121,   122,   123,     0,     0,     0,     0,   107,
   108,   110,   112,   111,   113,   106,     0,     0,   101,    91,
    92,   163,    96,     0,    56,    57,    49,     0,     0,     0,
     0,     0,     0,     0,     0,   260,   274,   291,   282,   283,
   280,   281,   284,   285,   286,   287,   275,   276,   277,   278,
   290,   279,     0,     0,     0,     0,    81,     0,   133,     0,
     0,     0,   138,     0,     0,     0,     0,   179,   225,   220,
     0,     0,     0,   192,   191,   193,   199,   200,   202,     0,
   243,     0,     0,   231,   232,     0,     0,     0,     0,   109,
     0,     0,   164,    97,     0,     0,     0,   269,   270,   271,
   272,   273,   265,   266,     0,    80,   251,   250,   252,     0,
     0,     0,     0,   139,     0,     0,     0,     0,     0,     0,
     0,   242,   244,     0,     0,   114,   115,   116,   117,     0,
     0,     0,     0,     0,    79,     0,     0,     0,     0,     0,
   150,     0,     0,   169,   210,     0,   205,     0,     0,     0,
     0,   264,   288,   289,   253,   255,   254,    87,     0,   162,
   174,   215,   246,   245,     0,     0,   140,     0,     0,   103,
   102,     0,     0,     0
};

static const short yydefgoto[] = {   522,
    46,    47,    48,    94,    49,    50,    51,   265,   266,    52,
    53,    54,    55,   127,    97,    56,   239,   240,    57,    58,
   131,   132,    59,    60,    61,    62,   149,   150,    63,   256,
   257,    64,   260,   261,    65,   251,   252,    66,    67,    68,
   360,   361,   362,   363,   364,   365,    69,   243,   244,    70,
   141,   142,    71,   145,   146,    72,   154,   155,   156,    73,
    74,    75,   163,   164,    76,   172,    77,   174,   175,   371,
    78,   182,   183,    79,   178,   179,    80,   186,   187,    81,
    82,    83,    84,   218,   219,   209,   210,   211,   212,   213,
   214,    85,   199,   200,    86,   203,   204,    87,   195,   196,
    88,   191,   192,    89,   231,   232,   348,   349,    90,   227,
   228,    91,   223,   224,   430,   431,    92,   137,   138,   458,
   459,   220,   115,   116,   117,   118,   119,   120,   121,   122,
    98,    99,   123,   124,   125
};

static const short yypact[] = {   792,
   -34,    -3,   602,   -11,    61,   -97,-32768,-32768,    11,   -79,
   -97,   -79,    57,    58,   -48,   -49,   116,   177,    60,   160,
   -47,   -44,   -43,   -38,   -79,   -36,   -31,   -30,   -21,   -39,
   446,   446,   -17,    70,   -15,   496,   545,   -53,    71,   -97,
   -97,   -52,    -8,    31,    40,   792,-32768,-32768,-32768,-32768,
    41,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,   -24,-32768,-32768,-32768,    -3,-32768,-32768,-32768,
     2,   602,   602,    13,    23,    28,    32,    38,    45,    47,
   602,-32768,-32768,  1030,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,   -67,-32768,-32768,-32768,-32768,   797,-32768,  1030,
   -97,-32768,    11,    11,-32768,    11,   -79,-32768,   -97,-32768,
   -79,-32768,   602,-32768,   -79,-32768,   -97,-32768,   -97,-32768,
   -97,-32768,    36,    72,-32768,   -97,    62,-32768,  1030,-32768,
  1030,-32768,   -79,-32768,    11,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,   -97,-32768,    11,-32768,   -97,-32768,   -97,
-32768,   -97,-32768,    11,-32768,   -97,-32768,   -97,-32768,-32768,
   -97,-32768,    95,-32768,   -97,-32768,   -97,-32768,   -97,-32768,
    11,-32768,   -97,-32768,   -97,-32768,     7,   -97,    88,-32768,
   108,-32768,-32768,-32768,    94,-32768,-32768,   602,-32768,  1030,
   602,-32768,   -97,-32768,    69,-32768,   -79,-32768,   -97,-32768,
-32768,-32768,   -79,-32768,  1030,-32768,  1030,-32768,    -3,-32768,
    77,-32768,   -79,-32768,    42,   185,   -97,-32768,    87,    89,
    51,-32768,-32768,-32768,-32768,   -97,-32768,   -79,-32768,   -97,
-32768,   -97,-32768,-32768,   -97,-32768,    90,   -79,-32768,   602,
   161,-32768,   602,   602,   602,   602,   602,   602,   602,   902,
    99,   104,   602,   602,   602,   602,   602,   602,   602,   602,
   602,   602,   602,   602,   602,   602,   -97,   -97,    -3,-32768,
    11,    11,    11,-32768,   -97,-32768,  1030,-32768,   602,-32768,
    11,   -97,-32768,   128,   602,-32768,    11,-32768,    11,-32768,
    11,-32768,    11,-32768,   -97,-32768,   -97,-32768,    11,-32768,
    11,-32768,    11,-32768,   120,-32768,    11,    11,    11,     7,
     7,   -69,-32768,-32768,   -97,-32768,-32768,   -79,-32768,   -97,
-32768,-32768,-32768,-32768,-32768,   155,   163,   164,   165,   185,
-32768,-32768,-32768,-32768,-32768,-32768,   602,   -97,-32768,-32768,
   -79,-32768,-32768,    11,-32768,-32768,-32768,   702,   840,   861,
   918,   934,   950,   966,   982,-32768,-32768,-32768,   149,   149,
   149,   149,   149,   149,   149,   149,   190,   190,    46,    46,
    46,   172,   143,   151,    11,   -79,-32768,    53,  1030,    11,
   -79,   602,  1030,    11,    11,    11,    11,-32768,-32768,-32768,
    11,    11,   -97,-32768,-32768,-32768,-32768,-32768,-32768,   -70,
-32768,   166,   -50,-32768,-32768,    53,   -97,    53,   -97,-32768,
   882,   162,-32768,-32768,   -97,   602,   602,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,   -79,-32768,-32768,   179,-32768,   194,
   -58,    11,   -79,  1030,   -79,   -79,    11,   -79,   -79,    11,
   182,-32768,-32768,   -97,   -97,   179,-32768,   179,-32768,    11,
    11,   186,   998,  1014,-32768,   -97,   -97,   -97,   -79,   -79,
-32768,   -79,   -79,-32768,-32768,   -79,-32768,   188,   189,    11,
    11,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   187,   -79,
-32768,-32768,-32768,-32768,    11,    11,-32768,    90,    90,-32768,
-32768,   285,   287,-32768
};

static const short yypgoto[] = {-32768,
-32768,   243,-32768,-32768,-32768,-32768,-32768,-32768,    25,-32768,
-32768,-32768,-32768,-32768,   167,-32768,-32768,    52,-32768,-32768,
-32768,   168,-32768,-32768,-32768,-32768,-32768,   147,-32768,-32768,
    44,-32768,-32768,    37,-32768,-32768,    50,-32768,-32768,-32768,
-32768,   -62,-32768,-32768,-32768,-32768,-32768,-32768,    63,-32768,
-32768,   170,-32768,-32768,   158,-32768,-32768,   150,-32768,-32768,
-32768,-32768,-32768,   142,-32768,-32768,-32768,-32768,   133,  -183,
-32768,-32768,   131,-32768,-32768,   137,-32768,-32768,   130,-32768,
-32768,-32768,-32768,   286,  -205,-32768,   110,-32768,  -203,   -25,
-32768,-32768,-32768,   122,-32768,-32768,   123,-32768,-32768,   132,
-32768,-32768,   134,-32768,-32768,-32768,-32768,   -14,-32768,-32768,
    96,-32768,-32768,   112,-32768,   -94,-32768,-32768,   196,  -330,
  -149,    68,  -343,-32768,-32768,-32768,   228,   229,-32768,    -9,
   -37,  -264,   422,   107,    20
};


#define	YYLAST		1115


static const short yytable[] = {   134,
   241,   432,   376,   334,   238,   248,   472,    95,   152,   148,
   173,   153,   343,   177,   181,   343,   249,   250,   206,   185,
   215,   190,   112,   488,   208,    93,   194,   198,   297,   135,
   298,   139,   143,   147,    95,    95,   202,   297,   165,   298,
   222,   207,   230,   475,   189,   297,   126,   298,   268,   253,
   101,    95,    96,   229,   208,    95,    95,    95,   245,   269,
    95,    95,   112,   113,   460,    95,    95,   102,    95,   103,
   114,   128,   130,    95,    95,   104,   105,   106,   107,   108,
   109,   110,   111,    95,   159,   161,   432,    95,   255,    95,
   112,   113,   460,    95,   460,   270,    95,   259,   264,   354,
   208,    95,    96,   235,   237,   476,   273,   478,   112,   113,
   457,    95,   112,   113,   140,   144,   274,   162,   129,   249,
   250,   275,   101,   301,   302,   276,   303,   226,   242,   295,
   296,   277,   153,   312,   315,    95,   427,   428,   278,   102,
   279,   103,   460,   345,    95,    95,    95,   104,   105,   106,
   107,   108,   109,   110,   111,   317,   139,    95,   112,   112,
   143,   112,   112,   113,   147,    95,   319,   327,   207,   271,
   272,   112,   112,   158,   323,   342,    95,   101,   280,   113,
   337,   208,   165,   338,   367,   339,   368,   340,   341,   112,
   113,   331,    95,   387,   102,    96,   103,   215,   388,   215,
   412,   241,   104,   105,   106,   107,   108,   109,   110,   111,
   307,   166,   167,   168,   169,   170,   171,   112,   113,   423,
    95,   285,   286,   287,   288,   289,   290,   436,   291,   292,
   293,   294,   295,   296,   160,   437,   438,   439,   101,   453,
   291,   292,   293,   294,   295,   296,   229,   356,   357,   454,
   358,   359,   350,   520,   521,   102,   296,   103,   486,   474,
   481,   269,   245,   104,   105,   106,   107,   108,   109,   110,
   111,   293,   294,   295,   296,   487,   497,   372,   112,   113,
   502,    95,   513,   514,   523,   517,   524,   377,   263,   375,
   351,   405,   406,   407,   299,   310,   373,   440,   300,   370,
   369,   410,   308,   313,   316,   353,   318,   414,   510,   415,
   306,   416,   322,   417,   320,   324,   429,   221,   336,   420,
   330,   421,   346,   422,   326,   332,   328,   424,   425,   426,
   215,   215,   304,   434,   344,   473,   505,   378,   281,   282,
   379,   380,   381,   382,   383,   384,   385,   352,     0,     0,
   389,   390,   391,   392,   393,   394,   395,   396,   397,   398,
   399,   400,   401,   402,   444,     0,     0,   350,     0,     0,
     0,     0,     0,     0,     0,     0,   409,     0,     0,     0,
     0,     0,   413,     0,     0,     0,     0,     0,     0,     0,
   443,     0,     0,     0,     0,   455,     0,     0,     0,     0,
   462,     0,     0,     0,   465,   466,   467,   468,     0,     0,
     0,   469,   470,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,   100,     0,   456,     0,   133,     0,     0,
   463,     0,   136,     0,   441,     0,   151,   157,     0,     0,
     0,     0,   176,   180,   184,   188,     0,   193,   197,   201,
   205,   216,   489,     0,   225,     0,   233,   493,     0,   100,
   496,   246,   247,     0,   254,   258,   262,     0,     0,     0,
   500,   501,   267,     0,   485,     0,     0,     0,     0,   464,
     0,     0,   490,     0,   491,   492,     0,   494,   495,     0,
   515,   516,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,   217,     0,   518,   519,   101,   508,   509,
     0,   372,   511,   483,   484,   512,     0,     0,   100,     0,
     0,     0,     0,     0,   102,     0,   103,     0,     0,   443,
     0,     0,   104,   105,   106,   107,   108,   109,   110,   111,
     0,     0,     0,     0,     0,     0,     0,   112,   113,   100,
    95,     0,   133,   234,     0,     0,     0,   101,     0,     0,
   305,     0,     0,     0,     0,     0,     0,     0,   309,     0,
   151,     0,   311,     0,   102,   157,   103,   314,     0,     0,
     0,     0,   104,   105,   106,   107,   108,   109,   110,   111,
     0,     0,     0,     0,     0,   176,     0,   112,   113,   180,
    95,   321,   236,   184,     0,     0,   101,   188,     0,   325,
     0,     0,   193,     0,     0,     0,   197,     0,   329,     0,
   201,     0,     0,   102,   205,   103,   333,     0,   216,   335,
   216,   104,   105,   106,   107,   108,   109,   110,   111,     0,
     0,     0,     0,     0,   225,     0,   112,   113,     0,    95,
   347,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   100,     0,     0,   101,     0,     0,   355,     0,   366,     0,
     0,     0,     0,     0,     0,     0,     0,   258,     0,     0,
   102,   262,   103,   374,     0,     0,   267,     0,   104,   105,
   106,   107,   108,   109,   110,   111,     0,     0,     0,     0,
     0,     0,     0,   112,   113,     0,    95,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,   403,   404,
   100,     0,     0,     0,     0,     0,   408,     0,     0,     0,
     0,     0,     0,   411,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,   418,     0,   419,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,   216,   216,   216,     0,     0,   433,     0,     0,     0,
     0,   435,   283,   284,   285,   286,   287,   288,   289,   290,
     0,   291,   292,   293,   294,   295,   296,     0,     0,   442,
     0,     0,     0,     0,     1,     2,     3,     4,     5,     6,
     0,   445,     7,     8,     9,    10,    11,    12,    13,    14,
    15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
    25,    26,    27,    28,    29,    30,    31,    32,     0,   461,
    33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
    43,    44,    45,     0,   471,     0,     0,     0,     0,     0,
     0,   433,     0,     0,     0,     0,     0,   461,   477,   461,
   479,     0,     0,     0,     0,     0,   482,   283,   284,   285,
   286,   287,   288,   289,   290,     0,   291,   292,   293,   294,
   295,   296,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,   498,   499,     0,     0,     0,
     0,    95,    96,     0,     0,     0,     0,   461,   506,   507,
   283,   284,   285,   286,   287,   288,   289,   290,     0,   291,
   292,   293,   294,   295,   296,     0,     0,     0,     0,     0,
     0,   283,   284,   285,   286,   287,   288,   289,   290,   446,
   291,   292,   293,   294,   295,   296,     0,     0,     0,     0,
     0,     0,   283,   284,   285,   286,   287,   288,   289,   290,
   447,   291,   292,   293,   294,   295,   296,     0,     0,     0,
     0,     0,   283,   284,   285,   286,   287,   288,   289,   290,
   480,   291,   292,   293,   294,   295,   296,     0,   283,   284,
   285,   286,   287,   288,   289,   290,   386,   291,   292,   293,
   294,   295,   296,     0,   283,   284,   285,   286,   287,   288,
   289,   290,   448,   291,   292,   293,   294,   295,   296,     0,
   283,   284,   285,   286,   287,   288,   289,   290,   449,   291,
   292,   293,   294,   295,   296,     0,   283,   284,   285,   286,
   287,   288,   289,   290,   450,   291,   292,   293,   294,   295,
   296,     0,   283,   284,   285,   286,   287,   288,   289,   290,
   451,   291,   292,   293,   294,   295,   296,     0,   283,   284,
   285,   286,   287,   288,   289,   290,   452,   291,   292,   293,
   294,   295,   296,     0,   283,   284,   285,   286,   287,   288,
   289,   290,   503,   291,   292,   293,   294,   295,   296,     0,
   283,   284,   285,   286,   287,   288,   289,   290,   504,   291,
   292,   293,   294,   295,   296
};

static const short yycheck[] = {     9,
    38,   345,   267,   207,    58,    58,    77,   105,    58,    58,
    58,    61,   218,    58,    58,   221,    69,    70,    58,    58,
    30,    58,   102,    82,    94,    60,    58,    58,    96,    10,
    98,    12,    13,    14,   105,   105,    58,    96,    19,    98,
    58,    81,    58,    94,    25,    96,    58,    98,    73,    58,
    62,   105,   106,    34,    94,   105,   105,   105,    39,    97,
   105,   105,   102,   103,   408,   105,   105,    79,   105,    81,
     3,     4,     5,   105,   105,    87,    88,    89,    90,    91,
    92,    93,    94,   105,    17,    18,   430,   105,    58,   105,
   102,   103,   436,   105,   438,    94,   105,    58,    58,    58,
    94,   105,   106,    36,    37,   436,    94,   438,   102,   103,
    58,   105,   102,   103,    58,    58,    94,    58,    58,    69,
    70,    94,    62,   133,   134,    94,   136,    58,    58,    84,
    85,    94,    61,    98,    73,   105,   340,   341,    94,    79,
    94,    81,   486,    75,   105,   105,   105,    87,    88,    89,
    90,    91,    92,    93,    94,   165,   137,   105,   102,   102,
   141,   102,   102,   103,   145,   105,   176,    73,    81,   102,
   103,   102,   102,    58,   184,    82,   105,    62,   111,   103,
    73,    94,   163,    76,    98,    78,    98,    80,    81,   102,
   103,   201,   105,    95,    79,   106,    81,   207,    95,   209,
    73,   239,    87,    88,    89,    90,    91,    92,    93,    94,
   143,    52,    53,    54,    55,    56,    57,   102,   103,   100,
   105,    73,    74,    75,    76,    77,    78,    73,    80,    81,
    82,    83,    84,    85,    58,    73,    73,    73,    62,    97,
    80,    81,    82,    83,    84,    85,   227,    63,    64,    99,
    66,    67,   233,   518,   519,    79,    85,    81,    80,    94,
    99,   299,   243,    87,    88,    89,    90,    91,    92,    93,
    94,    82,    83,    84,    85,    82,    95,   258,   102,   103,
    95,   105,    95,    95,     0,    99,     0,   268,    46,   265,
   239,   301,   302,   303,   128,   149,   260,   360,   131,   256,
   251,   311,   145,   154,   163,   243,   174,   317,   492,   319,
   141,   321,   182,   323,   178,   186,   342,    32,   209,   329,
   199,   331,   227,   333,   191,   203,   195,   337,   338,   339,
   340,   341,   137,   348,   223,   430,   486,   270,   111,   111,
   273,   274,   275,   276,   277,   278,   279,   241,    -1,    -1,
   283,   284,   285,   286,   287,   288,   289,   290,   291,   292,
   293,   294,   295,   296,   374,    -1,    -1,   348,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,   309,    -1,    -1,    -1,
    -1,    -1,   315,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
   371,    -1,    -1,    -1,    -1,   405,    -1,    -1,    -1,    -1,
   410,    -1,    -1,    -1,   414,   415,   416,   417,    -1,    -1,
    -1,   421,   422,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,     2,    -1,   406,    -1,     6,    -1,    -1,
   411,    -1,    11,    -1,   367,    -1,    15,    16,    -1,    -1,
    -1,    -1,    21,    22,    23,    24,    -1,    26,    27,    28,
    29,    30,   462,    -1,    33,    -1,    35,   467,    -1,    38,
   470,    40,    41,    -1,    43,    44,    45,    -1,    -1,    -1,
   480,   481,    51,    -1,   455,    -1,    -1,    -1,    -1,   412,
    -1,    -1,   463,    -1,   465,   466,    -1,   468,   469,    -1,
   500,   501,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    58,    -1,   515,   516,    62,   489,   490,
    -1,   492,   493,   446,   447,   496,    -1,    -1,    97,    -1,
    -1,    -1,    -1,    -1,    79,    -1,    81,    -1,    -1,   510,
    -1,    -1,    87,    88,    89,    90,    91,    92,    93,    94,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,   103,   128,
   105,    -1,   131,    58,    -1,    -1,    -1,    62,    -1,    -1,
   139,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   147,    -1,
   149,    -1,   151,    -1,    79,   154,    81,   156,    -1,    -1,
    -1,    -1,    87,    88,    89,    90,    91,    92,    93,    94,
    -1,    -1,    -1,    -1,    -1,   174,    -1,   102,   103,   178,
   105,   180,    58,   182,    -1,    -1,    62,   186,    -1,   188,
    -1,    -1,   191,    -1,    -1,    -1,   195,    -1,   197,    -1,
   199,    -1,    -1,    79,   203,    81,   205,    -1,   207,   208,
   209,    87,    88,    89,    90,    91,    92,    93,    94,    -1,
    -1,    -1,    -1,    -1,   223,    -1,   102,   103,    -1,   105,
   229,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
   239,    -1,    -1,    62,    -1,    -1,   245,    -1,   247,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,   256,    -1,    -1,
    79,   260,    81,   262,    -1,    -1,   265,    -1,    87,    88,
    89,    90,    91,    92,    93,    94,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,   102,   103,    -1,   105,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   297,   298,
   299,    -1,    -1,    -1,    -1,    -1,   305,    -1,    -1,    -1,
    -1,    -1,    -1,   312,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,   325,    -1,   327,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,   340,   341,   342,    -1,    -1,   345,    -1,    -1,    -1,
    -1,   350,    71,    72,    73,    74,    75,    76,    77,    78,
    -1,    80,    81,    82,    83,    84,    85,    -1,    -1,   368,
    -1,    -1,    -1,    -1,     3,     4,     5,     6,     7,     8,
    -1,   100,    11,    12,    13,    14,    15,    16,    17,    18,
    19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
    29,    30,    31,    32,    33,    34,    35,    36,    -1,   408,
    39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
    49,    50,    51,    -1,   423,    -1,    -1,    -1,    -1,    -1,
    -1,   430,    -1,    -1,    -1,    -1,    -1,   436,   437,   438,
   439,    -1,    -1,    -1,    -1,    -1,   445,    71,    72,    73,
    74,    75,    76,    77,    78,    -1,    80,    81,    82,    83,
    84,    85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,   474,   475,    -1,    -1,    -1,
    -1,   105,   106,    -1,    -1,    -1,    -1,   486,   487,   488,
    71,    72,    73,    74,    75,    76,    77,    78,    -1,    80,
    81,    82,    83,    84,    85,    -1,    -1,    -1,    -1,    -1,
    -1,    71,    72,    73,    74,    75,    76,    77,    78,   100,
    80,    81,    82,    83,    84,    85,    -1,    -1,    -1,    -1,
    -1,    -1,    71,    72,    73,    74,    75,    76,    77,    78,
   100,    80,    81,    82,    83,    84,    85,    -1,    -1,    -1,
    -1,    -1,    71,    72,    73,    74,    75,    76,    77,    78,
    99,    80,    81,    82,    83,    84,    85,    -1,    71,    72,
    73,    74,    75,    76,    77,    78,    95,    80,    81,    82,
    83,    84,    85,    -1,    71,    72,    73,    74,    75,    76,
    77,    78,    95,    80,    81,    82,    83,    84,    85,    -1,
    71,    72,    73,    74,    75,    76,    77,    78,    95,    80,
    81,    82,    83,    84,    85,    -1,    71,    72,    73,    74,
    75,    76,    77,    78,    95,    80,    81,    82,    83,    84,
    85,    -1,    71,    72,    73,    74,    75,    76,    77,    78,
    95,    80,    81,    82,    83,    84,    85,    -1,    71,    72,
    73,    74,    75,    76,    77,    78,    95,    80,    81,    82,
    83,    84,    85,    -1,    71,    72,    73,    74,    75,    76,
    77,    78,    95,    80,    81,    82,    83,    84,    85,    -1,
    71,    72,    73,    74,    75,    76,    77,    78,    95,    80,
    81,    82,    83,    84,    85
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

#if YY_patBisonSpec_USE_GOTO != 0
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

#ifdef YY_patBisonSpec_LSP_NEEDED
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
#define yyclearin       (YY_patBisonSpec_CHAR = YYEMPTY)
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
  if (YY_patBisonSpec_CHAR == YYEMPTY && yylen == 1)                               \
    { YY_patBisonSpec_CHAR = (token), YY_patBisonSpec_LVAL = (value);                 \
      yychar1 = YYTRANSLATE (YY_patBisonSpec_CHAR);                                \
      YYPOPSTACK;                                               \
      YYGOTO(yybackup);                                            \
    }                                                           \
  else                                                          \
    { YY_patBisonSpec_ERROR ("syntax error: cannot back up"); YYERROR; }   \
while (0)

#define YYTERROR        1
#define YYERRCODE       256

#ifndef YY_patBisonSpec_PURE
/* UNPURE */
#define YYLEX           YY_patBisonSpec_LEX()
#ifndef YY_USE_CLASS
/* If nonreentrant, and not class , generate the variables here */
int     YY_patBisonSpec_CHAR;                      /*  the lookahead symbol        */
YY_patBisonSpec_STYPE      YY_patBisonSpec_LVAL;              /*  the semantic value of the */
				/*  lookahead symbol    */
int YY_patBisonSpec_NERRS;                 /*  number of parse errors so far */
#ifdef YY_patBisonSpec_LSP_NEEDED
YY_patBisonSpec_LTYPE YY_patBisonSpec_LLOC;   /*  location data for the lookahead     */
			/*  symbol                              */
#endif
#endif


#else
/* PURE */
#ifdef YY_patBisonSpec_LSP_NEEDED
#define YYLEX           YY_patBisonSpec_LEX(&YY_patBisonSpec_LVAL, &YY_patBisonSpec_LLOC)
#else
#define YYLEX           YY_patBisonSpec_LEX(&YY_patBisonSpec_LVAL)
#endif
#endif
#ifndef YY_USE_CLASS
#if YY_patBisonSpec_DEBUG != 0
int YY_patBisonSpec_DEBUG_FLAG;                    /*  nonzero means print parse trace     */
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
  register char *f = from;
  register char *t = to;
  register int i = count;

  while (i-- > 0)
    *t++ = *f++;
}
#endif

int
#ifdef YY_USE_CLASS
 YY_patBisonSpec_CLASS::
#endif
     YY_patBisonSpec_PARSE(YY_patBisonSpec_PARSE_PARAM)
#ifndef __STDC__
#ifndef __cplusplus
#ifndef YY_USE_CLASS
/* parameter definition without protypes */
YY_patBisonSpec_PARSE_PARAM_DEF
#endif
#endif
#endif
{
  register int yystate;
  register int yyn;
  register short *yyssp;
  register YY_patBisonSpec_STYPE *yyvsp;
  int yyerrstatus;      /*  number of tokens to shift before error messages enabled */
  int yychar1=0;          /*  lookahead token as an internal (translated) token number */

  short yyssa[YYINITDEPTH];     /*  the state stack                     */
  YY_patBisonSpec_STYPE yyvsa[YYINITDEPTH];        /*  the semantic value stack            */

  short *yyss = yyssa;          /*  refer to the stacks thru separate pointers */
  YY_patBisonSpec_STYPE *yyvs = yyvsa;     /*  to allow yyoverflow to reallocate them elsewhere */

#ifdef YY_patBisonSpec_LSP_NEEDED
  YY_patBisonSpec_LTYPE yylsa[YYINITDEPTH];        /*  the location stack                  */
  YY_patBisonSpec_LTYPE *yyls = yylsa;
  YY_patBisonSpec_LTYPE *yylsp;

#define YYPOPSTACK   (yyvsp--, yyssp--, yylsp--)
#else
#define YYPOPSTACK   (yyvsp--, yyssp--)
#endif

  int yystacksize = YYINITDEPTH;

#ifdef YY_patBisonSpec_PURE
  int YY_patBisonSpec_CHAR;
  YY_patBisonSpec_STYPE YY_patBisonSpec_LVAL;
  int YY_patBisonSpec_NERRS;
#ifdef YY_patBisonSpec_LSP_NEEDED
  YY_patBisonSpec_LTYPE YY_patBisonSpec_LLOC;
#endif
#endif

  YY_patBisonSpec_STYPE yyval;             /*  the variable used to return         */
				/*  semantic values from the action     */
				/*  routines                            */

  int yylen;
/* start loop, in which YYGOTO may be used. */
YYBEGINGOTO

#if YY_patBisonSpec_DEBUG != 0
  if (YY_patBisonSpec_DEBUG_FLAG)
    fprintf(stderr, "Starting parse\n");
#endif
  yystate = 0;
  yyerrstatus = 0;
  YY_patBisonSpec_NERRS = 0;
  YY_patBisonSpec_CHAR = YYEMPTY;          /* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss - 1;
  yyvsp = yyvs;
#ifdef YY_patBisonSpec_LSP_NEEDED
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
      YY_patBisonSpec_STYPE *yyvs1 = yyvs;
      short *yyss1 = yyss;
#ifdef YY_patBisonSpec_LSP_NEEDED
      YY_patBisonSpec_LTYPE *yyls1 = yyls;
#endif

      /* Get the current used size of the three stacks, in elements.  */
      int size = yyssp - yyss + 1;

#ifdef yyoverflow
      /* Each stack pointer address is followed by the size of
	 the data in use in that stack, in bytes.  */
#ifdef YY_patBisonSpec_LSP_NEEDED
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
#ifdef YY_patBisonSpec_LSP_NEEDED
      yyls = yyls1;
#endif
#else /* no yyoverflow */
      /* Extend the stack our own way.  */
      if (yystacksize >= YYMAXDEPTH)
	{
	  YY_patBisonSpec_ERROR("parser stack overflow");
	  __ALLOCA_return(2);
	}
      yystacksize *= 2;
      if (yystacksize > YYMAXDEPTH)
	yystacksize = YYMAXDEPTH;
      yyss = (short *) __ALLOCA_alloca (yystacksize * sizeof (*yyssp));
      __yy_bcopy ((char *)yyss1, (char *)yyss, size * sizeof (*yyssp));
      __ALLOCA_free(yyss1,yyssa);
      yyvs = (YY_patBisonSpec_STYPE *) __ALLOCA_alloca (yystacksize * sizeof (*yyvsp));
      __yy_bcopy ((char *)yyvs1, (char *)yyvs, size * sizeof (*yyvsp));
      __ALLOCA_free(yyvs1,yyvsa);
#ifdef YY_patBisonSpec_LSP_NEEDED
      yyls = (YY_patBisonSpec_LTYPE *) __ALLOCA_alloca (yystacksize * sizeof (*yylsp));
      __yy_bcopy ((char *)yyls1, (char *)yyls, size * sizeof (*yylsp));
      __ALLOCA_free(yyls1,yylsa);
#endif
#endif /* no yyoverflow */

      yyssp = yyss + size - 1;
      yyvsp = yyvs + size - 1;
#ifdef YY_patBisonSpec_LSP_NEEDED
      yylsp = yyls + size - 1;
#endif

#if YY_patBisonSpec_DEBUG != 0
      if (YY_patBisonSpec_DEBUG_FLAG)
	fprintf(stderr, "Stack size increased to %d\n", yystacksize);
#endif

      if (yyssp >= yyss + yystacksize - 1)
	YYABORT;
    }

#if YY_patBisonSpec_DEBUG != 0
  if (YY_patBisonSpec_DEBUG_FLAG)
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

  if (YY_patBisonSpec_CHAR == YYEMPTY)
    {
#if YY_patBisonSpec_DEBUG != 0
      if (YY_patBisonSpec_DEBUG_FLAG)
	fprintf(stderr, "Reading a token: ");
#endif
      YY_patBisonSpec_CHAR = YYLEX;
    }

  /* Convert token to internal form (in yychar1) for indexing tables with */

  if (YY_patBisonSpec_CHAR <= 0)           /* This means end of input. */
    {
      yychar1 = 0;
      YY_patBisonSpec_CHAR = YYEOF;                /* Don't call YYLEX any more */

#if YY_patBisonSpec_DEBUG != 0
      if (YY_patBisonSpec_DEBUG_FLAG)
	fprintf(stderr, "Now at end of input.\n");
#endif
    }
  else
    {
      yychar1 = YYTRANSLATE(YY_patBisonSpec_CHAR);

#if YY_patBisonSpec_DEBUG != 0
      if (YY_patBisonSpec_DEBUG_FLAG)
	{
	  fprintf (stderr, "Next token is %d (%s", YY_patBisonSpec_CHAR, yytname[yychar1]);
	  /* Give the individual parser a way to print the precise meaning
	     of a token, for further debugging info.  */
#ifdef YYPRINT
	  YYPRINT (stderr, YY_patBisonSpec_CHAR, YY_patBisonSpec_LVAL);
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

#if YY_patBisonSpec_DEBUG != 0
  if (YY_patBisonSpec_DEBUG_FLAG)
    fprintf(stderr, "Shifting token %d (%s), ", YY_patBisonSpec_CHAR, yytname[yychar1]);
#endif

  /* Discard the token being shifted unless it is eof.  */
  if (YY_patBisonSpec_CHAR != YYEOF)
    YY_patBisonSpec_CHAR = YYEMPTY;

  *++yyvsp = YY_patBisonSpec_LVAL;
#ifdef YY_patBisonSpec_LSP_NEEDED
  *++yylsp = YY_patBisonSpec_LLOC;
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

#if YY_patBisonSpec_DEBUG != 0
  if (YY_patBisonSpec_DEBUG_FLAG)
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
#line 2168 "patSpecParser.yy.tab.c"

  switch (yyn) {

case 1:
#line 309 "patSpecParser.yy"
{
               DEBUG_MESSAGE("Finished parsing  <"
	       << scanner.filename() << ">");
;
    break;}
case 49:
#line 369 "patSpecParser.yy"
{
  DETAILED_MESSAGE("Section [DataFile] is now obsolete") ;
;
    break;}
case 50:
#line 373 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  pModel->setModelDescription(yyvsp[0].liststringtype) ;
;
    break;}
case 52:
#line 381 "patSpecParser.yy"
{

;
    break;}
case 55:
#line 387 "patSpecParser.yy"
{
  
;
    break;}
case 56:
#line 390 "patSpecParser.yy"
{

;
    break;}
case 57:
#line 394 "patSpecParser.yy"
{
  pModel->addLatexName(patString(*yyvsp[-1].stype),patString(*yyvsp[0].stype)) ;
;
    break;}
case 58:
#line 399 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  pModel->setChoice(yyvsp[0].arithType) ;
;
    break;}
case 60:
#line 408 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  pModel->setAggregateLast(yyvsp[0].arithType) ;
;
    break;}
case 62:
#line 417 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  pModel->setAggregateWeight(yyvsp[0].arithType) ;
;
    break;}
case 63:
#line 425 "patSpecParser.yy"
{

;
    break;}
case 64:
#line 428 "patSpecParser.yy"
{

;
    break;}
case 65:
#line 432 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[-1].arithType != NULL) ;
  pModel->setPanel(yyvsp[-1].arithType) ;
  pModel->setPanelVariables(yyvsp[0].liststringtype) ;
  DELETE_PTR(yyvsp[0].liststringtype) ;
;
    break;}
case 66:
#line 441 "patSpecParser.yy"
{
  WARNING("Empty list of names") ;
  yyval.liststringtype = NULL ;
;
    break;}
case 67:
#line 445 "patSpecParser.yy"
{
  yyval.liststringtype = new list<patString> ;
  yyval.liststringtype->push_back(*yyvsp[0].stype) ;
;
    break;}
case 68:
#line 449 "patSpecParser.yy"
{
  assert(yyvsp[-1].liststringtype != NULL) ;
  yyvsp[-1].liststringtype->push_back(*yyvsp[0].stype) ;
  yyval.liststringtype = yyvsp[-1].liststringtype ;
;
    break;}
case 73:
#line 462 "patSpecParser.yy"
{
  pModel->addMassAtZero(*yyvsp[-1].stype,yyvsp[0].ftype) ;
;
    break;}
case 74:
#line 468 "patSpecParser.yy"
{
  DEBUG_MESSAGE("No weight defined") ;
;
    break;}
case 75:
#line 472 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  pModel->setWeight(yyvsp[0].arithType) ;
;
    break;}
case 79:
#line 483 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[-4].stype != NULL) ;
  pModel->addBeta(patString(*yyvsp[-4].stype),yyvsp[-3].ftype,yyvsp[-2].ftype,yyvsp[-1].ftype,yyvsp[0].itype!=0) ;
  delete yyvsp[-4].stype ;
;
    break;}
case 80:
#line 492 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  pModel->setMu(yyvsp[-3].ftype,yyvsp[-2].ftype,yyvsp[-1].ftype,yyvsp[0].itype!=0) ;
;
    break;}
case 81:
#line 498 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[-2].stype != NULL) ;
  pModel->setGnuplot(patString(*yyvsp[-2].stype),yyvsp[-1].ftype,yyvsp[0].ftype) ;
  DELETE_PTR(yyvsp[-2].stype) ;
;
    break;}
case 82:
#line 505 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  pModel->setSampleEnumeration(yyvsp[0].itype) ;
;
    break;}
case 87:
#line 517 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[-5].stype != NULL) ;
  assert(yyvsp[-4].stype != NULL) ;
  pModel->addCovarParam(patString(*yyvsp[-5].stype),patString(*yyvsp[-4].stype),yyvsp[-3].ftype,yyvsp[-2].ftype,yyvsp[-1].ftype,yyvsp[0].itype!=0) ;
  delete yyvsp[-5].stype ;
  delete yyvsp[-4].stype ;
;
    break;}
case 92:
#line 532 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[-1].stype != NULL) ;
  pModel->addIIATest(patString(*yyvsp[-1].stype),yyvsp[0].listshorttype) ;
;
    break;}
case 97:
#line 544 "patSpecParser.yy"
{
  assert(yyvsp[-2].stype != NULL) ;
  assert(yyvsp[-1].stype != NULL) ;
  patError* err(NULL) ;
  pModel->addProbaStandardError(patString(*yyvsp[-2].stype),patString(*yyvsp[-1].stype),yyvsp[0].ftype,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    yyerror("Parsing error") ;
  }
;
    break;}
case 102:
#line 562 "patSpecParser.yy"
{
   DEBUG_MESSAGE("Proba " << *yyvsp[-5].stype) ;
   DEBUG_MESSAGE("Bandwith = " << yyvsp[-3].ftype) ;
   DEBUG_MESSAGE("lb = " << yyvsp[-2].ftype) ;
   DEBUG_MESSAGE("ub = " << yyvsp[-1].ftype) ;
   DEBUG_MESSAGE("Name = " << *yyvsp[0].stype) ;

   // Probability
   patOneZhengFosgerau aZheng(yyvsp[-3].ftype,yyvsp[-2].ftype,yyvsp[-1].ftype,patString(*yyvsp[-5].stype),patString(*yyvsp[0].stype)) ;
   patError* err(NULL) ;
   pModel->addZhengFosgerau(aZheng,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    yyerror("Parsing error") ;
  }
 ;
    break;}
case 103:
#line 578 "patSpecParser.yy"
{
   DEBUG_MESSAGE("Expression " << *yyvsp[-5].arithType) ;
   DEBUG_MESSAGE("Bandwith = " << yyvsp[-3].ftype) ;
   DEBUG_MESSAGE("lb = " << yyvsp[-2].ftype) ;
   DEBUG_MESSAGE("ub = " << yyvsp[-1].ftype) ;
   DEBUG_MESSAGE("Name = " << *yyvsp[0].stype) ;
   patOneZhengFosgerau aZheng(yyvsp[-3].ftype,yyvsp[-2].ftype,yyvsp[-1].ftype,patString(*yyvsp[0].stype),yyvsp[-5].arithType) ;
   patError* err(NULL) ;
   pModel->addZhengFosgerau(aZheng,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    yyerror("Parsing error") ;
  }
 ;
    break;}
case 104:
#line 594 "patSpecParser.yy"
{
  pModel->setGeneralizedExtremeValueParameter(*yyvsp[0].stype) ;
;
    break;}
case 106:
#line 599 "patSpecParser.yy"
{
  pModel->setStartingTime(patString(*yyvsp[-1].stype)) ;
  pModel->setDurationParameter(patString(*yyvsp[0].stype)) ;
  DEBUG_MESSAGE("Starting time: " << *yyvsp[-1].stype) ;
  DEBUG_MESSAGE("Model parameter: " << *yyvsp[0].stype) ;
;
    break;}
case 107:
#line 606 "patSpecParser.yy"
{
  pModel->setRegressionObservation(patString(*yyvsp[-1].stype)) ;
  DEBUG_MESSAGE("Regression dependent: " << *yyvsp[-1].stype) ;
;
    break;}
case 114:
#line 615 "patSpecParser.yy"
{

  assert(yyvsp[0].uftype != NULL) ;
  pModel->addAcqRegressionModel(yyvsp[0].uftype) ;
  DELETE_PTR(yyvsp[0].uftype) ;
;
    break;}
case 115:
#line 622 "patSpecParser.yy"
{
  pModel->setAcqSigma(patString(*yyvsp[0].stype)) ;
;
    break;}
case 116:
#line 628 "patSpecParser.yy"
{
  pModel->addValRegressionModel(yyvsp[0].uftype) ;
  DELETE_PTR(yyvsp[0].uftype) ;
;
    break;}
case 117:
#line 633 "patSpecParser.yy"
{
  pModel->setValSigma(patString(*yyvsp[0].stype)) ;
;
    break;}
case 122:
#line 643 "patSpecParser.yy"
{
  pModel->setOrdinalLogitLeftAlternative(yyvsp[-1].itype) ;
;
    break;}
case 123:
#line 646 "patSpecParser.yy"
{
  pModel->addOrdinalLogitThreshold(yyvsp[-1].itype,patString(*yyvsp[0].stype)) ;
;
    break;}
case 128:
#line 658 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  pModel->addNonLinearUtility(yyvsp[-1].itype,yyvsp[0].arithType) ;
;
    break;}
case 133:
#line 670 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[-1].stype != NULL) ;
  pModel->addDerivative(yyvsp[-2].itype,patString(*yyvsp[-1].stype),yyvsp[0].arithType) ;
;
    break;}
case 138:
#line 686 "patSpecParser.yy"
{
  assert(pModel!= NULL) ;
  assert(yyvsp[-2].stype != NULL) ;
  patError* err(NULL) ;
  pModel->addExpression(patString(*yyvsp[-2].stype),yyvsp[0].arithType,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    yyerror("Parsing error") ;
  }
  delete yyvsp[-2].stype ;
;
    break;}
case 139:
#line 697 "patSpecParser.yy"
{
  assert(pModel!= NULL) ;
  assert(yyvsp[-2].stype != NULL) ;
  patError* err(NULL) ;
  pModel->addExpressionLoop(patString(*yyvsp[-2].stype),yyvsp[0].arithType,yyvsp[-3].loopType,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    yyerror("Parsing error") ;
  }

  delete yyvsp[-2].stype ;
;
    break;}
case 140:
#line 710 "patSpecParser.yy"
{

  patLoop* theLoop = new patLoop ;
  theLoop->variable = patString(*yyvsp[-4].stype) ;
  theLoop->lower = yyvsp[-3].itype ; 
  theLoop->upper = yyvsp[-2].itype ; 
  theLoop->step = yyvsp[-1].itype ; 
  yyval.loopType = theLoop ;
;
    break;}
case 141:
#line 722 "patSpecParser.yy"
{
  DEBUG_MESSAGE("No group defined") ;
  assert (pModel != NULL) ;
  pModel->setDefaultGroup() ;
//   patArithConstant* ptr = new patArithConstant(NULL) ;
//   ptr->setValue(1) ;
//   pModel->setGroup(ptr) ;
//   pModel->addScale(1,1.0,1.0,1.0,patTRUE) ;
;
    break;}
case 142:
#line 731 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  pModel->setGroup(yyvsp[0].arithType) ;
;
    break;}
case 143:
#line 738 "patSpecParser.yy"
{
  DEBUG_MESSAGE("No exclusion condition") ;
  patArithConstant* ptr = new patArithConstant(NULL) ;
  ptr->setValue(0.0) ;
  pModel->setExclude(ptr) ;
;
    break;}
case 144:
#line 744 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  //  DEBUG_MESSAGE("Exclude condition " << *$2) ;
  pModel->setExclude(yyvsp[0].arithType) ;
;
    break;}
case 150:
#line 756 "patSpecParser.yy"
{
  assert (pModel != NULL) ;
  pModel->addScale(yyvsp[-4].itype,yyvsp[-3].ftype,yyvsp[-2].ftype,yyvsp[-1].ftype,yyvsp[0].itype!=0) ;
;
    break;}
case 152:
#line 764 "patSpecParser.yy"
{
  DEBUG_MESSAGE("Model OL") ;
  assert (pModel != NULL) ;
  pModel->setModelType(patModelSpec::patOLtype) ;
;
    break;}
case 153:
#line 769 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  pModel->setModelType(patModelSpec::patBPtype) ;
;
    break;}
case 154:
#line 773 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  pModel->setModelType(patModelSpec::patMNLtype) ;
;
    break;}
case 155:
#line 777 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  pModel->setModelType(patModelSpec::patNLtype) ;
;
    break;}
case 156:
#line 781 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  pModel->setModelType(patModelSpec::patCNLtype) ;
;
    break;}
case 157:
#line 785 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  pModel->setModelType(patModelSpec::patNetworkGEVtype) ;
;
    break;}
case 158:
#line 792 "patSpecParser.yy"
{
;
    break;}
case 162:
#line 798 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[-5].stype != NULL) ;
  assert(yyvsp[0].listshorttype != NULL) ;
  pModel->addNest(patString(*yyvsp[-5].stype),yyvsp[-4].ftype,yyvsp[-3].ftype,yyvsp[-2].ftype,yyvsp[-1].itype!=0,yyvsp[0].listshorttype) ;
  delete yyvsp[-5].stype ;
  delete yyvsp[0].listshorttype ;
;
    break;}
case 163:
#line 807 "patSpecParser.yy"
{
  yyval.listshorttype = new list<long> ;
  yyval.listshorttype->push_back(yyvsp[0].itype) ;
;
    break;}
case 164:
#line 811 "patSpecParser.yy"
{
  assert(yyvsp[-1].listshorttype != NULL) ;
  yyvsp[-1].listshorttype->push_back(yyvsp[0].itype) ;
  yyval.listshorttype = yyvsp[-1].listshorttype ;
;
    break;}
case 165:
#line 818 "patSpecParser.yy"
{
  DEBUG_MESSAGE("No nests defined for CNL model") ;
;
    break;}
case 169:
#line 825 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[-4].stype != NULL) ;
  pModel->addCNLNest(patString(*yyvsp[-4].stype),yyvsp[-3].ftype,yyvsp[-2].ftype,yyvsp[-1].ftype,yyvsp[0].itype!=0) ;
  delete yyvsp[-4].stype ;
;
    break;}
case 170:
#line 833 "patSpecParser.yy"
{
  DEBUG_MESSAGE("No alpha defined for CNL model") ;
;
    break;}
case 174:
#line 840 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[-5].stype != NULL) ;
  assert(yyvsp[-4].stype != NULL) ;
  pModel->addCNLAlpha(patString(*yyvsp[-5].stype),patString(*yyvsp[-4].stype),yyvsp[-3].ftype,yyvsp[-2].ftype,yyvsp[-1].ftype,yyvsp[0].itype!=0) ;
  delete yyvsp[-5].stype ;
  delete yyvsp[-4].stype ;
  static int count = 0 ;
  ++count ;
;
    break;}
case 179:
#line 856 "patSpecParser.yy"
{
  assert (pModel != NULL) ;
  assert (yyvsp[-2].stype != NULL) ;
  assert (yyvsp[-1].stype != NULL) ;
  assert (yyvsp[0].stype != NULL) ;
  pModel->addRatio(patString(*yyvsp[-2].stype),patString(*yyvsp[-1].stype),patString(*yyvsp[0].stype)) ;
  delete yyvsp[-2].stype ;
  delete yyvsp[-1].stype ;
  delete yyvsp[0].stype ;
;
    break;}
case 180:
#line 867 "patSpecParser.yy"
{
  pModel->setNumberOfDraws(yyvsp[0].itype) ;
;
    break;}
case 181:
#line 871 "patSpecParser.yy"
{
  pModel->setListLinearConstraints(yyvsp[0].llctype) ;
  DELETE_PTR(yyvsp[0].llctype) ;
;
    break;}
case 182:
#line 877 "patSpecParser.yy"
{
  pModel->setListNonLinearEqualityConstraints(yyvsp[0].lnlctype) ;
;
    break;}
case 183:
#line 882 "patSpecParser.yy"
{
  pModel->setListNonLinearInequalityConstraints(yyvsp[0].lnlctype) ;
;
    break;}
case 184:
#line 887 "patSpecParser.yy"
{
  patListNonLinearConstraints* ptr = NULL ;
  yyval.lnlctype = ptr ;
;
    break;}
case 185:
#line 891 "patSpecParser.yy"
{
  patListNonLinearConstraints* ptr = new patListNonLinearConstraints;
  ptr->push_back(*yyvsp[0].nlctype) ;
  DELETE_PTR(yyvsp[0].nlctype) ;
  yyval.lnlctype = ptr ;
;
    break;}
case 186:
#line 897 "patSpecParser.yy"
{
  yyvsp[-1].lnlctype->push_back(*yyvsp[0].nlctype) ;
  DELETE_PTR(yyvsp[0].nlctype) ;
  yyval.lnlctype = yyvsp[-1].lnlctype ;
;
    break;}
case 187:
#line 903 "patSpecParser.yy"
{
  patNonLinearConstraint* ptr = new patNonLinearConstraint(yyvsp[0].arithType) ;
  yyval.nlctype = ptr ;
;
    break;}
case 188:
#line 908 "patSpecParser.yy"
{
  patListLinearConstraint* ptr = NULL ;
  yyval.llctype = ptr ;
;
    break;}
case 189:
#line 912 "patSpecParser.yy"
{
  patListLinearConstraint* ptr = new patListLinearConstraint ;
  ptr->push_back(*yyvsp[0].lctype) ;
  DELETE_PTR(yyvsp[0].lctype) ;
  yyval.llctype = ptr ;
;
    break;}
case 190:
#line 918 "patSpecParser.yy"
{
  yyvsp[-1].llctype->push_back(*yyvsp[0].lctype) ;
  DELETE_PTR(yyvsp[0].lctype) ;
  yyval.llctype = yyvsp[-1].llctype ;
;
    break;}
case 191:
#line 925 "patSpecParser.yy"
{  
  patLinearConstraint* ptr = new patLinearConstraint ;
  ptr->theEquation = *yyvsp[-2].cetype ;
  ptr->theType = patLinearConstraint::patLESSEQUAL ;
  ptr->theRHS = yyvsp[0].ftype ;
  DEBUG_MESSAGE(*ptr) ;
  DELETE_PTR(yyvsp[-2].cetype) ;
  yyval.lctype = ptr ;
;
    break;}
case 192:
#line 934 "patSpecParser.yy"
{
  patLinearConstraint* ptr = new patLinearConstraint ;
  ptr->theEquation = *yyvsp[-2].cetype ;
  ptr->theType = patLinearConstraint::patEQUAL ;
  ptr->theRHS = yyvsp[0].ftype ;
  DELETE_PTR(yyvsp[-2].cetype) ;
  yyval.lctype = ptr ;
;
    break;}
case 193:
#line 942 "patSpecParser.yy"
{
  patLinearConstraint* ptr = new patLinearConstraint ;
  ptr->theEquation = *yyvsp[-2].cetype ;
  ptr->theType = patLinearConstraint::patGREATEQUAL ;
  ptr->theRHS = yyvsp[0].ftype ;
  DELETE_PTR(yyvsp[-2].cetype) ;
  DEBUG_MESSAGE(*ptr) ;
  yyval.lctype = ptr ;
;
    break;}
case 197:
#line 955 "patSpecParser.yy"
{
  patConstraintEquation* ptr = new patConstraintEquation ;
  ptr->push_back(*yyvsp[0].cttype) ;
  DELETE_PTR(yyvsp[0].cttype) ;
  yyval.cetype = ptr ;
;
    break;}
case 198:
#line 961 "patSpecParser.yy"
{
  patConstraintEquation* ptr = new patConstraintEquation ;
  yyvsp[0].cttype->fact = - yyvsp[0].cttype->fact ;
  ptr->push_back(*yyvsp[0].cttype) ;
  DELETE_PTR(yyvsp[0].cttype) ;
  yyval.cetype = ptr ;
;
    break;}
case 199:
#line 969 "patSpecParser.yy"
{
  yyvsp[-2].cetype->push_back(*yyvsp[0].cttype) ;
  DELETE_PTR(yyvsp[0].cttype) ;
  yyval.cetype = yyvsp[-2].cetype ;
;
    break;}
case 200:
#line 974 "patSpecParser.yy"
{
  yyvsp[0].cttype->fact = - yyvsp[0].cttype->fact ;
  yyvsp[-2].cetype->push_back(*yyvsp[0].cttype) ;
  DELETE_PTR(yyvsp[0].cttype) ;
  yyval.cetype = yyvsp[-2].cetype ;
;
    break;}
case 201:
#line 980 "patSpecParser.yy"
{
  patConstraintTerm* ptr = new patConstraintTerm ;
  ptr->fact = 1.0 ;
  ptr->param = *yyvsp[0].stype ;
  DELETE_PTR(yyvsp[0].stype) ;
  yyval.cttype = ptr ;
;
    break;}
case 202:
#line 987 "patSpecParser.yy"
{
  patConstraintTerm* ptr = new patConstraintTerm ;
  ptr->fact = yyvsp[-2].ftype ;
  ptr->param = *yyvsp[0].stype ;
  DELETE_PTR(yyvsp[0].stype) ;
  yyval.cttype = ptr ;
;
    break;}
case 203:
#line 995 "patSpecParser.yy"
{
  yyval.stype = yyvsp[0].stype ;
;
    break;}
case 204:
#line 998 "patSpecParser.yy"
{
  yyval.stype = yyvsp[0].stype ;
;
    break;}
case 205:
#line 1002 "patSpecParser.yy"
{
  patString* ptr = new patString(pModel->buildLinkName(*yyvsp[-3].stype,*yyvsp[-1].stype)) ;
  DELETE_PTR(yyvsp[-3].stype) ;
  DELETE_PTR(yyvsp[-1].stype) ;
  yyval.stype = ptr ;
;
    break;}
case 210:
#line 1012 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[-4].stype != NULL) ;
  pModel->addNetworkGevNode(patString(*yyvsp[-4].stype),yyvsp[-3].ftype,yyvsp[-2].ftype,yyvsp[-1].ftype,yyvsp[0].itype!=0) ;
  delete yyvsp[-4].stype ;
;
    break;}
case 215:
#line 1022 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[-5].stype != NULL) ;
  assert(yyvsp[-4].stype != NULL) ;

  pModel->addNetworkGevLink(patString(*yyvsp[-5].stype),patString(*yyvsp[-4].stype),yyvsp[-3].ftype,yyvsp[-2].ftype,yyvsp[-1].ftype,yyvsp[0].itype!=0) ;
  delete yyvsp[-5].stype ;
  delete yyvsp[-4].stype ;
;
    break;}
case 220:
#line 1037 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[-2].stype != NULL) ;
  assert(yyvsp[-1].stype != NULL) ;
  stringstream str ;
  str << "Syntax error on line " <<  scanner.lineno() << endl ;
  str << "Section [ConstantProduct] is obsolete. " ; 
  str << "Add the following line in section [NonLinearEqualityConstraints] instead: " ;
  str << *yyvsp[-2].stype << "*" << *yyvsp[-1].stype << "-" << yyvsp[0].ftype ;

  
  //pModel->addConstantProduct(patString(*$1),patString(*$2),$3) ;
  delete yyvsp[-2].stype ;
  delete yyvsp[-1].stype ;
  pModel->syntaxError = new patErrMiscError(str.str());
  //  exit(-1) ;

;
    break;}
case 225:
#line 1061 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[-2].stype != NULL) ;
  assert(yyvsp[0].stype != NULL) ;
  DEBUG_MESSAGE("Constraining nest parameters for " << *yyvsp[-2].stype << " and " << *yyvsp[0].stype) ;
  pModel->addConstraintNest(patString(*yyvsp[-2].stype),patString(*yyvsp[0].stype)) ;
  delete yyvsp[-2].stype ;
  delete yyvsp[0].stype ;
;
    break;}
case 229:
#line 1078 "patSpecParser.yy"
{
  assert(yyvsp[-1].stype != NULL) ;
  pModel->setSnpBaseParameter(patString(*yyvsp[-1].stype)) ;
  delete yyvsp[-1].stype ;
;
    break;}
case 232:
#line 1086 "patSpecParser.yy"
{
  assert(yyvsp[0].stype != NULL) ;
  pModel->addSnpTerm(yyvsp[-1].itype,patString(*yyvsp[0].stype)) ;
  delete yyvsp[0].stype ;

;
    break;}
case 237:
#line 1098 "patSpecParser.yy"
{
  assert(yyvsp[0].stype != NULL) ;
  pModel->addSelectionBiasParameter(yyvsp[-1].itype,*yyvsp[0].stype) ;
;
    break;}
case 242:
#line 1107 "patSpecParser.yy"
{
  patError* err(NULL) ;
  pModel->addDiscreteParameter(*yyvsp[-3].stype,*yyvsp[-1].discreteDistType,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    yyerror("Parsing error") ;
  }
  DELETE_PTR(yyvsp[-1].discreteDistType) ;
  DEBUG_MESSAGE("Identified discrete param " << *yyvsp[-3].stype) ;
  
;
    break;}
case 243:
#line 1119 "patSpecParser.yy"
{
  vector<patThreeStrings>* ptr = new vector<patThreeStrings> ;
  ptr->push_back(*yyvsp[0].discreteTermType) ;
  yyval.discreteDistType = ptr ;
;
    break;}
case 244:
#line 1123 "patSpecParser.yy"
{
  assert(yyvsp[-1].discreteDistType != NULL) ;
  yyvsp[-1].discreteDistType->push_back(*yyvsp[0].discreteTermType) ;
  yyval.discreteDistType = yyvsp[-1].discreteDistType ;
;
    break;}
case 245:
#line 1129 "patSpecParser.yy"
{
  yyval.discreteTermType= new patThreeStrings(*yyvsp[-3].stype,patString(),*yyvsp[-1].stype) ;
  DEBUG_MESSAGE("-> discrete term " << *yyvsp[-3].stype << "(" << *yyvsp[-1].stype << ")") ;
;
    break;}
case 246:
#line 1133 "patSpecParser.yy"
{
  yyval.discreteTermType= new patThreeStrings(yyvsp[-3].arithRandomType->getLocationParameter(),
			  yyvsp[-3].arithRandomType->getScaleParameter(),
			  *yyvsp[-1].stype) ;
  DEBUG_MESSAGE("-> discrete term " << yyvsp[-3].arithRandomType->getLocationParameter() << "[" <<  yyvsp[-3].arithRandomType->getScaleParameter() << "]" << "(" << *yyvsp[-1].stype << ")") ;
;
    break;}
case 250:
#line 1146 "patSpecParser.yy"
{
  assert(pModel != NULL) ;
  assert(yyvsp[-2].stype != NULL) ;
  assert(yyvsp[-1].stype != NULL) ;
  assert(yyvsp[0].uftype != NULL) ;
   patError* err(NULL) ;
   pModel->addUtil(yyvsp[-3].itype,patString(*yyvsp[-2].stype),patString(*yyvsp[-1].stype),yyvsp[0].uftype,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    yyerror("Parsing error") ;
  }
  delete yyvsp[-2].stype ;
  delete yyvsp[-1].stype ;
  delete yyvsp[0].uftype ;
;
    break;}
case 251:
#line 1162 "patSpecParser.yy"
{
  yyval.uftype = new patUtilFunction ;
;
    break;}
case 252:
#line 1165 "patSpecParser.yy"
{
  assert ( yyvsp[0].uttype != NULL) ;
  yyval.uftype = new patUtilFunction ;
  yyval.uftype->push_back(*yyvsp[0].uttype) ;
  delete yyvsp[0].uttype ;
;
    break;}
case 253:
#line 1171 "patSpecParser.yy"
{
  
  assert(yyvsp[-2].uftype != NULL) ;
  assert(yyvsp[0].uttype != NULL) ;
  yyvsp[-2].uftype->push_back(*yyvsp[0].uttype) ;
  delete yyvsp[0].uttype ;
  yyval.uftype = yyvsp[-2].uftype ;
;
    break;}
case 254:
#line 1182 "patSpecParser.yy"
{
  assert(yyvsp[-2].stype != NULL) ;
  assert(yyvsp[0].stype != NULL) ;
  patUtilTerm* term = new patUtilTerm ;
  term->beta = patString(*yyvsp[-2].stype) ;
  term->x = patString(*yyvsp[0].stype) ;
  term->random = patFALSE ;
  assert(pModel != NULL) ;
  pModel->addAttribute(*yyvsp[0].stype) ;
  delete yyvsp[-2].stype ;
  delete yyvsp[0].stype ;
  yyval.uttype = term ;
;
    break;}
case 255:
#line 1195 "patSpecParser.yy"
{
  assert(yyvsp[-2].arithRandomType != NULL) ;
  assert(yyvsp[0].stype != NULL) ;
  patUtilTerm* term = new patUtilTerm ;
  term->beta = yyvsp[-2].arithRandomType->getOperatorName() ;
  term->betaIndex = patBadId ;
  term->randomParameter = yyvsp[-2].arithRandomType ;
  term->x = patString(*yyvsp[0].stype) ;
  term->random = patTRUE ;
  assert(pModel != NULL) ;
  pModel->addAttribute(*yyvsp[0].stype) ;
  delete yyvsp[0].stype ;
  yyval.uttype = term ;
;
    break;}
case 256:
#line 1210 "patSpecParser.yy"
{
  yyval.arithType = yyvsp[0].arithType ;
;
    break;}
case 257:
#line 1213 "patSpecParser.yy"
{
  yyval.arithType = yyvsp[0].arithType ;
;
    break;}
case 258:
#line 1216 "patSpecParser.yy"
{
  yyval.arithType = yyvsp[0].arithType ;
;
    break;}
case 259:
#line 1219 "patSpecParser.yy"
{
  yyval.arithType = yyvsp[0].arithRandomType ;
;
    break;}
case 260:
#line 1222 "patSpecParser.yy"
{
  yyval.arithType = yyvsp[-1].arithType ;
;
    break;}
case 261:
#line 1225 "patSpecParser.yy"
{
  yyval.arithType = yyvsp[0].arithType ;
;
    break;}
case 262:
#line 1230 "patSpecParser.yy"
{
  yyval.arithRandomType = yyvsp[0].arithRandomType ;
;
    break;}
case 263:
#line 1233 "patSpecParser.yy"
{
  yyval.arithRandomType = yyvsp[0].arithRandomType ;
;
    break;}
case 264:
#line 1237 "patSpecParser.yy"
{
  assert(yyvsp[-3].arithType != NULL) ;
  assert(yyvsp[-1].stype != NULL) ;
  patArithDeriv* ptr = new patArithDeriv(NULL,yyvsp[-3].arithType,patString(*yyvsp[-1].stype)) ;
  assert(ptr != NULL) ;
  yyvsp[-3].arithType->setParent(ptr) ;
  delete yyvsp[-1].stype ;
  yyval.arithType = ptr ;
;
    break;}
case 265:
#line 1247 "patSpecParser.yy"
{
  //  pModel->addDrawHeader(*$1,*$3);
  patArithNormalRandom* ptr = new patArithNormalRandom(NULL) ;
  assert(ptr != NULL) ;
  ptr->setLocationParameter(*yyvsp[-3].stype) ;
  ptr->setScaleParameter(*yyvsp[-1].stype) ;
  patArithRandom* oldParam = pModel->addRandomExpression(ptr) ;
  delete yyvsp[-3].stype ;
  delete yyvsp[-1].stype ;
  if (oldParam == NULL) {
    yyval.arithRandomType = ptr ;
  }
  else {
    delete ptr ;
    yyval.arithRandomType = oldParam ;
  }
;
    break;}
case 266:
#line 1265 "patSpecParser.yy"
{
  //  pModel->addDrawHeader(*$1,*$3);
  patArithUnifRandom* ptr = new patArithUnifRandom(NULL) ;
  assert(ptr != NULL) ;
  ptr->setLocationParameter(*yyvsp[-3].stype) ;
  ptr->setScaleParameter(*yyvsp[-1].stype) ;
  patArithRandom* oldParam = pModel->addRandomExpression(ptr) ;
  delete yyvsp[-3].stype ;
  delete yyvsp[-1].stype ;
  if (oldParam == NULL) {
    yyval.arithRandomType = ptr ;
  }
  else {
    delete ptr ;
    yyval.arithRandomType = oldParam ;
  }
;
    break;}
case 267:
#line 1285 "patSpecParser.yy"
{
  assert(yyvsp[0].arithType != NULL) ;
  patArithUnaryMinus* ptr = new patArithUnaryMinus(NULL,yyvsp[0].arithType) ;
  assert(ptr != NULL) ;
  yyvsp[0].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;
    
;
    break;}
case 268:
#line 1293 "patSpecParser.yy"
{
  assert(yyvsp[0].arithType != NULL) ;
  patArithNot* ptr = new patArithNot(NULL,yyvsp[0].arithType) ;
  assert(ptr != NULL) ;
  yyvsp[0].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;

;
    break;}
case 269:
#line 1301 "patSpecParser.yy"
{
  assert(yyvsp[-1].arithType != NULL) ;
  patArithSqrt* ptr = new patArithSqrt(NULL,yyvsp[-1].arithType) ;
  assert(ptr != NULL) ;
  yyvsp[-1].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;
;
    break;}
case 270:
#line 1308 "patSpecParser.yy"
{
  assert(yyvsp[-1].arithType != NULL) ;
  patArithLog* ptr = new patArithLog(NULL,yyvsp[-1].arithType) ;
  assert(ptr != NULL) ;
  yyvsp[-1].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;

;
    break;}
case 271:
#line 1316 "patSpecParser.yy"
{
  assert(yyvsp[-1].arithType != NULL) ;
  patArithExp* ptr = new patArithExp(NULL,yyvsp[-1].arithType) ;
  assert(ptr != NULL) ;
  yyvsp[-1].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;
;
    break;}
case 272:
#line 1323 "patSpecParser.yy"
{
  assert(yyvsp[-1].arithType != NULL) ;
  patArithAbs* ptr = new patArithAbs(NULL,yyvsp[-1].arithType) ;
  assert(ptr != NULL) ;
  yyvsp[-1].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;

;
    break;}
case 273:
#line 1331 "patSpecParser.yy"
{
  assert(yyvsp[-1].arithType != NULL) ;
  patArithInt* ptr = new patArithInt(NULL,yyvsp[-1].arithType) ;
  assert(ptr != NULL) ;
  yyvsp[-1].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;
;
    break;}
case 274:
#line 1338 "patSpecParser.yy"
{
  yyval.arithType = yyvsp[-1].arithType;  
;
    break;}
case 275:
#line 1343 "patSpecParser.yy"
{
  assert(yyvsp[-2].arithType != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  patArithBinaryPlus* ptr = new patArithBinaryPlus(NULL,yyvsp[-2].arithType,yyvsp[0].arithType) ;
  assert(ptr != NULL) ;
  yyvsp[-2].arithType->setParent(ptr) ;
  yyvsp[0].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;
;
    break;}
case 276:
#line 1352 "patSpecParser.yy"
{
  patArithBinaryMinus* ptr = new patArithBinaryMinus(NULL,yyvsp[-2].arithType,yyvsp[0].arithType) ;
  assert(ptr != NULL) ;
  assert(yyvsp[-2].arithType != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  yyvsp[-2].arithType->setParent(ptr) ;
  yyvsp[0].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;

;
    break;}
case 277:
#line 1362 "patSpecParser.yy"
{
  patArithMult* ptr = new patArithMult(NULL,yyvsp[-2].arithType,yyvsp[0].arithType) ;
  assert(ptr != NULL) ;
  assert(yyvsp[-2].arithType != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  yyvsp[-2].arithType->setParent(ptr) ;
  yyvsp[0].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;

;
    break;}
case 278:
#line 1372 "patSpecParser.yy"
{
  patArithDivide* ptr = new patArithDivide(NULL,yyvsp[-2].arithType,yyvsp[0].arithType) ;
  assert(ptr != NULL) ;
  assert(yyvsp[-2].arithType != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  yyvsp[-2].arithType->setParent(ptr) ;
  yyvsp[0].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;
;
    break;}
case 279:
#line 1381 "patSpecParser.yy"
{
  patArithPower* ptr = new patArithPower(NULL,yyvsp[-2].arithType,yyvsp[0].arithType) ;
  assert(ptr != NULL) ;
  assert(yyvsp[-2].arithType != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  yyvsp[-2].arithType->setParent(ptr) ;
  yyvsp[0].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;
;
    break;}
case 280:
#line 1390 "patSpecParser.yy"
{
  patArithEqual* ptr = new patArithEqual(NULL,yyvsp[-2].arithType,yyvsp[0].arithType) ;
  assert(ptr != NULL) ;
  assert(yyvsp[-2].arithType != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  yyvsp[-2].arithType->setParent(ptr) ;
  yyvsp[0].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;

;
    break;}
case 281:
#line 1400 "patSpecParser.yy"
{
  patArithNotEqual* ptr = new patArithNotEqual(NULL,yyvsp[-2].arithType,yyvsp[0].arithType) ;
  assert(ptr != NULL) ;
  assert(yyvsp[-2].arithType != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  yyvsp[-2].arithType->setParent(ptr) ;
  yyvsp[0].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;

;
    break;}
case 282:
#line 1410 "patSpecParser.yy"
{
  patArithOr* ptr = new patArithOr(NULL,yyvsp[-2].arithType,yyvsp[0].arithType) ;
  assert(ptr != NULL) ;
  assert(yyvsp[-2].arithType != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  yyvsp[-2].arithType->setParent(ptr) ;
  yyvsp[0].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;

;
    break;}
case 283:
#line 1420 "patSpecParser.yy"
{
  patArithAnd* ptr = new patArithAnd(NULL,yyvsp[-2].arithType,yyvsp[0].arithType) ;
  assert(ptr != NULL) ;
  assert(yyvsp[-2].arithType != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  yyvsp[-2].arithType->setParent(ptr) ;
  yyvsp[0].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;

;
    break;}
case 284:
#line 1430 "patSpecParser.yy"
{
  patArithLess* ptr = new patArithLess(NULL,yyvsp[-2].arithType,yyvsp[0].arithType) ;
  assert(ptr != NULL) ;
  assert(yyvsp[-2].arithType != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  yyvsp[-2].arithType->setParent(ptr) ;
  yyvsp[0].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;

;
    break;}
case 285:
#line 1440 "patSpecParser.yy"
{
  patArithLessEqual* ptr = new patArithLessEqual(NULL,yyvsp[-2].arithType,yyvsp[0].arithType) ;
  assert(ptr != NULL) ;
  assert(yyvsp[-2].arithType != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  yyvsp[-2].arithType->setParent(ptr) ;
  yyvsp[0].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;

;
    break;}
case 286:
#line 1450 "patSpecParser.yy"
{
  patArithGreater* ptr = new patArithGreater(NULL,yyvsp[-2].arithType,yyvsp[0].arithType) ;
  assert(ptr != NULL) ;
  assert(yyvsp[-2].arithType != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  yyvsp[-2].arithType->setParent(ptr) ;
  yyvsp[0].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;

;
    break;}
case 287:
#line 1460 "patSpecParser.yy"
{
  patArithGreaterEqual* ptr = new patArithGreaterEqual(NULL,yyvsp[-2].arithType,yyvsp[0].arithType) ;
  assert(ptr != NULL) ;
  assert(yyvsp[-2].arithType != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  yyvsp[-2].arithType->setParent(ptr) ;
  yyvsp[0].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;

;
    break;}
case 288:
#line 1470 "patSpecParser.yy"
{
  patArithMax* ptr = new patArithMax(NULL,yyvsp[-3].arithType,yyvsp[-1].arithType) ;
  assert(ptr != NULL) ;
  assert(yyvsp[-3].arithType != NULL) ;
  assert(yyvsp[-1].arithType != NULL) ;
  yyvsp[-3].arithType->setParent(ptr) ;
  yyvsp[-1].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;

;
    break;}
case 289:
#line 1480 "patSpecParser.yy"
{
  patArithMin* ptr = new patArithMin(NULL,yyvsp[-3].arithType,yyvsp[-1].arithType) ;
  assert(ptr != NULL) ;
  assert(yyvsp[-3].arithType != NULL) ;
  assert(yyvsp[-1].arithType != NULL) ;
  yyvsp[-3].arithType->setParent(ptr) ;
  yyvsp[-1].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;

;
    break;}
case 290:
#line 1490 "patSpecParser.yy"
{
  patArithMod* ptr = new patArithMod(NULL,yyvsp[-2].arithType,yyvsp[0].arithType) ;
  assert(ptr != NULL) ;
  assert(yyvsp[-2].arithType != NULL) ;
  assert(yyvsp[0].arithType != NULL) ;
  yyvsp[-2].arithType->setParent(ptr) ;
  yyvsp[0].arithType->setParent(ptr) ;
  yyval.arithType = ptr ;
  
;
    break;}
case 291:
#line 1500 "patSpecParser.yy"
{
  yyval.arithType = yyvsp[-1].arithType;  
;
    break;}
case 292:
#line 1504 "patSpecParser.yy"
{
  patArithConstant* ptr = new patArithConstant(NULL) ;
  assert(ptr != NULL) ;
  ptr->setValue(yyvsp[0].ftype) ;
  yyval.arithType= ptr ;
;
    break;}
case 293:
#line 1510 "patSpecParser.yy"
{
  assert(yyvsp[0].stype != NULL);
  patArithVariable* ptr = new patArithVariable(NULL) ;
  assert(ptr != NULL) ;
  ptr->setName(*yyvsp[0].stype) ;
  delete yyvsp[0].stype ;
  yyval.arithType = ptr ;
;
    break;}
case 294:
#line 1519 "patSpecParser.yy"
{
  yyval.ftype = yyvsp[0].ftype ;
;
    break;}
case 295:
#line 1522 "patSpecParser.yy"
{
  yyval.ftype = float(yyvsp[0].itype) ;
;
    break;}
case 296:
#line 1526 "patSpecParser.yy"
{
  yyval.stype = yyvsp[0].stype ;
;
    break;}
case 297:
#line 1529 "patSpecParser.yy"
{
  yyval.stype = yyvsp[0].stype ;
;
    break;}
case 298:
#line 1533 "patSpecParser.yy"
{
  patString* str = new patString((scanner.removeDelimeters()));
  yyval.stype = str ;
;
    break;}
case 299:
#line 1538 "patSpecParser.yy"
{
  patString* str = new patString(scanner.value());
  //Remove the last character which is [ \t\n]
  str->erase(str->end()-1) ;
  yyval.stype = str ;
;
    break;}
case 300:
#line 1545 "patSpecParser.yy"
{
  yyval.ftype = atof( scanner.value().c_str() );
;
    break;}
case 301:
#line 1550 "patSpecParser.yy"
{
  yyval.itype = atoi( scanner.value().c_str() );
;
    break;}
}

#line 811 "/usr/local/lib/bison.cc"
   /* the action file gets copied in in place of this dollarsign  */
  yyvsp -= yylen;
  yyssp -= yylen;
#ifdef YY_patBisonSpec_LSP_NEEDED
  yylsp -= yylen;
#endif

#if YY_patBisonSpec_DEBUG != 0
  if (YY_patBisonSpec_DEBUG_FLAG)
    {
      short *ssp1 = yyss - 1;
      fprintf (stderr, "state stack now");
      while (ssp1 != yyssp)
	fprintf (stderr, " %d", *++ssp1);
      fprintf (stderr, "\n");
    }
#endif

  *++yyvsp = yyval;

#ifdef YY_patBisonSpec_LSP_NEEDED
  yylsp++;
  if (yylen == 0)
    {
      yylsp->first_line = YY_patBisonSpec_LLOC.first_line;
      yylsp->first_column = YY_patBisonSpec_LLOC.first_column;
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
      ++YY_patBisonSpec_NERRS;

#ifdef YY_patBisonSpec_ERROR_VERBOSE
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
	      YY_patBisonSpec_ERROR(msg);
	      free(msg);
	    }
	  else
	    YY_patBisonSpec_ERROR ("parse error; also virtual memory exceeded");
	}
      else
#endif /* YY_patBisonSpec_ERROR_VERBOSE */
	YY_patBisonSpec_ERROR("parse error");
    }

  YYGOTO(yyerrlab1);
YYLABEL(yyerrlab1)   /* here on error raised explicitly by an action */

  if (yyerrstatus == 3)
    {
      /* if just tried and failed to reuse lookahead token after an error, discard it.  */

      /* return failure if at end of input */
      if (YY_patBisonSpec_CHAR == YYEOF)
	YYABORT;

#if YY_patBisonSpec_DEBUG != 0
      if (YY_patBisonSpec_DEBUG_FLAG)
	fprintf(stderr, "Discarding token %d (%s).\n", YY_patBisonSpec_CHAR, yytname[yychar1]);
#endif

      YY_patBisonSpec_CHAR = YYEMPTY;
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
#ifdef YY_patBisonSpec_LSP_NEEDED
  yylsp--;
#endif

#if YY_patBisonSpec_DEBUG != 0
  if (YY_patBisonSpec_DEBUG_FLAG)
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

#if YY_patBisonSpec_DEBUG != 0
  if (YY_patBisonSpec_DEBUG_FLAG)
    fprintf(stderr, "Shifting error token, ");
#endif

  *++yyvsp = YY_patBisonSpec_LVAL;
#ifdef YY_patBisonSpec_LSP_NEEDED
  *++yylsp = YY_patBisonSpec_LLOC;
#endif

  yystate = yyn;
  YYGOTO(yynewstate);
/* end loop, in which YYGOTO may be used. */
  YYENDGOTO
}

/* END */

/* #line 1010 "/usr/local/lib/bison.cc" */
#line 3748 "patSpecParser.yy.tab.c"
#line 1558 "patSpecParser.yy"



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

 

