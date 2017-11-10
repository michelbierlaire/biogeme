#ifndef YY_patBisonSpec_h_included
#define YY_patBisonSpec_h_included

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
#line 21 "patSpecParser.yy.tab.h"
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

#line 14 "/usr/local/lib/bison.h"
 /* %{ and %header{ and %union, during decl */
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
/* WARNING obsolete !!! user defined YYLTYPE not reported into generated header */
/* use %define LTYPE */
#endif
#endif
#ifdef YYSTYPE
#ifndef YY_patBisonSpec_STYPE 
#define YY_patBisonSpec_STYPE YYSTYPE
/* WARNING obsolete !!! user defined YYSTYPE not reported into generated header */
/* use %define STYPE */
#endif
#endif
#ifdef YYDEBUG
#ifndef YY_patBisonSpec_DEBUG
#define  YY_patBisonSpec_DEBUG YYDEBUG
/* WARNING obsolete !!! user defined YYDEBUG not reported into generated header */
/* use %define DEBUG */
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

/* #line 63 "/usr/local/lib/bison.h" */
#line 234 "patSpecParser.yy.tab.h"

#line 63 "/usr/local/lib/bison.h"
/* YY_patBisonSpec_PURE */
#endif

/* #line 65 "/usr/local/lib/bison.h" */
#line 241 "patSpecParser.yy.tab.h"

#line 65 "/usr/local/lib/bison.h"
/* prefix */
#ifndef YY_patBisonSpec_DEBUG

/* #line 67 "/usr/local/lib/bison.h" */
#line 248 "patSpecParser.yy.tab.h"

#line 67 "/usr/local/lib/bison.h"
/* YY_patBisonSpec_DEBUG */
#endif
#ifndef YY_patBisonSpec_LSP_NEEDED

/* #line 70 "/usr/local/lib/bison.h" */
#line 256 "patSpecParser.yy.tab.h"

#line 70 "/usr/local/lib/bison.h"
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

/* TOKEN C */
#ifndef YY_USE_CLASS

#ifndef YY_patBisonSpec_PURE
extern YY_patBisonSpec_STYPE YY_patBisonSpec_LVAL;
#endif


/* #line 143 "/usr/local/lib/bison.h" */
#line 334 "patSpecParser.yy.tab.h"
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


#line 143 "/usr/local/lib/bison.h"
 /* #defines token */
/* after #define tokens, before const tokens S5*/
#else
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

/* #line 182 "/usr/local/lib/bison.h" */
#line 484 "patSpecParser.yy.tab.h"
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


#line 182 "/usr/local/lib/bison.h"
 /* decl const */
#else
enum YY_patBisonSpec_ENUM_TOKEN { YY_patBisonSpec_NULL_TOKEN=0

/* #line 185 "/usr/local/lib/bison.h" */
#line 598 "patSpecParser.yy.tab.h"
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


#line 185 "/usr/local/lib/bison.h"
 /* enum token */
     }; /* end of enum declaration */
#endif
public:
 int YY_patBisonSpec_PARSE(YY_patBisonSpec_PARSE_PARAM);
 virtual void YY_patBisonSpec_ERROR(char *msg) YY_patBisonSpec_ERROR_BODY;
#ifdef YY_patBisonSpec_PURE
#ifdef YY_patBisonSpec_LSP_NEEDED
 virtual int  YY_patBisonSpec_LEX(YY_patBisonSpec_STYPE *YY_patBisonSpec_LVAL,YY_patBisonSpec_LTYPE *YY_patBisonSpec_LLOC) YY_patBisonSpec_LEX_BODY;
#else
 virtual int  YY_patBisonSpec_LEX(YY_patBisonSpec_STYPE *YY_patBisonSpec_LVAL) YY_patBisonSpec_LEX_BODY;
#endif
#else
 virtual int YY_patBisonSpec_LEX() YY_patBisonSpec_LEX_BODY;
 YY_patBisonSpec_STYPE YY_patBisonSpec_LVAL;
#ifdef YY_patBisonSpec_LSP_NEEDED
 YY_patBisonSpec_LTYPE YY_patBisonSpec_LLOC;
#endif
 int YY_patBisonSpec_NERRS;
 int YY_patBisonSpec_CHAR;
#endif
#if YY_patBisonSpec_DEBUG != 0
public:
 int YY_patBisonSpec_DEBUG_FLAG;	/*  nonzero means print parse trace	*/
#endif
public:
 YY_patBisonSpec_CLASS(YY_patBisonSpec_CONSTRUCTOR_PARAM);
public:
 YY_patBisonSpec_MEMBERS 
};
/* other declare folow */
#endif


#if YY_patBisonSpec_COMPATIBILITY != 0
/* backward compatibility */
#ifndef YYSTYPE
#define YYSTYPE YY_patBisonSpec_STYPE
#endif

#ifndef YYLTYPE
#define YYLTYPE YY_patBisonSpec_LTYPE
#endif
#ifndef YYDEBUG
#ifdef YY_patBisonSpec_DEBUG 
#define YYDEBUG YY_patBisonSpec_DEBUG
#endif
#endif

#endif
/* END */

/* #line 236 "/usr/local/lib/bison.h" */
#line 760 "patSpecParser.yy.tab.h"

#line 1560 "patSpecParser.yy"

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

 #endif
