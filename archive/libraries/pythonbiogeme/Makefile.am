# Makefile.am
# bisonBiogeme

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(top_srcdir)/libraries/utils  -I$(top_srcdir)/libraries/randomNumbers -I$(top_srcdir)/libraries/gaussHermite -I$(top_srcdir)/libraries/cfsqp -I$(top_srcdir)/libraries/ipopt -I$(top_srcdir)/libraries/solvopt -I$(top_srcdir)/libraries/linearAlgebra -I$(top_srcdir)/libraries/trustRegion $(PYTHON_INCLUDES) 

if BUILD_IPOPT
AM_CPPFLAGS += $(IPOPT_CXXFLAGS)
endif

if BUILD_PYTHON

noinst_LTLIBRARIES = libpythonbiogeme.la

libpythonbiogeme_la_CXXFLAGS = $(AM_CXXFLAGS) $(PYTHON_CXXFLAGS)
 
if BUILD_STATIC
noinst_LIBRARIES = libpythonbiogeme.a
libpythonbiogeme_a_SOURCES = $(libpythonbiogeme_la_SOURCES)
libpythonbiogeme_a_CXXFLAGS = $(AM_CXXFLAGS)
endif

endif


libpythonbiogeme_la_SOURCES = \
bioPythonSingletonFactory.cc \
bioFunctionAndDerivatives.cc \
bioIterationBackup.cc \
bioArithMonteCarlo.cc \
bioArithMatrix.cc  \
bioArithAbs.cc  \
bioArithAnd.cc \
bioArithBinaryMinus.cc \
bioArithBinaryPlus.cc \
bioArithMultinaryPlus.cc \
bioArithConstant.cc \
bioArithDerivative.cc \
bioArithDivide.cc \
bioArithElem.cc \
bioArithLogLogit.cc \
bioArithNormalPdf.cc \
bioArithEqual.cc \
bioArithExp.cc \
bioArithGreater.cc \
bioArithGreaterEqual.cc \
bioArithGHIntegral.cc \
bioArithGaussHermite.cc \
bioArithIterator.cc \
bioArithLess.cc \
bioArithLessEqual.cc \
bioArithLiteral.cc \
bioArithCompositeLiteral.cc \
bioArithRandomVariable.cc \
bioArithFixedParameter.cc \
bioArithVariable.cc \
bioArithLog.cc \
bioArithMax.cc \
bioArithMin.cc \
bioArithMult.cc \
bioArithNormalCdf.cc \
bioArithNotEqual.cc \
bioArithOr.cc \
bioArithPower.cc \
bioArithPrint.cc \
bioArithProd.cc \
bioArithRandom.cc \
bioArithUnifRandom.cc \
bioArithSum.cc \
bioArithUnaryMinus.cc \
bioArithBayes.cc \
bioArithBayesMH.cc \
bioArithBayesMean.cc \
bioConstraintWrapper.cc \
bioConstraints.cc \
bioCountOperations.cc \
bioDrawIterator.cc \
bioExpression.cc \
bioArithElementaryExpression.cc \
bioArithUnaryExpression.cc \
bioArithBinaryExpression.cc \
bioArithMultinaryExpression.cc \
bioExpressionRepository.cc \
bioFixedParameter.cc \
bioCompositeLiteral.cc \
bioPrecompiledFunction.cc \
bioIteratorInfo.cc \
bioIteratorInfoRepository.cc \
bioIteratorSpan.cc \
bioLiteral.cc \
bioLiteralRepository.cc \
bioLiteralValues.cc \
bioMain.cc \
bioMetaIterator.cc \
bioMinimizationProblem.cc \
bioModel.cc \
bioModelParser.cc \
bioParameters.cc \
bioPythonWrapper.cc \
bioRandomDraws.cc \
bioRandomVariable.cc \
bioRawResults.cc \
bioReporting.cc \
bioRowIterator.cc \
bioIdIterator.cc \
bioSample.cc \
bioSimulation.cc \
bioSplit.cc \
bioStatistics.cc \
bioVariable.cc \
bioVersion.cc \
bioSensitivityAnalysis.cc \
bioSimulatedValues.cc \
bioBayesianResults.cc \
bioPythonSingletonFactory.h \
bioArithMonteCarlo.h \
bioArithMatrix.h \
bioArithAbs.h \
bioArithAnd.h \
bioArithBinaryExpression.h \
bioArithBinaryMinus.h \
bioArithMultinaryPlus.h \
bioArithBinaryPlus.h \
bioArithCompositeLiteral.h \
bioArithConstant.h \
bioArithDerivative.h \
bioArithDivide.h \
bioArithElem.h \
bioArithElementaryExpression.h \
bioArithEqual.h \
bioArithExp.h \
bioArithFixedParameter.h \
bioArithGaussHermite.h \
bioArithGreater.h \
bioArithGreaterEqual.h \
bioArithGHIntegral.h \
bioArithIterator.h \
bioArithLess.h \
bioArithLessEqual.h \
bioArithLiteral.h \
bioArithLog.h \
bioArithLogLogit.h \
bioArithNormalPdf.h \
bioArithMax.h \
bioArithMin.h \
bioArithMult.h \
bioArithMultinaryExpression.h \
bioArithNormalCdf.h \
bioArithNotEqual.h \
bioArithOr.h \
bioArithPower.h \
bioArithPrint.h \
bioArithProd.h \
bioArithRandom.h \
bioArithUnifRandom.h \
bioArithRandomVariable.h \
bioArithSum.h \
bioArithUnaryExpression.h \
bioArithUnaryMinus.h \
bioArithBayes.h \
bioArithBayesMH.h \
bioArithBayesMean.h \
bioArithVariable.h \
bioCompositeLiteral.h \
bioConstraintWrapper.h \
bioConstraints.h \
bioCountOperations.h \
bioDrawIterator.h \
bioExpression.h \
bioExpressionRepository.h \
bioFixedParameter.h \
bioFunctionAndDerivatives.h \
bioIterationBackup.h \
bioIterator.h \
bioIteratorInfo.h \
bioIteratorInfoRepository.h \
bioIteratorSpan.h \
bioLiteral.h \
bioLiteralRepository.h \
bioLiteralValues.h \
bioMain.h \
bioMetaIterator.h \
bioMinimizationProblem.h \
bioModel.h \
bioModelParser.h \
bioModelParser_i.h \
bioParameters.h \
bioOneParameter.h \
bioParameterIterator.h \
bioPrecompiledFunction.h \
bioPythonWrapper.h \
bioRandomDraws.h \
bioRandomVariable.h \
bioRawResults.h \
bioReporting.h \
bioRowIterator.h \
bioIdIterator.h \
bioSample.h \
bioSimulation.h \
bioSplit.h \
bioStatistics.h \
bioVariable.h \
bioVersion.h \
bioBayesianResults.h \
bioSensitivityAnalysis.h \
bioSimulatedValues.h

nodist_libpythonbiogeme_la_SOURCES = \
bioUser.h \
bioCompilationDate.h

BUILT_SOURCES = \
bioUser.h \
bioCompilationDate.h

bioUser.h:
	echo "#define __BIOUSER \"`id -un` on "`uname`"\""> bioUser.h

bioCompilationDate.h:
	echo "#define __BIOCOMPILATIONDATE \""`date`"\""> bioCompilationDate.h

clean-local:
	rm -f bioCompilationDate.h bioUser.h

