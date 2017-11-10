# Makefile.am
# libraries/bisonBiogeme

ACLOCAL_AMFLAGS = -I m4
AM_YFLAGS = -d
AM_LFLAGS = -+ -Pyyspec -olex.yy.c

AM_CPPFLAGS = -I$(top_srcdir)/libraries/utils  -I$(top_srcdir)/libraries/randomNumbers -I$(top_srcdir)/libraries/parameters -I$(top_srcdir)/libraries/cfsqp -I$(top_srcdir)/libraries/solvopt -I$(top_srcdir)/libraries/linearAlgebra -I$(top_srcdir)/libraries/trustRegion

if BUILD_BISONBIOGEME

noinst_LTLIBRARIES = libbisonbiogeme.la

libbisonbiogeme_la_CXXFLAGS = $(AM_CXXFLAGS)
 
if BUILD_STATIC
noinst_LIBRARIES = libbisonbiogeme.a
libbisonbiogeme_a_SOURCES = $(libbisonbiogeme_la_SOURCES)
libbisonbiogeme_a_CXXFLAGS = $(AM_CXXFLAGS)
endif

endif

libbisonbiogeme_la_SOURCES = patSpecParser.yy patSpecScanner.ll \
patBisonSingletonFactory.cc \
patAdditiveUtility.cc \
patAggregateObservationData.cc \
patArithAbs.cc \
patArithAnd.cc \
patArithAttribute.cc \
patArithBinaryMinus.cc \
patArithBinaryPlus.cc \
patArithConstant.cc \
patArithDeriv.cc \
patArithDivide.cc \
patArithEqual.cc \
patArithExp.cc \
patArithGreater.cc \
patArithGreaterEqual.cc \
patArithInt.cc \
patArithLess.cc \
patArithLessEqual.cc \
patArithLog.cc \
patArithMax.cc \
patArithMin.cc \
patArithMod.cc \
patArithMult.cc \
patArithNode.cc \
patArithNormalRandom.cc \
patArithNot.cc \
patArithNotEqual.cc \
patArithOr.cc \
patArithPower.cc \
patArithRandom.cc \
patArithSqrt.cc \
patArithUnaryMinus.cc \
patArithUnifRandom.cc \
patArithVariable.cc \
patArrayIterator.cc \
patAttributeNamesIterator.cc \
patBetaLikeIterator.cc \
patBetaLikeParameter.cc \
patBiogeme.cc \
patBiogemeIterationBackup.cc \
patBiogemeScripting.cc \
patCNL.cc \
patCnlAlphaIterator.cc \
patConstraintNestIterator.cc \
patDataFile.cc \
patDataFileAggObsIterator.cc \
patDataFileIndIterator.cc \
patDataFileObsIterator.cc \
patDataMemory.cc \
patDiscreteParameterIterator.cc \
patDiscreteParameterProba.cc \
patEstimationResult.cc \
patFastBiogeme.cc \
patFileIterator.cc \
patFullCnlAlphaIterator.cc \
patGEV.cc \
patGeneralizedUtility.cc \
patIndividualData.cc \
patLegendre.cc \
patLikelihood.cc \
patLinearConstraint.cc \
patLinearUtility.cc \
patMaxLikeProblem.cc \
patMemoryObsNoPanelIterator.cc \
patMemoryObservationIterator.cc \
patMinimizedFunction.cc \
patModelSpec.cc \
patModelSpecCtd.cc \
patNL.cc \
patNetworkAlphaIterator.cc \
patNetworkGevAlt.cc \
patNetworkGevIterator.cc \
patNetworkGevModel.cc \
patNetworkGevNest.cc \
patNetworkGevNode.cc \
patNlNestIterator.cc \
patNonLinearConstraint.cc \
patObservationData.cc \
patOneZhengFosgerau.cc \
patProbaGevModel.cc \
patProbaGevPanelModel.cc \
patProbaMnlModel.cc \
patProbaMnlPanelModel.cc \
patProbaModel.cc \
patProbaOrdinalLogit.cc \
patProbaPanelModel.cc \
patProbaProbitModel.cc \
patPythonResults.cc \
patRandomParameter.cc \
patSample.cc \
patSampleEnuGetIndices.cc \
patSampleEnumeration.cc \
patSecondDerivatives.cc \
patSimBasedMaxLikeOptimization.cc \
patTestNormal.cc \
patUsedAttributeNamesIterator.cc \
patUtility.cc \
patValueVariables.cc \
patZhengFosgerau.cc \
patZhengTest.cc \
patBisonSingletonFactory.h \
patAdditiveUtility.h \
patAggregateObservationData.h \
patAlternative.h \
patArithAbs.h \
patArithAnd.h \
patArithAttribute.h \
patArithBinaryMinus.h \
patArithBinaryPlus.h \
patArithConstant.h \
patArithDeriv.h \
patArithDivide.h \
patArithEqual.h \
patArithExp.h \
patArithGreater.h \
patArithGreaterEqual.h \
patArithInt.h \
patArithLess.h \
patArithLessEqual.h \
patArithLog.h \
patArithMax.h \
patArithMin.h \
patArithMod.h \
patArithMult.h \
patArithNode.h \
patArithNormalRandom.h \
patArithNot.h \
patArithNotEqual.h \
patArithOr.h \
patArithPower.h \
patArithRandom.h \
patArithSqrt.h \
patArithUnaryMinus.h \
patArithUnifRandom.h \
patArithVariable.h \
patArrayIterator.h \
patAttributeNamesIterator.h \
patBetaLikeIterator.h \
patBetaLikeParameter.h \
patBiogeme.h \
patBiogemeIterationBackup.h \
patBiogemeScripting.h \
patCNL.h \
patCnlAlphaIterator.h \
patCnlAlphaParameter.h \
patConstraintNestIterator.h \
patDataFile.h \
patDataFileAggObsIterator.h \
patDataFileIndIterator.h \
patDataFileObsIterator.h \
patDataMemory.h \
patDataStorage.h \
patDiscreteParameter.h \
patDiscreteParameterIterator.h \
patDiscreteParameterProba.h \
patDistribType.h \
patEstimationResult.h \
patFastBiogeme.h \
patFileIterator.h \
patFullCnlAlphaIterator.h \
patGEV.h \
patGeneralizedUtility.h \
patIndividualData.h \
patLegendre.h \
patLikelihood.h \
patLinearConstraint.h \
patLinearUtility.h \
patMaxLikeProblem.h \
patMemoryObsNoPanelIterator.h \
patMemoryObservationIterator.h \
patMinimizedFunction.h \
patModelSpec.h \
patModelSpecCtd.h \
patNL.h \
patNetworkAlphaIterator.h \
patNetworkGevAlt.h \
patNetworkGevIterator.h \
patNetworkGevLinkParameter.h \
patNetworkGevModel.h \
patNetworkGevNest.h \
patNetworkGevNode.h \
patNlNestDefinition.h \
patNlNestIterator.h \
patNonLinearConstraint.h \
patObservationData.h \
patOneZhengFosgerau.h \
patProbaGevModel.h \
patProbaGevPanelModel.h \
patProbaMnlModel.h \
patProbaMnlPanelModel.h \
patProbaModel.h \
patProbaOrdinalLogit.h \
patProbaPanelModel.h \
patProbaProbitModel.h \
patPythonResults.h \
patRandomDraws.h \
patRandomParameter.h \
patSample.h \
patSampleEnuGetIndices.h \
patSampleEnumeration.h \
patSecondDerivatives.h \
patSimBasedMaxLikeOptimization.h \
patTestNormal.h \
patUsedAttributeNamesIterator.h \
patUtility.h \
patValueVariables.h \
patZhengFosgerau.h \
patZhengTest.h


