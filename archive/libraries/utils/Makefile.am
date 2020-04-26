# Makefile.am
# libraries/utils

ACLOCAL_AMFLAGS = -I m4 

noinst_LTLIBRARIES = libutils.la

libutils_la_CXXFLAGS = $(AM_CXXFLAGS)
 
#if BUILD_PYTHON
#libutils_la_CXXFLAGS += $(PYTHON_CXXFLAGS)
#endif

#if BUILD_GUI
#libutils_la_CXXFLAGS += $(GTK_CXXFLAGS)
#endif

if BUILD_STATIC
noinst_LIBRARIES = libutils.a
libutils_a_SOURCES = $(libutils_la_SOURCES)
libutils_a_CXXFLAGS = $(AM_CXXFLAGS)
#if BUILD_PYTHON
#libutils_a_CXXFLAGS += $(PYTHON_CXXFLAGS)
#endif
#if BUILD_GUI
#libutils_a_CXXFLAGS += $(GTK_CXXFLAGS)
#endif

endif

libutils_la_SOURCES = patAbsTime.cc \
    patAngle.cc \
    patCompareCorrelation.cc \
    patDisplay.cc \
    patEnvironmentVariable.cc \
    patEnvPathVariable.cc \
    patErrFileNotFound.cc \
    patErrMiscError.cc \
    patErrNullPointer.cc \
    patError.cc \
    patExecScript.cc \
    patFileExists.cc \
    patFileNames.cc \
    patOutputFiles.cc \
    patSingletonFactory.cc \
    patFormatRealNumbers.cc \
    patFunction.cc \
    patCorrelation.cc \
    patGeoCoordinates.cc \
    patImportance.cc \
    patLargeNumbers.cc \
    patFileSize.cc \
    patStatistics.cc \
    patLoopTime.cc \
    patMessage.cc \
    patNonParamRegression.cc \
    patNormalCdf.cc \
    patNormalPdf.cc \
    patOutputTable.cc \
    patPValue.cc \
    patPower.cc \
    patPrimeNumbers.cc \
    patPsTricks.cc \
    patString.cc \
    patTimeInterval.cc \
    patTimer.cc \
    patVariables.cc \
    patVersion.cc \
    patQuantiles.cc \
    patLap.cc \
    patInterval.cc \
    patGetWorkingDir.cc \
FlexLexer.h \
patAbsTime.h \
patAngle.h \
patBiogemeRun.h \
patClass.h \
patCompare.h \
patCompareCorrelation.h \
patConst.h \
patCorrelation.h \
patDisplay.h \
patEnvironmentVariable.h \
patEnvPathVariable.h \
patErrFileNotFound.h \
patErrMiscError.h \
patErrNullPointer.h \
patErrOutOfRange.h \
patError.h \
patExecScript.h \
patFileExists.h \
patFileNames.h \
patOutputFiles.h \
patSingletonFactory.h \
patFormatRealNumbers.h \
patFunction.h \
patGenerateCombinations.h \
patGeoCoordinates.h \
patImportance.h \
patIterationBackup.h \
patIterator.h \
patLargeNumbers.h \
patLogMessage.h \
patFileSize.h \
patStatistics.h \
patLoop.h \
patLoopTime.h \
patMath.h \
patMessage.h \
patMtl.h \
patNonParamRegression.h \
patNormalCdf.h \
patNormalPdf.h \
patOutputTable.h \
patPValue.h \
patPower.h \
patPrimeNumbers.h \
patPsTricks.h \
patSequenceIterator.h \
patShuffleVector.h \
patStlSetIterator.h \
patStlVectorIterator.h \
patString.h \
patSubvectorIterator.h \
patSubvectorPtrIterator.h \
patThreeStrings.h \
patTimeInterval.h \
patTimer.h \
patType.h \
patUtilFunction.h \
patVariables.h \
patVectorIterator.h \
patVectorPtrIterator.h \
patVersion.h \
trNonLinearAlgo.h \
patQuantiles.h \
patLap.h \
patInterval.h \
patGetWorkingDir.h \
pstream.h



nodist_libutils_la_SOURCES = \
patUser.h \
patCompilationDate.h

BUILT_SOURCES = \
patUser.h \
patCompilationDate.h

patUser.h:
	echo "#define __PATUSER \"`id -un` on "`uname`"\""> patUser.h

patCompilationDate.h:
	echo "#define __PATCOMPILATIONDATE \""`date`"\""> patCompilationDate.h

clean-local:
	rm -f patCompilationDate.h patUser.h
