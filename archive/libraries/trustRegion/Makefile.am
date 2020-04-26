# Makefile.am
# libraries/trustRegion

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(top_srcdir)/libraries/utils -I$(top_srcdir)/libraries/linearAlgebra

noinst_LTLIBRARIES = libtrustRegion.la

libtrustRegion_la_CXXFLAGS = $(AM_CXXFLAGS)
 
if BUILD_STATIC
noinst_LIBRARIES = libtrustRegion.a
libtrustRegion_a_SOURCES = $(libtrustRegion_la_SOURCES)
libtrustRegion_a_CXXFLAGS = $(AM_CXXFLAGS)
endif

libtrustRegion_la_SOURCES =  \
patNonLinearProblem.cc \
patSimBasedOptimizationProblem.cc \
trBFGS.cc \
trBounds.cc \
trBoxGCP.cc \
trFunction.cc \
trHessian.cc \
trHybridMatrix.cc \
trLineSearchAlgo.cc \
trNonLinearAlgo.cc \
trPrecond.cc \
trPrecondCG.cc \
trQuadratic.cc \
trSR1.cc \
trSchnabelEskow.cc \
trSecantUpdate.cc \
trSimBasedSimpleBoundsAlgo.cc \
trSimpleBoundsAlgo.cc \
trTheoryGCP.cc \
trTointSteihaug.cc \
trParameters.cc \
patNonLinearProblem.h \
patSimBasedOptimizationProblem.h \
trBFGS.h \
trBounds.h \
trBoxGCP.h \
trFunction.h \
trHessian.h \
trHybridMatrix.h \
trLineSearchAlgo.h \
trMatrixVector.h \
trParameters.h \
trPrecond.h \
trPrecondCG.h \
trQuadratic.h \
trSR1.h \
trSchnabelEskow.h \
trSecantUpdate.h \
trSimBasedSimpleBoundsAlgo.h \
trSimpleBoundsAlgo.h \
trTheoryGCP.h \
trTointSteihaug.h \
trVector.h 
