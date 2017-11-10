# Makefile.am
# linearAlgebra

ACLOCAL_AMFLAGS = -I m4

AM_CPPFLAGS = -I$(top_srcdir)/libraries/utils

noinst_LTLIBRARIES = liblinearAlgebra.la

liblinearAlgebra_la_CXXFLAGS = $(AM_CXXFLAGS)

if BUILD_STATIC
noinst_LIBRARIES = liblinearAlgebra.a
liblinearAlgebra_a_SOURCES = $(liblinearAlgebra_la_SOURCES)
liblinearAlgebra_a_CXXFLAGS = $(AM_CXXFLAGS)
endif

liblinearAlgebra_la_SOURCES =  \
patEigenVectors.cc \
patGeneralizedInverseIteration.cc \
patHouseholder.cc \
patHybridMatrix.cc \
patInverse.cc \
patInverseIteration.cc \
patLu.cc \
patMatrix.cc \
patMyMatrix.cc \
patPythag.cc \
patQR.cc \
patSvd.cc \
patKalman.cc \
patEigenVectors.h \
patGeneralizedInverseIteration.h \
patHouseholder.h \
patHybridMatrix.h \
patInverse.h \
patInverseIteration.h \
patLu.h \
patMatrix.h \
patMyMatrix.h \
patNrSgn.h \
patPythag.h \
patQR.h \
patSvd.h \
patKalman.h 