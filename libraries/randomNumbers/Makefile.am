# Makefile.am
# randomNumbers

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(top_srcdir)/libraries/utils -I$(top_srcdir)/libraries/linearAlgebra

noinst_LTLIBRARIES = librandomNumbers.la

librandomNumbers_la_CXXFLAGS = $(AM_CXXFLAGS)

if BUILD_STATIC
noinst_LIBRARIES = librandomNumbers.a
librandomNumbers_a_SOURCES = $(librandomNumbers_la_SOURCES)
librandomNumbers_a_CXXFLAGS = $(AM_CXXFLAGS)
endif


librandomNumbers_la_SOURCES =  \
patRandomNumberGenerator.cc \
patBinomial.cc \
patCenteredUniform.cc \
patDiscreteDistribution.cc \
patDrawsFromFile.cc \
patGenerateNormalDraws.cc \
patHalton.cc \
patHessTrain.cc \
patNormal.cc \
patNormalWichura.cc \
patRandomInteger.cc \
patUnixUniform.cc \
patUniform.cc \
patZeroOneUniform.cc \
patMultivariateNormal.cc \
patBinomial.h \
patCenteredUniform.h \
patDiscreteDistribution.h \
patDrawsFromFile.h \
patGenerateNormalDraws.h \
patHalton.h \
patHessTrain.h \
patNormal.h \
patNormalWichura.h \
patRandomInteger.h \
patRandomNumberGenerator.h \
patUniform.h \
patUnixUniform.h \
patZeroOneUniform.h \
patMultivariateNormal.h
