# Makefile.am
# parameters

ACLOCAL_AMFLAGS = -I m4
AM_YFLAGS = -d
AM_LFLAGS = -+ -olex.yy.c
AM_CPPFLAGS = -I$(top_srcdir)/libraries/utils 

if BUILD_BISONBIOGEME

noinst_LTLIBRARIES = libparameters.la

libparameters_la_CXXFLAGS = $(AM_CXXFLAGS)
 
if BUILD_STATIC
noinst_LIBRARIES = libparameters.a
libparameters_a_SOURCES = $(libparameters_la_SOURCES)
libparameters_a_CXXFLAGS = $(AM_CXXFLAGS)
endif

endif

libparameters_la_SOURCES = patParserParam.yy patScannerParam.ll patParameters.cc  patParameters.h



