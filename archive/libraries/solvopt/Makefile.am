# Makefile.am
# solvopt

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(top_srcdir)/libraries/utils -I$(top_srcdir)/libraries/linearAlgebra  -I$(top_srcdir)/libraries/trustRegion 

noinst_LTLIBRARIES = libsolvopt.la

libsolvopt_la_CXXFLAGS = $(AM_CXXFLAGS)

if BUILD_STATIC
noinst_LIBRARIES = libsolvopt.a
libsolvopt_a_SOURCES = $(libsolvopt_la_SOURCES)
libsolvopt_a_CXXFLAGS = $(AM_CXXFLAGS)
endif

libsolvopt_la_SOURCES = \
myapprgrdn.cc \
mysolvopt.cc \
patSolvOpt.cc \
patSolvOpt.h \
solvopt.h \
solvoptParameters.h 
