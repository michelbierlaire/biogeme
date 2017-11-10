# Makefile.am
# libraries/gaussHermite

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(top_srcdir)/libraries/utils 

noinst_LTLIBRARIES = libgaussHermite.la

libgaussHermite_la_CXXFLAGS = $(AM_CXXFLAGS)

if BUILD_STATIC
noinst_LIBRARIES = libgaussHermite.a
libgaussHermite_a_SOURCES = $(libgaussHermite_la_SOURCES)
libgaussHermite_a_CXXFLAGS = $(AM_CXXFLAGS)
endif


libgaussHermite_la_SOURCES = \
bioGaussHermite.cc \
bioGhFunction.cc \
bioGaussHermite.h \
bioGhFunction.h
