# Makefile.am
# libraries/cfsqp

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(top_srcdir)/libraries/utils -I$(top_srcdir)/libraries/linearAlgebra  -I$(top_srcdir)/libraries/trustRegion 

noinst_LTLIBRARIES = libcfsqp.la

libcfsqp_la_CXXFLAGS = $(AM_CXXFLAGS)
 
if BUILD_STATIC
noinst_LIBRARIES = libcfsqp.a
libcfsqp_a_SOURCES = $(libcfsqp_la_SOURCES)
libcfsqp_a_CXXFLAGS = $(AM_CXXFLAGS)
endif

libcfsqp_la_SOURCES = \
mycfsqp.cc \
myqld.cc \
patCfsqp.cc \
cfsqpusr.h \
patCfsqp.h 