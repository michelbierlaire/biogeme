# Makefile.am
# libraries/ipopt

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS =  -I$(top_srcdir)/libraries/pythonbiogeme -I$(top_srcdir)/libraries/utils -I$(top_srcdir)/libraries/linearAlgebra  -I$(top_srcdir)/libraries/trustRegion 

noinst_LTLIBRARIES = libipopt.la

libipopt_la_CXXFLAGS = $(AM_CXXFLAGS)

if BUILD_IPOPT
libipopt_la_CXXFLAGS += $(IPOPT_CXXFLAGS)
endif

if BUILD_STATIC
noinst_LIBRARIES = libipopt.a
libipopt_a_SOURCES = $(libipopt_la_SOURCES)
libipopt_a_CXXFLAGS = $(AM_CXXFLAGS) $(IPOPT_STATIC_CXXFLAGS)
endif

libipopt_la_SOURCES = \
bioIpopt.cc \
bioIpopt.h
