# Makefile.am
# biogeme

include $(top_srcdir)/libraries.mk

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(top_srcdir)/libraries/utils  -I$(top_srcdir)/libraries/randomNumbers -I$(top_srcdir)/libraries/parameters -I$(top_srcdir)/libraries/cfsqp  -I$(top_srcdir)/libraries/solvopt -I$(top_srcdir)/libraries/linearAlgebra -I$(top_srcdir)/libraries/trustRegion -I$(top_srcdir)/libraries/bisonbiogeme

if BUILD_BISONBIOGEME

biogeme_CXXFLAGS = $(AM_CXXFLAGS) 
biogeme_LDADD = $(BISON_SHARED_LIBS) $(COMMON_SHARED_LIBS)

bin_PROGRAMS = biogeme
if BUILD_STATIC
biogeme_static_CXXFLAGS = $(biogeme_CXXFLAGS)
biogeme_static_SOURCES = $(biogeme_SOURCES)
biogeme_static_LDADD = $(BISON_STATIC_LIBS) $(COMMON_STATIC_LIBS)
bin_PROGRAMS += biogeme_static
endif

if INSTALL_VERSIONED
install-exec-hook:
	cd $(DESTDIR)$(bindir) && \
	for p in $(bin_PROGRAMS); do \
	mv -f $${p}$(EXEEXT) $${p}$(INSTALL_VERSION_SUFFIX)$(EXEEXT) && \
	$(LN_S) $${p}$(INSTALL_VERSION_SUFFIX)$(EXEEXT) $${p}$(EXEEXT); \
	done
endif

endif


biogeme_SOURCES = \
biogeme.cc

