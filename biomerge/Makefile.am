include $(top_srcdir)/libraries.mk

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(top_srcdir)/libraries/utils  -I$(top_srcdir)/libraries/randomNumbers -I$(top_srcdir)/libraries/parameters -I$(top_srcdir)/libraries/cfsqp  -I$(top_srcdir)/libraries/solvopt -I$(top_srcdir)/libraries/linearAlgebra -I$(top_srcdir)/libraries/trustRegion -I$(top_srcdir)/libraries/bisonbiogeme


biomerge_CXXFLAGS = $(AM_CXXFLAGS) 
biomerge_LDADD = $(COMMON_SHARED_LIBS)

bin_PROGRAMS = biomerge
if BUILD_STATIC
biomerge_static_CXXFLAGS = $(AM_CXXFLAGS) $(biomerge_CXXFLAGS)
biomerge_static_SOURCES = $(biomerge_SOURCES)
biomerge_static_LDADD = $(COMMON_STATIC_LIBS)
bin_PROGRAMS += biomerge_static
endif

if INSTALL_VERSIONED
install-exec-hook:
	cd $(DESTDIR)$(bindir) && \
	for p in $(bin_PROGRAMS); do \
	mv -f $${p}$(EXEEXT) $${p}$(INSTALL_VERSION_SUFFIX)$(EXEEXT) && \
	$(LN_S) $${p}$(INSTALL_VERSION_SUFFIX)$(EXEEXT) $${p}$(EXEEXT); \
	done
endif




biomerge_SOURCES = biomerge.cc

