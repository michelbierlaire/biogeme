include $(top_srcdir)/libraries.mk

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(top_srcdir)/libraries/utils  -I$(top_srcdir)/libraries/randomNumbers -I$(top_srcdir)/libraries/parameters -I$(top_srcdir)/libraries/cfsqp  -I$(top_srcdir)/libraries/solvopt -I$(top_srcdir)/libraries/linearAlgebra -I$(top_srcdir)/libraries/trustRegion -I$(top_srcdir)/libraries/bisonbiogeme

if BUILD_BISONBIOGEME

biosim_CXXFLAGS = $(AM_CXXFLAGS)
biosim_LDADD = $(BISON_SHARED_LIBS) $(COMMON_SHARED_LIBS)

bin_PROGRAMS = biosim
if BUILD_STATIC
biosim_static_CXXFLAGS = $(AM_CXXFLAGS) $(biosim_CXXFLAGS)
biosim_static_SOURCES = $(biosim_SOURCES)
biosim_static_LDADD = $(BISON_STATIC_LIBS) $(COMMON_STATIC_LIBS)
bin_PROGRAMS += biosim_static
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

biosim_SOURCES = biosim.cc patSimVersion.cc patSimVersion.h
