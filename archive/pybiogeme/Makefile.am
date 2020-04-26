# Makefile.am
# pybiogeme

include $(top_srcdir)/libraries.mk

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(top_srcdir)/libraries/utils  -I$(top_srcdir)/libraries/randomNumbers -I$(top_srcdir)/libraries/cfsqp -I$(top_srcdir)/libraries/ipopt -I$(top_srcdir)/libraries/solvopt -I$(top_srcdir)/libraries/linearAlgebra -I$(top_srcdir)/libraries/trustRegion -I$(top_srcdir)/libraries/pythonbiogeme $(PYTHON_INCLUDES)


if BUILD_PYTHON

pybiogeme_CXXFLAGS = $(AM_CXXFLAGS) $(PYTHON_CXXFLAGS)
pybiogeme_LDADD = $(PYTHON_SHARED_LIBS) $(COMMON_SHARED_LIBS)
pybiogeme_LDFLAGS = $(AM_LDFLAGS) $(PYTHON_LDFLAGS)
if APPLE
pybiogeme_LDFLAGS += -Wl,-stack_size -Wl,1000000
endif
LIBS += $(PYTHON_LIBS)

if BUILD_IPOPT
pybiogeme_LDFLAGS += $(IPOPT_LDFLAGS)
endif

bin_PROGRAMS = pybiogeme
if BUILD_STATIC
pybiogeme_static_CXXFLAGS = $(AM_CXXFLAGS) $(pybiogeme_CXXFLAGS)
pybiogeme_static_SOURCES = $(pybiogeme_SOURCES)
pybiogeme_static_LDFLAGS = $(AM_LDFLAGS) $(PYTHON_LDFLAGS)
if APPLE
pybiogeme_static_LDFLAGS += -Wl,-stack_size -Wl,1000000
endif
pybiogeme_static_LDADD = $(PYTHON_STATIC_LIBS) $(COMMON_STATIC_LIBS) $(PYTHON_LIBS)
if BUILD_IPOPT
pybiogeme_static_LDFLAGS += $(IPOPT_STATIC_LDFLAGS)
endif
bin_PROGRAMS += pybiogeme_static
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

pybiogeme_SOURCES = \
pybiogeme.cc

