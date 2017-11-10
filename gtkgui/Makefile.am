include $(top_srcdir)/libraries.mk
pkgdatdir = ${pkgdatadir}$(INSTALL_VERSION_SUFFIX)

ACLOCAL_AMFLAGS = -I m4 
AM_CPPFLAGS = -I$(top_srcdir)/libraries/utils -I$(top_srcdir)/libraries/randomNumbers -I$(top_srcdir)/libraries/parameters -I$(top_srcdir)/libraries/cfsqp -I$(top_srcdir)/libraries/ipopt -I$(top_srcdir)/libraries/solvopt -I$(top_srcdir)/libraries/linearAlgebra -I$(top_srcdir)/libraries/trustRegion -I$(top_srcdir)/libraries/bisonbiogeme -I$(top_srcdir)/libraries/pythonbiogeme -D__PYTHONBIOGEME=\"$(pkgdatdir)\"


if BUILD_GUI

AM_CPPFLAGS += $(AM_CXXFLAGS) $(GTK_CXXFLAGS) "-D__DATADIR=\"$(pkgdatdir)\""

gtkguibiogeme_LDFLAGS = $(AM_LDFLAGS)
gtkguibiogeme_LDADD =  $(GTK_LDFLAGS)
if BUILD_BISONBIOGEME
gtkguibiogeme_LDADD += $(BISON_SHARED_LIBS) $(COMMON_SHARED_LIBS)
endif
if BUILD_PYTHON
gtkguibiogeme_LDADD += $(PYTHON_SHARED_LIBS) $(COMMON_SHARED_LIBS)
gtkguibiogeme_LDFLAGS += $(PYTHON_LDFLAGS)
LIBS += $(PYTHON_LIBS) 
endif

if BUILD_IPOPT
gtkguibiogeme_LDFLAGS += $(IPOPT_LDFLAGS)
endif


bin_PROGRAMS = gtkguibiogeme
if BUILD_DEBUG
bin_SCRIPTS = debuggui
endif
if BUILD_STATIC
gtkguibiogeme_static_CXXFLAGS = $(AM_CXXFLAGS) $(gtkguibiogeme_CXXFLAGS) $(GTK_STATIC_CXXFLAGS)
gtkguibiogeme_static_LDFLAGS = $(GTK_STATIC_LDFLAGS)
gtkguibiogeme_static_SOURCES = $(gtkguibiogeme_SOURCES)
gtkguibiogeme_static_GTKSOURCES = $(gtkguibiogeme_GTKSOURCES)
gtkguibiogeme_static_LDADD = 
if BUILD_IPOPT
gtkguibiogeme_static_LDFLAGS += $(IPOPT_STATIC_LDFLAGS)
endif

if BUILD_BISONBIOGEME
gtkguibiogeme_static_LDADD += $(BISON_STATIC_LIBS) $(COMMON_STATIC_LIBS)
endif
if BUILD_PYTHON
gtkguibiogeme_static_LDADD += $(PYTHON_STATIC_LIBS) $(COMMON_STATIC_LIBS)
endif

bin_PROGRAMS += gtkguibiogeme_static
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


gtkguibiogeme_SOURCES = \
gtkguibiogeme.cc \
bioGtkGui.cc \
bioGtkGui.h \
bioLogWindow.cc \
bioLogWindow.h \
bioGtkScriptDialog.cc \
bioGtkScriptDialog.h \
bioGtkFileViewer.cc \
bioGtkFileViewer.h \
bioFilePackage.cc \
bioFilePackage.h \
bioUiFile.cc \
bioUiFile.h \
logo.h

EXTRA_DIST = windowsInstallation gtkbiogeme.ui 

dist_pkgdat_DATA = gtkbiogeme.ui 
