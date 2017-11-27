#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

#
# NEED TO DISABLE THE OPTION:  --disable-dependency-tracking
# did not work without it on Farideh's mac. Bug of automake. 
#
AC_PREREQ([2.68])
LT_PREREQ([2.4])
AC_INIT([biogeme], [2.6a], [https://groups.google.com/d/forum/biogeme],[biogeme],[http://biogeme.epfl.ch])
AC_LANG(C++)
AC_CONFIG_AUX_DIR([.])
AC_CONFIG_MACRO_DIR([m4])
LT_INIT([disable-static])
#LT_INIT
LT_LANG([C++])

AC_CANONICAL_HOST 

AM_INIT_AUTOMAKE([no-dependencies -Wno-portability])

AM_CXXFLAGS="$AM_CXXFLAGS -std=gnu++11"
AM_LDFLAGS="$AM_LDFLAGS -no-undefined"
AC_PROG_CXX([clang++ g++])
AC_PROG_CC([clang gcc])

# Remove the next statement before distribution
#AC_DISABLE_SHARED
AC_PROG_LIBTOOL

AC_CONFIG_FILES([Makefile 
	libraries/utils/Makefile 
	libraries/linearAlgebra/Makefile 
	libraries/trustRegion/Makefile 
	libraries/cfsqp/Makefile 
	libraries/ipopt/Makefile 
	libraries/solvopt/Makefile 
	libraries/randomNumbers/Makefile 
	libraries/gaussHermite/Makefile 
	libraries/parameters/Makefile 
	libraries/bisonbiogeme/Makefile 
	libraries/pythonbiogeme/Makefile 
        tools/histograms/Makefile
        tools/prepareData/Makefile
        biogeme/Makefile
	biosim/Makefile 
	mod2py/Makefile 
	biomerge/Makefile 
        pythonbiogeme/Makefile        
	pybiogeme/Makefile
        gtkgui/Makefile])


# For private purposes, add
#	budget/Makefile
#	testing/Makefile

#
AC_CONFIG_FILES([pythonbiogeme/pythonbiogeme],[chmod +x pythonbiogeme/pythonbiogeme])
AC_CONFIG_FILES([pythonbiogeme/debugbiogeme],[chmod +x pythonbiogeme/debugbiogeme])
AC_CONFIG_FILES([gtkgui/debuggui],[chmod +x gtkgui/debuggui])

AC_CONFIG_FILES([tools/prepareData/biocheckdata],[chmod +x tools/prepareData/biocheckdata])

AC_CONFIG_FILES([tools/prepareData/biosampledata],[chmod +x tools/prepareData/biosampledata])

AC_CONFIG_FILES([tools/prepareData/biopreparedata],[chmod +x tools/prepareData/biopreparedata])

AC_CONFIG_FILES([tools/histograms/histogram],[chmod +x tools/histograms/histogram])

AC_CONFIG_FILES([tools/histograms/weightedhistogram],[chmod +x tools/histograms/weightedhistogram])

AC_CONFIG_FILES([pythonbiogeme/pythonbiogeme_static],[chmod +x pythonbiogeme/pythonbiogeme_static])

AC_CONFIG_SRCDIR([libraries/utils/patAbsTime.cc])

AC_CONFIG_HEADERS([config.h])

# Check the options given as argument of configure

# Support for extended precision
AC_ARG_ENABLE([longdouble],AC_HELP_STRING([--enable-longdouble],[Use long double floating point data type (slower)])
)

AC_MSG_CHECKING([whether to build with debug information])
AC_ARG_ENABLE([debug],
    [AS_HELP_STRING([--enable-debug],
        [enable debug data generation (def=no)])],
    [debugit="$enableval"],
    [debugit=no])
AC_MSG_RESULT([$debugit])

if test x"$debugit" = x"yes"; then
    AC_DEFINE([DEBUG],[],[Debug Mode])
    AM_CXXFLAGS="$AM_CXXFLAGS -g -Wall -O0 -fno-omit-frame-pointer"
    AM_LDFLAGS="$AM_LDFLAGS -g"
    WITHDEBUG="yes"
else
    AC_DEFINE([NODEBUG],[],[No-debug Mode])
    AM_CXXFLAGS="$AM_CXXFLAGS -O3"
    WITHDEBUG="no (use option --enable-debug)"
fi

AC_ARG_ENABLE([gui],
  AS_HELP_STRING([--disable-gui],[Does not build the version with a GUI]),[],[enable_gui=yes]
)

AC_ARG_ENABLE([python],
  AS_HELP_STRING([--disable-python],[Does not build Python Biogeme]),[],[enable_python=yes]
)

AC_ARG_ENABLE([bison],
  AS_HELP_STRING([--disable-bison],[Does not build Bison biogeme.]),[],[enable_bison=yes]
)

AC_ARG_ENABLE([staticexec],
  AS_HELP_STRING([--enable-staticexec],[Build executables with static libraries.])
)

# Support for versioned installs
AC_ARG_ENABLE([versioned],AC_HELP_STRING([--enable-versioned],[Install with versioned filenames and paths])
)


# custom program locations

#IPOPT

AC_ARG_WITH([ipopt],
       AS_HELP_STRING([--without-ipopt],
              [disable support for IPOPT]),
            [],
            [with_ipopt=yes]
)

# python31
AC_ARG_WITH([python_dir],
  AS_HELP_STRING([--with-python-dir=DIR],[Specify the directory where Python 3 is.]),
  [python_path=$withval],
  [python_path=$PATH]
)

# bison++
AC_ARG_WITH([bison_dir],
  AS_HELP_STRING([--with-bison-dir=DIR],[Specify the directory where bison++ is installed.]),
  [bison_path=$withval],
  [bison_path=$PATH]
)

# lldb
AC_ARG_WITH([lldb_dir],
  AS_HELP_STRING([--with-lldb-dir=DIR],[Specify the directory where lldb is installed.]),
  [lldb_path=$withval],
  [lldb_path=$PATH]
)

# flex++
AC_ARG_WITH([flex_dir],
  AS_HELP_STRING([--with-flex-dir=DIR],[Specify the directory where flex++ is installed.]),
  [flex_path=$withval],
  [flex_path=$PATH]
)

# latexmk
AC_ARG_WITH([latexmk_dir],
  AS_HELP_STRING([--with-latexmk-dir=DIR],[Specify the directory where latexmk is installed.]),
  [latexmk_path=$withval],
  [latexmk_path=$PATH]
)

#gtkmm

if test "x$enable_gui" = "xyes" ; then 
  PKG_CHECK_MODULES(GTK, [gtkmm-3.0 >= 3.0],[HAVE_GTK="1"],[HAVE_GTK="0"])
  if test "x$HAVE_GTK" = "x1" ; then
    GTKBIO="yes"
    GTKBIO_SUGGEST=""
    AC_MSG_CHECKING([whether to build with a GUI])
    AC_MSG_RESULT(yes)
  else
    GTKBIO="no"
    GTKBIO_SUGGEST="(install gtkmm-3.0)"
    AC_MSG_CHECKING([whether to build with a GUI])
    AC_MSG_RESULT(no)
  fi
else
    GTKBIO="no"
    GTKBIO_SUGGEST="(use option --enable-gui)"
    AC_MSG_CHECKING([whether to build with a GUI])
    AC_MSG_RESULT(no)
fi

if test "x$GTKBIO" = "xyes" ; then
    GTK_CXXFLAGS="`$PKG_CONFIG --cflags-only-I gtkmm-3.0`"
    GTK_STATIC_CXXFLAGS="`$PKG_CONFIG --static --cflags-only-I gtkmm-3.0`"
    GTK_LDFLAGS="`$PKG_CONFIG --libs gtkmm-3.0`"
    GTK_STATIC_LDFLAGS="`$PKG_CONFIG --static --libs gtkmm-3.0`"
    AC_SUBST(GTK_CXXFLAGS)
    AC_SUBST(GTK_LDFLAGS)
    AC_SUBST(GTK_STATIC_CXXFLAGS)
    AC_SUBST(GTK_STATIC_LDFLAGS)
    case "$host_os" in 
      mingw* | msys*)
        CPPFLAGS="-mms-bitfields $CPPFLAGS"
# the -mwindows option removes the console on Windows. 
# the -mms-bitfields option seems to be mandatory on msys2
        LDFLAGS="-mms-bitfields -mwindows $LDFLAGS"
#        LDFLAGS="-mms-bitfields $LDFLAGS"
        AC_DEFINE([IS_WINDOWS],[1],[Define if the operating system is Windows])
      ;;
    esac

   GTK_LIBS="`$PKG_CONFIG --libs-only-l gtkmm-3.0` $LIBS"
  AC_SUBST(GTK_LIBS)

fi

AM_CONDITIONAL([BUILD_GUI], [test "x$GTKBIO" = "xyes"])


# IPOPT


if test "x$with_ipopt" = "xyes" ; then 
  PKG_CHECK_MODULES(IPOPT, [ipopt],[HAVE_IPOPT="1"],[HAVE_IPOPT="0"])
  if test "x$HAVE_IPOPT" = "x1" ; then
    IPOPTBIO="yes"
    IPOPTBIO_SUGGEST=""
    AC_MSG_CHECKING([whether to build with IPOPT])
    AC_MSG_RESULT(yes)
  else
    IPOPTBIO="no"
    IPOPTBIO_SUGGEST="(install IPOPT)"
    AC_MSG_CHECKING([whether to build with IPOPT])
    AC_MSG_RESULT(no)
  fi
else
    IPOPTBIO="no"
    IPOPTBIO_SUGGEST="(disabled by the --without-ipopt option)"
    AC_MSG_CHECKING([whether to build with IPOPT])
    AC_MSG_RESULT(no)
fi


if test "x$IPOPTBIO" = "xyes" ; then
    IPOPT_CXXFLAGS="`$PKG_CONFIG --cflags-only-I ipopt`"
    IPOPT_STATIC_CXXFLAGS="`$PKG_CONFIG --static --cflags-only-I ipopt`"
    IPOPT_LDFLAGS="`$PKG_CONFIG --libs ipopt`"
    IPOPT_STATIC_LDFLAGS="`$PKG_CONFIG --static --libs ipopt`"
    AC_DEFINE([IPOPT],[],[IPOPT optimization package available])
    AC_SUBST(IPOPT_CXXFLAGS)
    AC_SUBST(IPOPT_LDFLAGS)
    AC_SUBST(IPOPT_STATIC_CXXFLAGS)
    AC_SUBST(IPOPT_STATIC_LDFLAGS)
    IPOPT_LIBS="`$PKG_CONFIG --libs-only-l ipopt` $LIBS"
    AC_SUBST(IPOPT_LIBS)
fi


AM_CONDITIONAL([BUILD_IPOPT], [test "x$IPOPTBIO" = "xyes"])


# Python

AC_PATH_PROGS([PYTHONBIN],[python3],[no],[$python_path])
if test "x$PYTHONBIN" = "xno"; then
  NEWBIO="no"
  if test "x$with_python_dir" = "x" ; then
    NEWBIO_SUGGEST="(Download and install python3)"
  else
    NEWBIO_SUGGEST="(python3 not found in $with_python_dir)"
  fi
else 

  PYTHONBINBASE="`basename $PYTHONBIN`"
  PYTHONCONFIG="$srcdir/pythonbiogeme/python/biogeme-python3-config.py"
  PYTHONHEADERDIR="`$PYTHONBIN $PYTHONCONFIG headerdir`"
  AC_CHECK_HEADERS([$PYTHONHEADERDIR/Python.h],[python_header=yes],[python_header=no])
  if test "x$enable_python" = "xyes" -a "x$python_header" = "xyes"; then
    NEWBIO="yes"
    NEWBIO_SUGGEST="(Python version `$PYTHONBIN $PYTHONCONFIG version`)"
    PYTHON_INCLUDES="`$PYTHONBIN $PYTHONCONFIG includes`"
    PYTHON_CXXFLAGS="`$PYTHONBIN $PYTHONCONFIG cxxflags`"
    PYTHON_LIBS="`$PYTHONBIN $PYTHONCONFIG libs`"
    PYTHON_LDFLAGS="`$PYTHONBIN $PYTHONCONFIG ldflags`"
    AC_SUBST(PYTHON_INCLUDES)
    AC_SUBST(PYTHON_CXXFLAGS)
    AC_SUBST(PYTHON_LIBS)
    AC_SUBST(PYTHON_LDFLAGS)
    AC_SUBST(PYTHONBIN)
    AC_DEFINE([PYTHONBIOGEME],[],[PythonBiogeme is installed])
  else 
    NEWBIO="no"
    if test "x$python_header" = "xno"; then
      NEWBIO_SUGGEST="(Install python3-dev)"
    else
      NEWBIO_SUGGEST="(use option --enable-python)"
    fi
  fi
fi
# Bison

if test "x$enable_bison" = "xyes" ; then
  BISONBIO="yes"
  BISONBIO_SUGGEST=""
  AC_DEFINE([BISONBIOGEME],[],[BisonBiogeme is installed])
else 
  BISONBIO="no"
  BISONBIO_SUGGEST="(use option --enable-bison)"
fi

AC_PATH_PROG([BISON],[bison++],[bison++],[$bison_path])
AM_MISSING_PROG([YACC], [$BISON])

AC_PATH_PROG([LLDB],[lldb],[lldb],[$lldb_path])

AC_PATH_PROG([FLEX],[flex++],[flex++],[$flex_path])
AM_MISSING_PROG([LEX], [$FLEX])

AC_PATH_PROG([LATEXMK],[latexmk],[:],[$latexmk_path])
AC_SUBST([LEX_OUTPUT_ROOT],[lex.yy])

AC_CHECK_DECLS([_SC_NPROCESSORS_CONF],,,[[#include <unistd.h>]])

# Support for portable batch system

AC_PATH_PROG([PBS],[qsub],[no])
AM_CONDITIONAL([SYSTEM_WITH_PBS],[test "x$PBS" != "xno"])

# Checks for libraries.

# Checks for header files.
AC_CHECK_HEADERS([float.h stdlib.h sys/time.h sys/wait.h errno.h math.h malloc.h sys/malloc.h stdio.h pthread.h unistd.h direct.h])

#Check setenv
AC_CHECK_FUNC(setenv,AC_DEFINE([HAVE_SETENV],[1],[Define if setenv exists.]),[])

AC_CHECK_FUNC(_putenv_s,AC_DEFINE([HAVE_PUTENV_S],[1],[Define if _putenv_s exists.]),[])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE

# Checks for library functions.
AC_FUNC_MKTIME
AC_CHECK_FUNCS([floor ceil gettimeofday localtime_r localtime time pow exp log sqrt atan mktime sched_getaffinity sysctlbyname getcwd _getcwd execlp system stat])
AC_CHECK_FUNC(chdir,
  [AC_DEFINE([HAVE_CHDIR], [1], [Define to 1 if you have chdir.])],
  [AC_CHECK_FUNC(SetCurrentDirectory,
    [AC_DEFINE([HAVE_SETCURRENTDIRECTORY], [1], [Define to 1 if you have SetCurrentDirectory.])])]
)

#Decide what to build

AM_CONDITIONAL([BUILD_DEBUG],[test "x$debugit" = "xyes"])
AM_CONDITIONAL([BUILD_LONGDOUBLE], [test "x$enable_longdouble" = "xyes"])
AM_CONDITIONAL([BUILD_PYTHON],[test "x$PYTHONBIN" != "xno" -a "x$enable_python" = "xyes"])
AM_CONDITIONAL([BUILD_BISONBIOGEME],[test "x$enable_bison" = "xyes"])

if test "x$enable_staticexec" = "xyes" ; then
  STATIC="yes"
  if test "x$lt_cv_prog_compiler_static_works" = "xno" ; then
    STATIC="no (flag -static not supported by the compiler)"
  else
    AM_LDFLAGS="$AM_LDFLAGS -static -static-libgcc -static-libstdc++"
  fi
else
  STATIC="no (use option --enable-staticexec)"
fi

AM_CONDITIONAL([BUILD_STATIC], [test "x$STATIC" = "xyes"])

if test "x$enable_longdouble" = "xyes" ; then
  WITHLONGDOUBLE="yes"
  AC_DEFINE([LONGDOUBLE],[1],[Use long double floating point data type])
else
  WITHLONGDOUBLE="no (use option --enable-longdouble)"
fi

# Check if it is an Apple computer. This is to insert Apple specific options
AM_CONDITIONAL([APPLE], [AS_CASE([$host_os], [darwin*], [true], [false])])

#Decide where to install

AM_CONDITIONAL([INSTALL_VERSIONED], [test "x$enable_versioned" = "xyes" ])
if test "x$enable_versioned" = "xyes" ; then
  INSTALL_VERSION_SUFFIX="-${VERSION}"
  INSTALL_RELEASE="-release ${VERSION}"
  VERSIONED="yes (with suffix -${VERSION})"
else
  INSTALL_VERSION_SUFFIX=""
  INSTALL_RELEASE=""
  VERSIONED="no (use option --enable-versioned)"
fi
AC_SUBST(INSTALL_RELEASE)
AC_SUBST([INSTALL_VERSION_SUFFIX])
AC_SUBST([AM_CXXFLAGS])
AC_SUBST([AM_LDFLAGS])


AC_OUTPUT

# GNU Scientific library...$WITHGSL

AC_MSG_RESULT([
------------------------------------------------------------------------

  $PACKAGE $VERSION for $host_os:  Automatic configuration OK.
 Build biogeme............$BISONBIO $BISONBIO_SUGGEST
 Build biomerge...........$BISONBIO $BISONBIO_SUGGEST
 Build biosim.............$BISONBIO $BISONBIO_SUGGEST
 Build python biogeme.....$NEWBIO $NEWBIO_SUGGEST
 Build GTK GUI version....$GTKBIO $GTKBIO_SUGGEST
 Use IPOPT optimization...$IPOPTBIO $IPOPTBIO_SUGGEST
 Static executables.......$STATIC
 Long double..............$WITHLONGDOUBLE
 Executables for debug....$WITHDEBUG
 Versioned installation...$VERSIONED
])
