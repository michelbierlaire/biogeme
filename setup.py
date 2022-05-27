
"""Instructions for the installation of Biogeme

:author: Michel Bierlaire
:date: Tue Mar 26 16:45:15 2019

"""


import os
import sys
import platform
from setuptools import setup, Extension, find_packages


import numpy

if platform.system() == 'Darwin':
    os.environ["CC"] = 'clang'
    os.environ["CXX"] = 'clang++'
else:
    os.environ["CC"] = 'gcc'
    os.environ["CXX"] = 'g++'

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
    print('Not using Cython')
else:
    USE_CYTHON = True
    print('Using Cython')

ext = '.pyx' if USE_CYTHON else '.cpp'

def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


cmdclass = {}

source = ['src/biogeme.cc',
          'src/evaluateExpressions.cc',
          'src/bioMemoryManagement.cc',
          'src/bioNormalCdf.cc',
          'src/bioFormula.cc',
          'src/bioSeveralFormulas.cc',
          'src/bioThreadMemory.cc',
          'src/bioThreadMemoryOneExpression.cc',
          'src/bioThreadMemorySimul.cc',
          'src/bioString.cc',
          'src/bioExprNormalCdf.cc',
          'src/bioExprIntegrate.cc',
          'src/bioExprGaussHermite.cc',
          'src/bioExprRandomVariable.cc',
          'src/bioExprMontecarlo.cc',
          'src/bioExprPanelTrajectory.cc',
          'src/bioExprDraws.cc',
          'src/bioExprDerive.cc',
          'src/bioExprMin.cc',
          'src/bioExprMax.cc',
          'src/bioExprAnd.cc',
          'src/bioExprOr.cc',
          'src/bioExprEqual.cc',
          'src/bioExprNotEqual.cc',
          'src/bioExprLessOrEqual.cc',
          'src/bioExprLess.cc',
          'src/bioExprGreaterOrEqual.cc',
          'src/bioExprGreater.cc',
          'src/bioExprElem.cc',
          'src/bioExprMultSum.cc',
          'src/bioExprLiteral.cc',
          'src/bioExprFreeParameter.cc',
          'src/bioExprFixedParameter.cc',
          'src/bioExprVariable.cc',
          'src/bioExprPlus.cc',
          'src/bioExprMinus.cc',
          'src/bioExprTimes.cc',
          'src/bioExprDivide.cc',
          'src/bioExprPower.cc',
          'src/bioExprUnaryMinus.cc',
          'src/bioExprExp.cc',
          'src/bioExprLog.cc',
          'src/bioExprNumeric.cc',
          'src/bioExprLogLogit.cc',
          'src/bioExprLogLogitFullChoiceSet.cc',
          'src/bioExprLinearUtility.cc',
          'src/bioExpression.cc',
          'src/bioSeveralExpressions.cc',
          'src/bioExceptions.cc',
          'src/bioDerivatives.cc',
          'src/bioVectorOfDerivatives.cc',
          'src/bioGaussHermite.cc',
          'src/bioGhFunction.cc']


extra_compile_args = ['-std=c++11', '-Wall']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args.append('-stdlib=libc++')
    extra_link_args.append('-lc++')

if sys.platform == 'win32':
    # mismatch between library names
    extra_compile_args.append('-D_hypot=hypot')
    # as one cannot assume that libstdc++, libgcc and pthreads exists
    # on windows, static link them so they are included in the
    # compiled python extension.
    extra_link_args.extend(['-mwindows',
                            '-mms-bitfields',
                            '-static-libstdc++',
                            '-static-libgcc',
                            '-Wl,-Bstatic',
                            '-lstdc++',
                            '-lpthread'])

biogeme_extension = Extension('biogeme.cbiogeme',
                              ['src/cbiogeme'+ext] + source,
                              include_dirs=[numpy.get_include()],
                              extra_compile_args=extra_compile_args,
                              language='c++11',
                              define_macros=[('NPY_NO_DEPRECATED_API',
                                              'NPY_1_7_API_VERSION')],
                              extra_link_args=extra_link_args)

expressions_extension = Extension('biogeme.cexpressions',
                                  ['src/cexpressions'+ext] + source,
                                  include_dirs=[numpy.get_include()],
                                  extra_compile_args=extra_compile_args,
                                  language='c++11',
                                  define_macros=[('NPY_NO_DEPRECATED_API',
                                                  'NPY_1_7_API_VERSION')],
                                  extra_link_args=extra_link_args)

extensions = [biogeme_extension, expressions_extension]

if USE_CYTHON:
    if sys.platform == 'win32':
    
        extensions = cythonize(extensions,
                               compiler_directives={
                                   'language_level' : "3",
                                   'embedsignature': True},
                               include_path=[numpy.get_include()])
    else:
        extensions = cythonize(extensions,
                               nthreads=8,
                               compiler_directives={
                                   'language_level' : "3",
                                   'embedsignature': True},
                               include_path=[numpy.get_include()])
else:
    no_cythonize(extensions)

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")


setup(
    include_package_data=True,
    package_data={'biogeme': ['_biogeme.pyd']},
    install_requires=install_requires,
    package_dir={'biogeme.cbiogeme': 'src'},
    cmdclass=cmdclass,
    ext_modules=extensions)
