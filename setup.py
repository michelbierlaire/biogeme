"""Instructions for the installation of Biogeme

:author: Michel Bierlaire
:date: Tue Mar 26 16:45:15 2019

"""

# Too constraining
# pylint: disable=invalid-name, protected-access, too-many-arguments

import os
import sys
import platform
import distutils.ccompiler
import multiprocessing.pool
import setuptools
#from setuptools import setup, Extension
#import setuptools.command.test


import numpy

from biogeme.version import __version__

if platform.system() == 'Darwin':
    os.environ["CC"] = 'clang'
    os.environ["CXX"] = 'clang++'
else:
    os.environ["CC"] = 'gcc'
    os.environ["CXX"] = 'g++'




# monkey-patch for parallel compilation
def parallelCCompile(self,
                     sources,
                     output_dir=None,
                     macros=None,
                     include_dirs=None,
                     debug=0,
                     extra_preargs=None,
                     extra_postargs=None,
                     depends=None):
    """ those lines are copied from distutils.ccompiler.CCompiler directly"""
    macros, objects, extra_postargs, pp_opts, build = \
        self._setup_compile(output_dir,
                            macros,
                            include_dirs,
                            sources,
                            depends,
                            extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    # parallel code
    N = 8 # number of parallel compilations
    def _single_compile(obj):
        try:
            src, extension = build[obj]
        except KeyError:
            return
        self._compile(obj, src, extension, cc_args, extra_postargs, pp_opts)
    # convert to list, imap is evaluated on-demand
    list(multiprocessing.pool.ThreadPool(N).imap(_single_compile, objects))
    return objects

distutils.ccompiler.CCompiler.compile = parallelCCompile

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
    print('Not using Cython')
else:
    USE_CYTHON = True
    print('Using Cython')

ext = '.pyx' if USE_CYTHON else '.cpp'

cmdclass = {}

source = ['src/biogeme.cc',
          'src/bioNormalCdf.cc',
          'src/bioFormula.cc',
          'src/bioThreadMemory.cc',
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
          'src/bioExceptions.cc',
          'src/bioDerivatives.cc',
          'src/bioGaussHermite.cc',
          'src/bioGhFunction.cc',
          'src/mycfsqp.cc',
          'src/myqld.cc',
          'src/bioCfsqp.cc']


extra_compile_args = ['-std=c++11', '-Wall']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args.append('-stdlib=libc++')
    extra_link_args.append('-lc++')

if sys.platform == 'win32':
    # mismatch between library names
    extra_compile_args.append('-D_hypot=hypot')
    # as one cannot assume that libstdc++, libgcc and pthreads exists on windows,
    # static link them so they are included in the compiled python extension.
    extra_link_args.extend(['-mwindows',
                            '-mms-bitfields',
                            '-static-libstdc++',
                            '-static-libgcc',
                            '-Wl,-Bstatic',
                            '-lstdc++',
                            '-lpthread'])

extensions = [setuptools.Extension('biogeme.cbiogeme',
                                   ['src/cbiogeme'+ext] + source,
                                   include_dirs=[numpy.get_include()],
                                   extra_compile_args=extra_compile_args,
                                   language='c++11',
                                   extra_link_args=extra_link_args)]

#extensions = [Extension('biogeme.cbiogeme',
#                        ['src/cbiogeme'+ext]+source,
#                        include_dirs=[numpy.get_include()],
#                        extra_compile_args=extra_compile_args,
#                        language='c++11',
#                        extra_link_args = ['-lprofiler'])]


#extra_compile_args=['-O0'],
#extra_link_args=['-fsanitize=address','-O1','-fno-omit-frame-pointer','-g']

if USE_CYTHON:
    extensions = cythonize(extensions, language='c++', include_path=[numpy.get_include()])
    cmdclass.update({'build_ext': build_ext})

#exec(open('biogeme/version.py').read())
# now we have a `__version__` variable
# later on we use: __version__

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(name='biogeme',
                 version=__version__,
                 description='Estimation and application of discrete choice models',
                 url='http://biogeme.epfl.ch',
                 author='Michel Bierlaire',
                 keywords='discrete choice maximum likelihood estimation',
                 author_email='michel.bierlaire@epfl.ch',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 install_requires=['numpy', 'cython', 'unidecode', 'scipy', 'pandas'],
                 packages=setuptools.find_packages(),
                 include_package_data=True,
                 package_data={'biogeme': ['_biogeme.pyd']},
                 package_dir={'biogeme.cbiogeme': 'src'},
                 cmdclass=cmdclass,
                 classifiers=[
                     'Programming Language :: Python :: 3',
                     'Operating System :: OS Independent',
                 ],
                 ext_modules=extensions)
