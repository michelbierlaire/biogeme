# Libraries

COMMON_SHARED_LIBS = $(shell pwd)/../libraries/randomNumbers/librandomNumbers.la \
           $(shell pwd)/../libraries/cfsqp/libcfsqp.la \
           $(shell pwd)/../libraries/ipopt/libipopt.la \
           $(shell pwd)/../libraries/solvopt/libsolvopt.la \
           $(shell pwd)/../libraries/trustRegion/libtrustRegion.la \
           $(shell pwd)/../libraries/linearAlgebra/liblinearAlgebra.la \
           $(shell pwd)/../libraries/utils/libutils.la 

COMMON_STATIC_LIBS = $(shell pwd)/../libraries/randomNumbers/librandomNumbers.a \
           $(shell pwd)/../libraries/cfsqp/libcfsqp.a \
           $(shell pwd)/../libraries/ipopt/libipopt.a \
           $(shell pwd)/../libraries/solvopt/libsolvopt.a \
           $(shell pwd)/../libraries/trustRegion/libtrustRegion.a \
           $(shell pwd)/../libraries/linearAlgebra/liblinearAlgebra.a \
           $(shell pwd)/../libraries/utils/libutils.a 


BISON_SHARED_LIBS = $(shell pwd)/../libraries/bisonbiogeme/libbisonbiogeme.la \
                     $(shell pwd)/../libraries/parameters/libparameters.la

BISON_STATIC_LIBS =  $(shell pwd)/../libraries/bisonbiogeme/libbisonbiogeme.a \
                     $(shell pwd)/../libraries/parameters/libparameters.a 

PYTHON_SHARED_LIBS =  $(shell pwd)/../libraries/pythonbiogeme/libpythonbiogeme.la \
                      $(shell pwd)/../libraries/gaussHermite/libgaussHermite.la 

PYTHON_STATIC_LIBS = $(shell pwd)/../libraries/pythonbiogeme/libpythonbiogeme.a \
                     $(shell pwd)/../libraries/gaussHermite/libgaussHermite.a                    
