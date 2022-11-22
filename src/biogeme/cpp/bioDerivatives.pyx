# distutils: language=c++
# cython: embedsignature=True

cimport numpy as np
import numpy as np

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool as bool_t

ctypedef vector[double] double_vector
ctypedef vector[double_vector] double_matrix
ctypedef vector[double_matrix] double_tensor

ctypedef double[::1] double_vector_view
ctypedef double[:, ::1] double_matrix_view

cdef extern from "bioDerivatives.h"

     cdef cppclass bioDerivatives:
          bioDerivatives() except +

          double get_function()

          void get_gradient(double* g)

          voit get_hessian(double* h)

cdef class pyDerivatives:
     cdef bioDerivatives theDerivatives

     def __cinit__(self):
     self.theDerivatives = bioDerivatives()