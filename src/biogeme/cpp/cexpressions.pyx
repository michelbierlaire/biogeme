# distutils: language=c++
# cython: embedsignature=True

cimport numpy as np
import numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool as bool_t

ctypedef vector[unsigned long] uint_vector
ctypedef vector[uint_vector] uint_matrix
ctypedef vector[double] double_vector
ctypedef vector[double_vector] double_matrix
ctypedef vector[double_matrix] double_tensor
ctypedef vector[string] string_vector

ctypedef int[::1] uint_vector_view
ctypedef int[:, ::1] uint_matrix_view
ctypedef double[::1] double_vector_view
ctypedef double[:, ::1] double_matrix_view
ctypedef double[:, :, ::1] double_tensor_view


cdef extern from "evaluateExpressions.h":

    cdef cppclass evaluateOneExpression:
        evaluateOneExpression() except +

        void setExpression(string_vector loglikeSignatures) except +

        void calculate(bool_t gradient,
                       bool_t hessian,
                       bool_t bhhh,
               bool_t aggregation) except +

        void setNumberOfThreads(unsigned long numberOfThreads)

        void setFreeBetas(double_vector freeBetas)

        void setFixedBetas(double_vector fixedBetas)

        void setData(double_matrix& d)

        void setDataMap(uint_matrix& dm)

        void setMissingData(double md)

        void setDraws(double_tensor& draws)

        void setPanel(bool_t panel)

        void getResults(double* f, double* g, double* h, double* bhhh) except +

        unsigned int getDimension()

        unsigned int getSampleSize()


cdef class pyEvaluateOneExpression:
    cdef evaluateOneExpression theEvaluation

    def __cinit__(self):
        self.theEvaluation = evaluateOneExpression()

    def setExpression(self, formula):
        self.theEvaluation.setExpression(formula)

    def setFreeBetas(self, freeBetas):
        self.theEvaluation.setFreeBetas(freeBetas)

    def setFixedBetas(self, fixedBetas):
        self.theEvaluation.setFixedBetas(fixedBetas)

    def setNumberOfThreads(self, n):
        self.theEvaluation.setNumberOfThreads(n)

    def setData(self, d):
        d = np.ascontiguousarray(d)
        self.theEvaluation.setData(d)

    def setDraws(self, draws):
        draws = np.ascontiguousarray(draws)
        self.theEvaluation.setDraws(draws)


    def setDataMap(self, dm):
        dm = np.ascontiguousarray(dm)
        self.theEvaluation.setDataMap(dm)

    def setMissingData(self, md):
        self.theEvaluation.setMissingData(md)

    def calculate(self, gradient, hessian, bhhh, aggregation):
        self.theEvaluation.calculate(
            gradient,
            hessian,
            bhhh,
            aggregation,
        )


    def getResults(self):
        n = self.theEvaluation.getDimension()
        sample = self.theEvaluation.getSampleSize() ;
        f = np.empty(sample)
        if not f.flags['C_CONTIGUOUS']:
            f = np.ascontiguousarray(f)
        cdef double_vector_view f_view = f

        if n == 0:
            self.theEvaluation.getResults(&f_view[0], NULL, NULL, NULL)
            return f, None, None, None

        g = np.empty([sample, n])
        if not g.flags['C_CONTIGUOUS']:
            g = np.ascontiguousarray(g)
        h = np.empty([sample, n, n])
        if not h.flags['C_CONTIGUOUS']:
            h = np.ascontiguousarray(h)
        bhhh = np.empty([sample, n, n])
        if not bhhh.flags['C_CONTIGUOUS']:
            bhhh = np.ascontiguousarray(bhhh)

        cdef double_matrix_view g_view = g
        cdef double_tensor_view h_view = h
        cdef double_tensor_view bhhh_view = bhhh

        self.theEvaluation.getResults(
	    &f_view[0],
            &g_view[0, 0],
	    &h_view[0, 0, 0],
	    &bhhh_view[0, 0, 0],
	)

        return f, g, h, bhhh
