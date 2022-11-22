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


cdef extern from "biogeme.h":

	cdef cppclass biogeme:
		biogeme() except +

		double calculateLikelihood(double_vector betas, 
			double_vector fixedBetas) except +

		double calculateLikeAndDerivatives(double_vector betas, 
			double_vector fixedBetas, 
			uint_vector betaIds, 
			double_vector& g,
			double_matrix& h,
			double_matrix& bhhh,
			bool_t hessian,
			bool_t bhhh) except +

		string cfsqp(double_vector betas,
			double_vector fixedBetas,
			uint_vector betaIds,
			unsigned long nit,
			unsigned long nf,
			unsigned long mode,
			unsigned long iprint,
			unsigned long miter,
			double eps) except +

		void setPanel(bool_t p)

		void setBounds(double_vector lb, double_vector ub)

		void simulateFormula(vector[string] loglikeSignatures,
					double_vector betas, 
					double_vector fixedBetas,
				     double_matrix& data,
				     double_vector& results) except +

		void setExpressions(vector[string] loglikeSignatures, 
						vector[string] weightSignatures,
						unsigned long numberOfThreads)

		void setData(double_matrix& d)

		void setDataMap(uint_matrix& dm)

		void setMissingData(double md)
		
		void setDraws(double_tensor& draws)


cdef class pyBiogeme:
	cdef biogeme theBiogeme

	def __cinit__(self):
		self.theBiogeme = biogeme()

	def setPanel(self,panel=True):
		self.theBiogeme.setPanel(panel)

	def calculateLikelihoodAndDerivatives(self,betas,fixedBetas,betaIds,hessian,bhhh,draws=None):
		cdef double_vector g
		cdef double_matrix h
		cdef double_matrix b
		g = np.empty(len(betas))
		h = np.empty([len(betas),len(betas)])
		b = np.empty([len(betas),len(betas)])
		f = self.theBiogeme.calculateLikeAndDerivatives(betas,fixedBetas,betaIds,g,h,b,hessian,bhhh)
		return f,g,h,b

	def cfsqp(self,betas,fixedBetas,betaIds,mode,iprint,moter,eps):
		cdef double_vector b = betas
		cdef unsigned long nit
		cdef unsigned long nf
		diag = self.theBiogeme.cfsqp(b,fixedBetas,betaIds,nit,nf,mode,iprint,moter,eps)
		return b,nit,nf,diag

	def setBounds(self,lb,ub):
		self.theBiogeme.setBounds(lb,ub)

	def calculateLikelihood(self, betas,fixedBetas):
		r = self.theBiogeme.calculateLikelihood(betas,fixedBetas)
		return r

	def simulateFormula(self, formula, betas,fixedBetas, d):	
		cdef double_vector r
		d = np.ascontiguousarray(d)
		self.theBiogeme.simulateFormula(formula,betas,fixedBetas,d,r)
		return r
	
	def setExpressions(self,loglikeFormulas,nbrOfThreads,weightFormulas=None):
		cdef vector[string] w
		if (weightFormulas is not None):
			w = weightFormulas
		self.theBiogeme.setExpressions(loglikeFormulas,w,nbrOfThreads)

	def setData(self, d):
		d = np.ascontiguousarray(d)
		self.theBiogeme.setData(d)

	def setDataMap(self, m):
		m = np.ascontiguousarray(m)
		self.theBiogeme.setDataMap(m)

	def setMissingData(self, md):
		self.theBiogeme.setMissingData(md)


	def setDraws(self, draws):
		draws = np.ascontiguousarray(draws)
		self.theBiogeme.setDraws(draws)

				


				


