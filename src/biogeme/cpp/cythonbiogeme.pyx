# distutils: language=c++
# cython: embedsignature=True

# Hack to create the initialization code
STUFF = 'hack'

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


cdef extern from "biogeme.h":

	cdef cppclass biogeme:
		biogeme() except +

		double calculateLikelihood(double_vector betas, 
			double_vector fixedBetas) except +

		double calculateLikeAndDerivatives(double_vector betas,
			double_vector fixedBetas,
			uint_vector betaIds,
			double* g,
			double* h,
			double* bhhh,
			bool_t hessian,
			bool_t bhhh) except +

		void setPanel(bool_t p)

		void setBounds(double_vector lb, double_vector ub)

		void simulateFormula(vector[string] loglikeSignatures,
				     double_vector betas, 
				     double_vector fixedBetas,
				     double_matrix& data,
				     double* results) except +

		double simulateSimpleFormula(vector[string] loglikeSignatures,
                                              double_vector betas, 
                                              double_vector fixedBetas,
                                              bool_t gradient,
                                              bool_t hessian,
                                              double* g,
                                              double* h) except +


		void simulateSeveralFormulas(vector[string_vector] loglikeSignatures,
				     double_vector betas, 
				     double_vector fixedBetas,
				     unsigned long numberOfThreads,
				     double_matrix& data,
				     double* results) except +

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

	def calculateLikelihoodAndDerivatives(self,
	                                      betas,
					      fixedBetas,
					      betaIds,
					      gmem,
					      hmem,
					      bmem,
					      hessian,
					      bhhh,
					      draws=None):
		n = len(betas)

		if not gmem.flags['C_CONTIGUOUS']:
			print('gmem not contiguous')
			gmem = np.ascontiguousarray(gmem)
		if not hmem.flags['C_CONTIGUOUS']:
			print('hmem not contiguous')
			hmem = np.ascontiguousarray(hmem)
		if not bmem.flags['C_CONTIGUOUS']:
			print('bmem not contiguous')
			bmem = np.ascontiguousarray(bmem)

		cdef double_vector_view gmem_view = gmem
		cdef double_matrix_view hmem_view = hmem
		cdef double_matrix_view bmem_view = bmem


		f = self.theBiogeme.calculateLikeAndDerivatives(betas,
			                                        fixedBetas,
								betaIds,
								&gmem_view[0],
								&hmem_view[0,0],
								&bmem_view[0,0],
								hessian,
								bhhh)
		return f, gmem, hmem, bmem

	def setBounds(self,lb,ub):
		self.theBiogeme.setBounds(lb,ub)

	def calculateLikelihood(self, betas,fixedBetas):
		r = self.theBiogeme.calculateLikelihood(betas, fixedBetas)
		return r

	def simulateSimpleFormula(self, 
                                  formula, 
                                  betas, 
                                  fixedBetas,
                                  gradient,
                                  hessian,
                                  gmem,
                                  hmem):
		if not gmem.flags['C_CONTIGUOUS']:
			print('gmem not contiguous')
			gmem = np.ascontiguousarray(gmem)
		if not hmem.flags['C_CONTIGUOUS']:
			print('hmem not contiguous')
			hmem = np.ascontiguousarray(hmem)

		cdef double_vector_view gmem_view = gmem
		cdef double_matrix_view hmem_view = hmem

		r = self.theBiogeme.simulateSimpleFormula(formula,
					                 betas, 
   						        fixedBetas,
                                                           gradient,
                                                           hessian,
                                                           &gmem_view[0],
                                                           &hmem_view[0,0])
		return r, gmem, hmem

	def simulateFormula(self, formula, betas, fixedBetas, d):
		n = d.shape[0]	
		r = np.empty(n)
		if not r.flags['C_CONTIGUOUS']:
			print('r not contiguous')
			r = np.ascontiguousarray(r)
		d = np.ascontiguousarray(d)

		cdef double_vector_view r_view = r
		self.theBiogeme.simulateFormula(formula,
					       betas, 
   					       fixedBetas, 
					       d, 
					       &r_view[0])
		return r
	
	def simulateSeveralFormulas(self, formulas, betas, fixedBetas, d, nThreads):
		n = d.shape[0]
		nf = len(formulas)
		r = np.zeros([nf, n])
		if not r.flags['C_CONTIGUOUS']:
			print('r not contiguous')
			r = np.ascontiguousarray(r)
		d = np.ascontiguousarray(d)

		cdef double_matrix_view r_view = r
		self.theBiogeme.simulateSeveralFormulas(formulas,
				        	      betas, 
   						      fixedBetas, 
 						      nThreads,
						      d, 
						      &r_view[0,0])
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


				


