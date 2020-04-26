//-*-c++-*------------------------------------------------------------
//
// File name : trHessian.cc
// Author :    Michel Bierlaire
// Date :      Mon Jan 24 14:36:57 2000
//
//--------------------------------------------------------------------

#include "trHessian.h"
#include "trSchnabelEskow.h"
#include "patErrNullPointer.h"

trHessian::trHessian(trParameters p,
		     unsigned long size) : theParameters(p),
					   matrix(size), 
					   submatrix(NULL) {

  matrix.setType(patHybridMatrix::patSymmetric) ;

}

trHessian::trHessian(const trHessian& h) : theParameters(h.theParameters), matrix(h.matrix), submatrix(NULL)  {

}

void trHessian::copy(const trHessian& h) {
  theParameters = h.theParameters ;
  matrix = h.matrix ;
  if (submatrix != NULL) {
    DELETE_PTR(submatrix) ;
  }
  submatrix = NULL ;
}

trHessian::~trHessian() {

  if (submatrix != NULL) {
    DELETE_PTR(submatrix) ;
  }
}

trVector trHessian::operator()(const trVector& x, 
			       patError*& err)  {

  if (err != NULL) {
    WARNING(err->describe()) ;
    return trVector() ;
  }

  if (!matrix.isSymmetric()) {
    err = new patErrMiscError("Hessian must be a symmetric matrix") ;
    WARNING(err->describe()) ;
    return trVector() ;
  }

  if (matrix.getSize() != x.size()) {
    err = new patErrMiscError("Incompatible sizes") ;
    WARNING(err->describe()) ;
    return trVector() ;
  }

  // There is a smarter way to implement this, but I have to rush. Hopefully,
  // I'll have time to optimize it later on.

  trVector res(x.size(),0.0) ;

  //i = row index
  for (unsigned long i = 0 ;
       i < x.size() ;
       ++i) {
    //j = column index
    for (unsigned long j = 0 ;
	 j < x.size() ;
	 ++j) {
      res[i] += matrix(i,j,err) * x[j] ; 
    }
  }
  return res ;
}

patBoolean trHessian::providesPreconditionner() const {
  return patTRUE ;
}

trPrecond* 
trHessian::createPreconditionner(patError*& err) const {
  trSchnabelEskow* p = new trSchnabelEskow(matrix) ;
  if (p == NULL) {
    err = new patErrNullPointer("trPrecond") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  p->factorize(theParameters.toleranceSchnabelEskow,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
//   DEBUG_MESSAGE("Preconditionner") ;
//   DEBUG_MESSAGE(*p) ;
  return p ;
}

void trHessian::setElement(unsigned long i, 
			   unsigned long j, 
			   patReal x,
			   patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  matrix.setElement(i,j,x,err) ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
}

void trHessian::addElement(unsigned long i, 
			   unsigned long j, 
			   patReal x,
			   patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  matrix.addElement(i,j,x,err) ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
}

void trHessian::multElement(unsigned long i, 
			   unsigned long j, 
			   patReal x,
			   patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  matrix.multElement(i,j,x,err) ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
}


patReal trHessian::getElement(unsigned int i, 
			   unsigned int j, 
			   patError*& err) {

  return matrix.getElement(i,j,err) ;
  
}




ostream& operator<<(ostream &str, const trHessian& x) {

  str << x.matrix ;

  return str ;
}

patMyMatrix trHessian::getMatrixForLinAlgPackage(patError*& err) const {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patMyMatrix(0,0);
  }
  
  patMyMatrix result(matrix.getSize(),matrix.getSize()) ;
  for (unsigned long i = 0 ; i < matrix.getSize() ; ++i) {
    for (unsigned long j = 0 ; j < matrix.getSize() ; ++j) {
      result[i][j] = matrix.getElement(i,j,err) ;
    }
  }
  return result ;
}

trHessian* trHessian::getReducedHessian(vector<trBounds::patActivityStatus> 
					status,
					patError*& err) {
  

//   DEBUG_MESSAGE("Get reduced hessian") ;
//   DEBUG_MESSAGE("Full hessian = " << *this) ;
  if (matrix.getSize() != status.size()) {
    stringstream str ;
    str << "Incompatibility between matrix size ( " << matrix.getSize() 
	<< ") and status size ( " << status.size() << ")" << '\0' ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL;
  }
  
  unsigned long nbrFree = 0;
  
  for (vector<trBounds::patActivityStatus>::iterator i = status.begin() ;
       i != status.end() ;
       ++i) {
    if (*i == trBounds::patFree) {
      ++nbrFree  ;
    }
  }

  if (nbrFree == status.size()) {
    //    DEBUG_MESSAGE("Full dimension") ;
    return this ;
  }

  if (submatrix != NULL) {
    DELETE_PTR(submatrix) ;
  }

  submatrix = new trHessian(theParameters,nbrFree) ;

  unsigned long indexi = 0 ;
  for (unsigned long i = 0 ; i < status.size() ; ++i) {
    if (status[i] == trBounds::patFree) {
      unsigned long indexj = 0 ;
      for (unsigned long j = 0 ; j <= i ; ++j) {
	if (status[j] == trBounds::patFree) {
	  patReal elem = matrix.getElement(i,j,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL;
	  }
	  submatrix->matrix.setElement(indexi, 
				       indexj,
				       elem,
				       err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL;
	  }
	  ++indexj ;
	}
      }
      ++indexi ;
    }
  }

//   DEBUG_MESSAGE("Reduced matrix") ;
//   DEBUG_MESSAGE(*submatrix) ;

  return submatrix ;
}

trMatrixVector* 
trHessian::getReduced(vector<trBounds::patActivityStatus> status,
		   patError*& err) {
  trHessian* ptr = getReducedHessian(status,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return ptr ;

  
  
}

unsigned long trHessian::getDimension() const  {
  return matrix.getSize() ;
}

void trHessian::add(patReal alpha, const trHessian& M, patError*& err) {
  if (getDimension() != M.getDimension()) {
    stringstream str ;
    str << "Cannot add a " << getDimension() << "x" << getDimension() 
	<< " matrix with a " << M.getDimension() << "x" 
	<< M.getDimension() << " matrix" << '\0' ;
    err = new patErrMiscError(err->describe()) ;
    WARNING(err->describe()) ;
    return ;
  }

  if (alpha == 1.0) {
    matrix.add(M.matrix,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
  else {
    matrix.addAlpha(alpha,M.matrix,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
  }
  }

}

void trHessian::set(const trHessian& M, patError*& err) {
  if (getDimension() != M.getDimension()) {
    stringstream str ;
    str << "Cannot set a " << getDimension() << "x" << getDimension() 
	<< " matrix from a " << M.getDimension() << "x" 
	<< M.getDimension() << " matrix" ;
    err = new patErrMiscError(err->describe()) ;
    WARNING(err->describe()) ;
    return ;
  }

  matrix.set(M.matrix,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}




patBoolean trHessian::correctForSingularity(int svdMaxIter, 
					    patReal threshold, 
					    patError*& err) {
  matrix.correctForSingularity(svdMaxIter,threshold,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patFALSE ;
  }
  return patTRUE ;
}
  

void trHessian::updatePenalty(patReal singularityThreshold, 
			      const trVector& step,patError*& err) {
  matrix.updatePenalty(singularityThreshold,step,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}

void trHessian::print(ostream& os) {
  os << matrix ;
}

void trHessian::setToZero() {
  matrix.init(0.0) ;
}

void trHessian::setToIdentity(patError*& err) {
  matrix.init(0.0) ;
  for (patULong k = 0 ; k < getDimension() ; ++k) {
    matrix.setElement(k,k,1.0,err) ;  
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
}


void trHessian::changeSign() {
  matrix.multAlpha(-1.0);
}

void trHessian::resize(patULong size) {
  matrix.resize(size) ;
}

void trHessian::multAllEntries(patReal x,
			       patError*& err) {
  matrix.multAlpha(x) ;
}

