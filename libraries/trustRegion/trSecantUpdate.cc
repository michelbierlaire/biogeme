//-*-c++-*------------------------------------------------------------
//
// File name : trSecantUpdate.cc
// Author :    Michel Bierlaire
// Date :      Tue Jan 25 11:24:36 2000
//
//--------------------------------------------------------------------

#include <numeric>
#include "patMath.h"
#include "patErrNullPointer.h"
#include "trSecantUpdate.h"
#include "trSchnabelEskow.h"

trSecantUpdate::trSecantUpdate() : matrix(0) {
  WARNING("Should never be called") ;
}

trSecantUpdate::trSecantUpdate(unsigned long size, 
			       trParameters p,
			       patError*& err) :
  matrix(trVector(size,1.0),err),
  theParameters(p) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  matrix.setType(patHybridMatrix::patSymmetric) ;
}

trSecantUpdate::trSecantUpdate(const trVector& x, 
			       trParameters p,
			       patError*& err) :
  matrix(x,err),
  theParameters(p) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  matrix.setType(patHybridMatrix::patSymmetric) ;
  DEBUG_MESSAGE("First matrix") ;
  cout << matrix  << endl ;
}

trSecantUpdate::trSecantUpdate(const patHybridMatrix& x, 
			       trParameters p,
			       patError*& err ):
  matrix(x),
  theParameters(p) {
}

trSecantUpdate::~trSecantUpdate() {
}

trVector trSecantUpdate::operator()(const trVector& x, 
			    patError*& err)  {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trVector() ;
  }

  if (!matrix.isSymmetric()) {
    err = new patErrMiscError("Quasi-Newton approx. must be a symmetric matrix") ;
    WARNING(err->describe()) ;
    return trVector() ;
  }

  if (matrix.getSize() != x.size()) {
    stringstream str ;
    str << "Incompatible sizes: matrix is " << matrix.getSize() 
	<< " x is " << x.size() << '\0' ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return trVector() ;
  }

  // I know a smarter way to implement this, but I have to rush. Hopefully,
  // I'll have time to optimize it later on.
  // Or should I write it quickly in the margin ;-)

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

patBoolean trSecantUpdate::providesPreconditionner() const {
  return patTRUE ;
}

trPrecond* 
trSecantUpdate::createPreconditionner(patError*& err) const {
  //  DEBUG_MESSAGE("### Create preconditionner") ;
  trSchnabelEskow* p = new trSchnabelEskow(matrix) ;
  if (p == NULL) {
    err = new patErrNullPointer("trPRecond") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  //  DEBUG_MESSAGE("### Factorize preconditionner") ;
  p->factorize(theParameters.toleranceSchnabelEskow,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return p ;
}


 trMatrixVector* 
 trSecantUpdate::getReduced(vector<trBounds::patActivityStatus> status,
 		      patError*& err)  {
 
   trSecantUpdate* ptr = getReducedHessian(status,err) ;
   if (err != NULL) {
     WARNING(err->describe()) ;
     return NULL;
   }
   return ptr ;
 }

patBoolean trSecantUpdate::correctForSingularity(int svdMaxIter,
				   patReal threshold,
				   patError*& err) {
  matrix.correctForSingularity(svdMaxIter,threshold,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patFALSE ;
  }
  return patTRUE ;
}
  

void trSecantUpdate::updatePenalty(patReal singularityThreshold, // patParameters::the()->BTRSingularityThreshold()
				   const trVector& step,
				   patError*& err) {
  matrix.updatePenalty(singularityThreshold,step,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}
