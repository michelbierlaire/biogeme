//-*-c++-*------------------------------------------------------------
//
// File name : trHybridMatrix.cc
// Author :    Michel Bierlaire
// Date :      Thu Jun  8 17:00:11 2000
//
// This object encapsulates patHybridMAtrix.h to comply with the 
// trMatrixVector interface
//--------------------------------------------------------------------

#include "trHybridMatrix.h"

trHybridMatrix::trHybridMatrix(patHybridMatrix* _mPtr)  : 
  hMatrix(_mPtr), submatrix(NULL) {

}

trHybridMatrix::~trHybridMatrix() {
  if (submatrix != NULL) {
    DELETE_PTR(submatrix) ;
  }
}

trVector trHybridMatrix::operator()(const trVector& x, 
		    patError*& err)  {
  
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trVector();
  }
 
  trVector::size_type dim = x.size() ;

  if (dim != hMatrix->getSize()) {
    err = new patErrMiscError("Incompatible sizes in matrix vector mult.") ;
    WARNING(err->describe()) ;
    return trVector() ;
  }

  trVector res(dim,0.0) ;

  for (trVector::size_type row = 0 ; row < dim ; ++row) {

     for (trVector::size_type col = 0 ; col < dim ; ++col ) {
       res[row] += hMatrix->getElement(row,col,err) * x[row] ;
     } 
  } 
  return res ;
}

patBoolean trHybridMatrix::providesPreconditionner() const {
  return patFALSE ;
}

trPrecond* 
trHybridMatrix::createPreconditionner(patError*& err) const {
  return NULL ;
}

trMatrixVector* 
trHybridMatrix::getReduced(vector<trBounds::patActivityStatus> status,
			   patError*& err) {
  
  if (hMatrix->getSize() != status.size()) {
    stringstream str ;
    str << "Incompatibility between matrix size ( " << hMatrix->getSize() 
	<< ") and status size ( " << status.size() << ")" << '\0' ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  list<vector<patReal>::size_type> indexList ;
  
  for (unsigned long i = 0 ;
       i < status.size() ;
       ++i) {
    if (status[i] == trBounds::patFree) {
      indexList.push_back(i) ;
    }
  }

  patHybridMatrix* mPtr = hMatrix->getSubMatrix(indexList,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  
  if (submatrix != NULL) {
    DELETE_PTR(submatrix) ;
  }

  submatrix = new trHybridMatrix(mPtr) ;

  return submatrix ;
}


patBoolean trHybridMatrix::correctForSingularity(int svdMaxIter, 
						 patReal threshold, 
						 patError*& err) {
  hMatrix->correctForSingularity(svdMaxIter,threshold,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patFALSE ;
  }
  return patTRUE ;
}
  

void trHybridMatrix::updatePenalty(patReal singularityThreshold, 
				   const trVector& step,
				   patError*& err) {
  hMatrix->updatePenalty(singularityThreshold,step,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}

void trHybridMatrix::print(ostream& os) {
  if (hMatrix == NULL) {
    return ;
  }
  os << *hMatrix ;
}
