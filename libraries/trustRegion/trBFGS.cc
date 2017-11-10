//-*-c++-*------------------------------------------------------------
//
// File name : trBFGS.cc
// Author :    Michel Bierlaire
// Date :      Tue Jan 25 11:24:36 2000
//
//--------------------------------------------------------------------

#include <numeric>
#include "patMath.h"
#include "patErrNullPointer.h"
#include "trBFGS.h"
#include "trSchnabelEskow.h"

trBFGS::trBFGS(unsigned long size, 
	       trParameters theParameters,
	       patError*& err) :
  trSecantUpdate(size,theParameters,err) ,
  submatrix(NULL) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}

trBFGS::trBFGS(const trVector& x, 
	       trParameters theParameters,
	       patError*& err) :
  trSecantUpdate(x,theParameters,err),
  submatrix(NULL) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}

trBFGS::trBFGS(const patHybridMatrix& x, 
	       trParameters theParameters,
	       patError*& err ):
  trSecantUpdate(x,theParameters,err),
  submatrix(NULL) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}

trBFGS::~trBFGS() {
  if (submatrix != NULL) {
    DELETE_PTR(submatrix) ;
  }
}

void trBFGS::update(const trVector& sk,
		    const trVector& currentGradient,
		    const trVector& previousGradient,
		    ostream& str,
		    patError*& err) {


  // yk is the difference of gradients
  // sk is the difference of points

  trVector yk = currentGradient-previousGradient ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return  ;
  }

  //  DEBUG_MESSAGE("yk=" << yk) ;

  trVector Hs =  (*this)(sk,err) ;

  patReal normSk = sqrt(inner_product(sk.begin(),sk.end(),sk.begin(),0.0)) ;
  patReal normYk = sqrt(inner_product(yk.begin(),yk.end(),yk.begin(),0.0)) ;
  patReal sHs = inner_product(sk.begin(),sk.end(),Hs.begin(),0.0) ;
  patReal ys = inner_product(yk.begin(),yk.end(),sk.begin(),0.0) ;

//    DEBUG_MESSAGE("normSk=" << normSk) ;
//    DEBUG_MESSAGE("normYk=" << normYk) ;
//    DEBUG_MESSAGE("ys=" << ys) ;
//    DEBUG_MESSAGE("sHs=" << sHs) ;
  if (normSk < patEPSILON || 
      normYk < patEPSILON || 
      patAbs(ys) < patEPSILON * normSk * normYk) {
    str << "% " ;
    return ;
  }
  
  static patReal eta = 
    pow(10.0,
	patReal(-theParameters.significantDigits)) ;

  patBoolean skipUpdate = patTRUE ;
  for (unsigned long l = 0 ; l < yk.size() && skipUpdate ; ++l) {
    if (patAbs(yk[l]-Hs[l]) >= eta * patMax(patAbs(currentGradient[l]),
					    patAbs(previousGradient[l]))) {
      skipUpdate = patFALSE ;
    }
  }

  if (skipUpdate) {
    str << "# " ;
    return ;
  }

  str << "  " ;

  for (unsigned long i = 0 ; i < matrix.getSize() ; ++i) {
    for (unsigned long j = i ; j < matrix.getSize() ; ++j) {
      matrix.addElement(i,j,yk[i]*yk[j]/ys,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return  ;
      }

      matrix.addElement(i,j,-Hs[i]*Hs[j]/sHs,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return  ;
      }

    }
  }

}

trBFGS* trBFGS::getReducedHessian(vector<trBounds::patActivityStatus> 
				 status,
				 patError*& err) {
  
  if (matrix.getSize() != status.size()) {
    stringstream str ;
    str << "Incompatibility between matrix size ( " << matrix.getSize() 
	<< ") and status size ( " << status.size() << ")" << '\0' ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL ;
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
    return this ;
  }
  
  list<vector<patReal>::size_type> indexList ;
  
  for (unsigned long i = 0 ;
       i < status.size() ;
       ++i) {
    if (status[i] == trBounds::patFree) {
      indexList.push_back(i) ;
    }
  }

  patHybridMatrix* mPtr = matrix.getSubMatrix(indexList,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }

  if (submatrix != NULL) {
    DELETE_PTR(submatrix) ;
  }

  submatrix = new trBFGS(*mPtr,
			 theParameters,
			 err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  return submatrix ;
  
}


patString trBFGS::getUpdateName() const {
  return patString("BFGS") ;
}


patReal trBFGS::getElement(unsigned int i, unsigned int j, patError*& err) const {

  return matrix.getElement(i,j,err) ;

}

void trBFGS::print(ostream& os) {
  os << matrix ;
}
