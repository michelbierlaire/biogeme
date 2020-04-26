//-*-c++-*------------------------------------------------------------
//
// File name : trSR1.cc
// Author :    Michel Bierlaire
// Date :      Tue Apr 24 10:39:59 2001
//
//--------------------------------------------------------------------

#include <numeric>
#include "patMath.h"
#include "patErrNullPointer.h"
#include "trSR1.h"
#include "trSchnabelEskow.h"

trSR1::trSR1(trParameters theParameters,
	     patError*& err) : 
  trSecantUpdate(0,theParameters,err),
  submatrix(NULL) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  err = new patErrMiscError("Default ctor should not be called") ;
  WARNING(err->describe()) ;
  return ;
}

trSR1::trSR1(unsigned long size, 
	     trParameters theParameters,
	     patError*& err) :
  trSecantUpdate(size,theParameters,err),
  submatrix(NULL) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}

trSR1::trSR1(const trVector& x, 
	     trParameters theParameters,
	     patError*& err) :
  trSecantUpdate(x,theParameters,err),
  submatrix(NULL) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}

trSR1::trSR1(const patHybridMatrix& x,
	     trParameters theParameters,
	     patError*& err ):
  trSecantUpdate(x,theParameters,err),
  submatrix(NULL) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}

trSR1::~trSR1() {
  if (submatrix != NULL) {
    DELETE_PTR(submatrix) ;
  }
}
void trSR1::update(const trVector& sk,
		   const trVector& currentGradient,
		   const trVector& previousGradient,
		   ostream& str ,
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

  trVector yMinusHs = yk - Hs ;

  patReal denom = inner_product(yMinusHs.begin(),yMinusHs.end(),sk.begin(),0.0);

  patReal normSk = sqrt(inner_product(sk.begin(),sk.end(),sk.begin(),0.0)) ;
  patReal normYk = sqrt(inner_product(yk.begin(),yk.end(),yk.begin(),0.0)) ;
  patReal ys = inner_product(yk.begin(),yk.end(),sk.begin(),0.0) ;

//    DEBUG_MESSAGE("normSk=" << normSk) ;
//    DEBUG_MESSAGE("normYk=" << normYk) ;
//    DEBUG_MESSAGE("ys=" << ys) ;
//    DEBUG_MESSAGE("sHs=" << sHs) ;
  if (normSk < patEPSILON || 
      normYk < patEPSILON || 
      patAbs(denom) < patEPSILON ||
      patAbs(ys) < patEPSILON * normSk * normYk) {
    str << "% " ;
    return ;
  }
  
  static patReal eta = 
    pow(10.0,patReal(-theParameters.significantDigits)) ;

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
      matrix.addElement(i,j,yMinusHs[i]*yMinusHs[j]/denom,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return  ;
      }
    }
  }

}


trSR1* trSR1::getReducedHessian(vector<trBounds::patActivityStatus> 
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

  submatrix = new trSR1(*mPtr,theParameters,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  return submatrix ;
  
}

patString trSR1::getUpdateName() const {
  return patString("Symmetrix Rank One") ;
}


patReal trSR1::getElement(unsigned int i, unsigned int j, patError*& err) const {

  return matrix.getElement(i,j,err) ;

}


void trSR1::print(ostream& os) {
  os << matrix ;
}
