#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include <algorithm>
#include "patMath.h"
#include "patQR.h"
#include "patErrMiscError.h"
#include "patErrOutOfRange.h"
#include "patErrNullPointer.h"

patQR::patQR(patMyMatrix *_patA): 
  A(_patA), 
  Q(NULL), 
  rank(0), 
  factorized(patFALSE), 
  qComputed(patFALSE) {
  if (A != NULL) {
    perm.resize(A->nCols()) ;
    gamma.resize(A->nCols()) ;
  }
}

void patQR::setMatrix(patMyMatrix *_patA) {
  A = _patA ;
  if (A != NULL) {
    perm.resize(A->nCols()) ;
    gamma.resize(A->nCols()) ;
  }
  factorized = patFALSE ;
}

patMyMatrix* patQR::getR() const  {
  if (factorized) {
    return A ;
  }
  else {
    return NULL ;
  }
}

patMyMatrix* patQR::computeQR(patError*& err) {

  if (factorized) {
    return A ;
  }
  if (err != NULL) {
    WARNING(err->describe()) ;
    return A ;
  }
  if (A == NULL) {
    err = new patErrNullPointer("patMyMatrix") ;
    WARNING(err->describe()) ;
    return A ;
  }
  unsigned long m = A->nRows() ;
  if (m <= 0) {
    stringstream str ;
    str << "Matrix A of size " 
	<< m ; 
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return A ;
  }
  unsigned long n = A->nCols() ; 

  perm.resize(n) ;
  for (unsigned long k = 0 ; k < n ; ++k) {
    perm[k] = k ;
    gamma[k] = 0.0 ;
    for (unsigned long l = 0 ; l < m ; ++l) {
      gamma[k] += (*A)[l][k] * (*A)[l][k];
    }
  }
  rank = 0 ;
  unsigned long k;

  pk.erase(pk.begin(),pk.end()) ;

  for ( k = 0 ; (k < n) && (m-k > 1); ++k) {
    // Find the max element in gamma
    unsigned long maxg = max_element(gamma.begin()+k,gamma.end()) - gamma.begin() ;
    if (patAbs(gamma[maxg]) <= patEPSILON) {
      WARNING(gamma[maxg] << " too small. Considered as zero") ;
      rank = k ;
      DEBUG_MESSAGE("RANK="<<rank) ;
      factorized = patTRUE ;
      DEBUG_MESSAGE("FRANK comme gamma est trop petit il retourne la matrice A suivante : "<< *A ) ;
      return A ;
     
    }
    if (maxg != k) {
      // Swap maxg and k ;
      swap(perm[k],perm[maxg]) ;
      swap(gamma[k],gamma[maxg]) ;
      for (unsigned long ll = 0 ; ll < m ; ++ll) {
	swap((*A)[ll][k],(*A)[ll][maxg]) ;
      }
    }
    patVariables x ;
    for (unsigned long ll = 0 ; ll < m ;++ll) {
      x.push_back((*A)[ll][k]) ;
    }
    patHouseholder P ;
    P.setToNullify(x,k,m-1,patTRUE,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return A ;
    }
    
    if (P.isIdentity()) {
      // DEBUG_MESSAGE("RANK="<<rank) ;
      factorized = patTRUE ;
      return A ;
     }
    //Successful Householer transformation. The rank estimation can be
    //increased
    ++rank ;

    pk.push_back(P) ;
    
    P.multiply(A,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return A ;
    }
    // Update gamma
    for (unsigned long ll = k+1 ; ll < n ; ++ll) {
      gamma[ll] -= (*A)[k][ll] * (*A)[k][ll];
    }
  }

  // At this point, either k == n or k == m-1 (or both) In the latter case,
  // the rank estimation did not consider the last column.  If the largest
  // element on the last row is not 0, we increase the rank.
  if (k < n) {
    patVariables lastRow = A->getRow(m-1) ;
    unsigned long maxe =
      max_element(lastRow.begin(),lastRow.end()) - lastRow.begin() ;
    //    unsigned long maxe =
    //  max_element(&(*A)[m-1][m-1],(*A)[m-1].end()) - (*A)[m-1].begin() ;
    DEBUG_MESSAGE("Max element of the last row=" << (*A)[m-1][maxe]) ;
    DEBUG_MESSAGE("epsilon=" <<patEPSILON) ;
    if (m-1 >= n) {
      err = new patErrOutOfRange<unsigned long>(m-1,0,n-1) ;
      WARNING(err->describe()) ;
      return A ;
    }
    if (patAbs((*A)[m-1][maxe]) >= patEPSILON) {
      ++rank ;
    }
  }
  // DEBUG_MESSAGE("RANK="<<rank) ;
  factorized = patTRUE ;
  return A ;
}

pair<patMyMatrix*,patMyMatrix*> patQR::QR(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    pair<patMyMatrix*,patMyMatrix*> dummy(A,NULL) ;
    return dummy ;
  }
  if (factorized || qComputed) {
    return pair<patMyMatrix*,patMyMatrix*>(A,Q) ;
  }
  if (A == NULL) {
    err = new patErrNullPointer("patMyMatrix") ;
    WARNING(err->describe()) ;
    return pair<patMyMatrix*,patMyMatrix*>(A,NULL) ;
  }
  computeQR(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return pair<patMyMatrix*,patMyMatrix*>(A,NULL) ;
  }
  unsigned long dim = A->nRows() ;
  Q = new patMyMatrix(dim,dim,0.0) ;
  for (unsigned long i = 0 ;
       i < dim ;
       ++i) {
    (*Q)[i][i] = 1.0 ;
  }
  for (vector<patHouseholder>::iterator p = pk.begin() ;
       p != pk.end() ;
       ++p) {
    (*p).multiply(Q,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return pair<patMyMatrix*,patMyMatrix*>(A,Q) ;
    }
  }

  qComputed = patTRUE ;

  return pair<patMyMatrix*,patMyMatrix*>(A,Q) ;
    
}

patVariables patQR::Qtimes(const patVariables& x,patError*& err) const {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return x ;
  }
  patVariables res(x) ;
  for (vector<patHouseholder>::const_iterator p = pk.begin() ;
       p != pk.end() ;
       ++p) {
    (*p).multiply(&res,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return x ;
    }
  }
  return res;
  
}
