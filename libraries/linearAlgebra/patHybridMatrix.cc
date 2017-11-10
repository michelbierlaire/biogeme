#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <cmath>
#include <fstream>
#include <cassert>
#include <numeric>

#include "patErrOutOfRange.h"
#include "patErrNullPointer.h"
#include "patMyMatrix.h"
#include "patHybridMatrix.h"
#include "patMath.h"
#include "patEigenVectors.h"

patHybridMatrix::patHybridMatrix() {
  WARNING("Default ctor should never be called") ;

}

patHybridMatrix::patHybridMatrix(const patHybridMatrix& h) 
  : data(h.data),
    type(h.type),
    dim(h.dim),
    pivot(h.pivot),
    pivotInverse(h.pivotInverse),
    submatrix(NULL) ,
    singularityPenalty(0.0),
    Q(NULL)
{
  if (h.submatrix != NULL) {
    submatrix = new patHybridMatrix(*(h.submatrix)) ;
  }
}


patHybridMatrix::patHybridMatrix(const vector<patReal>& diag,
				 patError*&  err) : 
  data(diag.size()*(diag.size()+1)/2 , 0.0),
  type(patDiagonal),
  dim(diag.size()),
  pivot(diag.size()*(diag.size()+1)/2),
  pivotInverse(diag.size()*(diag.size()+1)/2),
  submatrix(NULL),
  singularityPenalty(0.0),
  Q(NULL) {
  
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  for (vector<patReal>::size_type i = 0 ; i < diag.size() ; ++i) {
    setElement(i,i,diag[i],err) ;
  }
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  if (data.size() == 0) {

    err = new patErrMiscError("Data size null at construction") ;
    WARNING(err->describe()) ;
    return ;
    
  }
}


patHybridMatrix::patHybridMatrix(vector<patReal>::size_type size) : 
  data(size*(size+1)/2),
  type(patSymmetric),
  dim(size),
  pivot(size),
  pivotInverse(size),
  submatrix(NULL),
  singularityPenalty(0.0),
  Q(NULL)
{  

  for (patVariables::size_type i = 0 ; i < size ; ++i) {
    pivot[i] = pivotInverse[i] = i ;
  }
  //    copy(pivot.begin(),pivot.end(),
  //	 ostream_iterator<patVariables::size_type>(cout," ")) ;
  
} ;


patHybridMatrix::patHybridMatrix(vector<patReal>::size_type size, patReal init) :
  data(size*(size+1)/2),
  type(patSymmetric),
  dim(size),
  pivot(size),
  pivotInverse(size),
  submatrix(NULL),
  singularityPenalty(0.0),
  Q(NULL) {
  for (patVariables::size_type i = 0 ; i < size ; ++i) {
    data[i] = init ;
  }
  for (patVariables::size_type i = 0 ; i < size ; ++i) {
    pivot[i] = pivotInverse[i] = i ;
  }
}


void patHybridMatrix::resize(patULong size) {
  data.resize(size*(size+1)/2) ;
  dim = size ;
  pivot.resize(size) ;
  pivotInverse.resize(size) ;
} 

patHybridMatrix::~patHybridMatrix() {
  if (!data.empty()) {
    data.erase(data.begin(),data.end()) ;
  }
  if (submatrix != NULL) {
    DELETE_PTR(submatrix) ;
  }
  DELETE_PTR(Q) ;
  
}


void patHybridMatrix::init(const patReal& x) {
  fill(data.begin(),data.end(),x) ;
}


void patHybridMatrix::addElement(vector<patReal>::size_type i, 
				 vector<patReal>::size_type j, 
				 const patReal& x,
				 patError*&  err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return; 
  }
  if (index(i,j) >= data.size()) {
    stringstream str ;
    str << "index(" << i << "," << j << ")=" 
	<< index(i,j) << " size=" << data.size()  ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  assert(index(i,j) >= 0) ;
  if (!isfinite(x)) {
    stringstream str ;
    str << "Cannot add an element with value: " << x ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
  }
  if (!isfinite(data[index(i,j)]+x)) {
    stringstream str ;
    str << "Sum not finite: " << data[index(i,j)] << "+" << x << "=" << data[index(i,j)]+x ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
  }
  data[index(i,j)] += x ;
} 

void patHybridMatrix::multElement(vector<patReal>::size_type i, 
				 vector<patReal>::size_type j, 
				 const patReal& x,
				 patError*&  err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return; 
  }
  if (index(i,j) >= data.size()) {
    stringstream str ;
    str << "index(" << i << "," << j << ")=" 
	<< index(i,j) << " size=" << data.size()  ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  assert(index(i,j) >= 0) ;
  data[index(i,j)] *= x ;
} 


  patReal patHybridMatrix::getElement(vector<patReal>::size_type i, 
		     vector<patReal>::size_type j,
		     patError*&  err) const {
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    switch (type) {
    case patDiagonal :
      if (i != j) return patReal(0) ;
      if (index(i,j) >= data.size()) {
	stringstream str ;
	str << "index(" << i << "," << j << ")=" 
	    << index(i,j) << " size=" << data.size() ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return patReal();
      }
      return data[index(i,j)] ;
    case patSymmetric :
      if (index(i,j) >= data.size()) {
	stringstream str ;
	str << "index(" << i << "," << j << ")=" 
	    << index(i,j) << " size=" << data.size() ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return patReal();
      }
      return data[index(i,j)] ;
    case patLower :
      if (i >= j) {
	if (index(i,j) >= data.size()) {
	  stringstream str ;
	  str << "index(" << i << "," << j << ")=" 
	      << index(i,j) << " size=" << data.size() ;
	  err = new patErrMiscError(str.str()) ;
	  WARNING(err->describe()) ;
	  return patReal();
	}
      return data[index(i,j)] ;
      }
      else {
	return patReal(0) ;
      }
    case patUpper :
      if (i <= j) {
	if (index(i,j) >= data.size()) {
	  WARNING("Element ("<<i<<","<<j<<") out of bounds data["<<index(i,j)<<"]") ;
	}
	return data[index(i,j)] ;
      }
      else {
	return patReal(0) ;
      }
    }
    // This statement should never be reached. It has been added to please the
    // compiler.
    return 0 ;
  }

patReal patHybridMatrix::operator()(vector<patReal>::size_type i,
				    vector<patReal>::size_type j,
				    patError*&  err) const {
  return getElement(i,j,err) ;
}



void patHybridMatrix::setElement(vector<patReal>::size_type i, 
				 vector<patReal>::size_type j, 
				 const patReal& x,
				 patError*&  err) {
    if (err != NULL) {
      WARNING(err->describe()) ;
      return; 
    }
    if (index(i,j) >= data.size()) {
      stringstream str ;
      str << "index(" << i << "," << j << ")=" 
	  << index(i,j) << " size=" << data.size() ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
    if (!isfinite(x)) {
      stringstream str ;
      str << "Cannot set an element with value: " << x ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
    }
    assert(index(i,j) >= 0) ;
    data[index(i,j)] = x ;
  } 

patBoolean patHybridMatrix::cholesky(patReal tolSchnabelEskow,
				     patError*&  err) {

  if (err != NULL) {
    WARNING(err->describe()) ;
    return patTRUE ;
  }

  patBoolean defPos = patTRUE ;
  patBoolean phase1 = patTRUE ;

  patReal delta = 0.0 ;
  patVariables::size_type n = getSize() ;

  patHybridMatrix copyForDebug = *this ;

  //  DEBUG_MESSAGE("Init pivot") ;
  for (patVariables::size_type i = 0 ; i < n ; i++) {
    pivot[i] = i ;
  }
 
  // Find the diagonal element with the largest norm. If a negative element
  // isfound, end of phase 1.

  patReal tmp=data[0] ;
  patReal gamma = patAbs(tmp) ;
  
  if (tmp < 0.0) {
    DEBUG_MESSAGE("Phase 2 : the first diagonal element is negative") ;
    phase1 = patFALSE ;
  }
  
  for (patVariables::size_type i = 1 ; i < n ; i++) {
    tmp = data[index(i,i)] ;
    if (patAbs(tmp) > gamma)  {
      gamma = patAbs(tmp) ;
    }
    if (tmp < 0.0) {
      //      DEBUG_MESSAGE("Phase 2: the largest diagonal element is negative") ;
      phase1 = patFALSE ;
    }
  }

  patReal taugamma = 
    patMax(tolSchnabelEskow * gamma,patEPSILON) ;


  // If not in phase 1, compute Gershgorin bounds, need in phase 2

  patVariables g ;
  if (!phase1) {
    g = calcgersch(0) ;
  }

  for (patVariables::size_type j=0 ; j < n-1 ; j++) {

    patVariables::size_type jj = index(j,j) ;
    
    if (phase1) {

      // Find index of largest element among those such that  i>=j 
      patReal maxd  = data[jj] ;
      patVariables::size_type imaxd = j ;
      for (patVariables::size_type i = j+1 ; i < n ; i++) {
	tmp = data[index(i,i)] ;
	if (maxd < tmp) {
	  maxd = tmp ;
	  imaxd = i ;
	}
      }

      // Pivoting

      if (imaxd != j) {

	// Swap row j and row with largest element

	for (patVariables::size_type i = 0 ; i < j ; ++i) {
	  patVariables::size_type iswap1 = index(j,i) ;
	  patVariables::size_type iswap2 = index(imaxd,i) ;
	  patReal temp = data[iswap1] ;
	  data[iswap1] = data[iswap2] ;
	  data[iswap2] = temp ;
	}

	// Swap column j and row between j and maxdiag
	for (patVariables::size_type i = j+1 ; i < imaxd ; i++) {
	  patVariables::size_type iswap1    = index(i,j)     ;
	  patVariables::size_type iswap2    = index(imaxd,i) ;
	  patReal temp      = data[iswap1]            ;
	  data[iswap1] = data[iswap2]            ;
	  data[iswap2] = temp                 ;
	}

	// Swap column j and column with largest element

	for (patVariables::size_type i = imaxd+1 ; i < n ; i++) {
	  patVariables::size_type iswap1    = index(i,j)     ;
	  patVariables::size_type iswap2    = index(i,imaxd) ;
	  patReal temp      = data[iswap1]            ;
	  data[iswap1] = data[iswap2]            ;
	  data[iswap2] = temp                 ;
	}
	
	// Swap diagonal elements

	patVariables::size_type iswap2    = index(imaxd,imaxd) ;
	patReal temp      = data[jj]                    ;
	data[jj]     = data[iswap2]                ;
	data[iswap2] = temp                     ;

	// Swap pivot elements
	//	DEBUG_MESSAGE("Swap pivot elements") ;
	
	patVariables::size_type itemp  = pivot[j]     ;
	pivot[j]     = pivot[imaxd] ;
	pivot[imaxd] = itemp        ;
      }

      // Check that a normalCholesky update would lead to a positive
      // diagonal. If not, go to phase 2.

      patVariables::size_type jp1 = j+1 ;
      patReal jdmin = patMaxReal ; 

      if (data[jj] > 0) {
	for (patVariables::size_type i=jp1 ; i < n ; i++) {
	  patVariables::size_type ij = index(i,j) ;
	  patReal temp = data[ij] * data[ij] / data[jj] ;
	  patReal tdmin = data[index(i,i)] - temp ;
	  
	  if (i != jp1) 
	    jdmin = patMin(jdmin,tdmin) ;
	  else
	    jdmin = tdmin ;
	}
	
	if (jdmin < taugamma)  {
	  phase1 = patFALSE ;
	}
       
      }
      else {
	DEBUG_MESSAGE("Phase 2: negative diagogal element:" << data[jj]) ;
	phase1 = patFALSE ;
      }
      
      if (phase1) {
	// Perform normal Cholesky update

	data[jj] = sqrt(data[jj]) ;
	for (patVariables::size_type i = j+1 ; i < n ; i++) {
	  data[index(i,j)] /=  data[jj] ;
	}
	for (patVariables::size_type i = j+1 ; i < n ; i++) {
	  for (patVariables::size_type k = j+1 ; k <= i ; k++) {
	    data[index(i,k)] -= data[index(i,j)] * data[index(k,j)] ;
	  } 
	}
	if (j == n-2) {
	  patVariables::size_type itemp = index(n-1,n-1) ;
	  data[itemp] = sqrt(data[itemp]) ;
	}
      }
      else {
                                        /* Calculer les bornes inferieures  */
                                        /* de gershgorin, changees de signe */

	g = calcgersch(j) ;
      }
    }

                                        /* PHASE 2                          */

    if (!phase1) {
      if (j != n-2) {

	// Compute the most negative Gershgorin bound
	
	patReal ming = 0 ;
	patVariables::size_type iming = 0;

	for (patVariables::size_type i=j ; i < n ; i++) {
	  if (i != j) {
	    if (ming > g[i]) {
	      ming = g[i] ;
	      iming = i ;
	    }
	  }
	  else {
	    iming = j    ;
	    ming  = g[j] ;
	  }
	}

	// Pivot upwards row and column corresponding to that bound

	if (iming != j) {
	  // Swap row j and gersh

	  for (patVariables::size_type i=0 ; i < j ; i++) {
	    patVariables::size_type iswap1    = index(j,i)     ;
	    patVariables::size_type iswap2    = index(iming,i) ;
	    patReal temp      = data[iswap1]            ;
	    data[iswap1] = data[iswap2]            ;
	    data[iswap2] = temp                 ;
	  }

	  // Swap column j and row gersh between j and iming

	  for (patVariables::size_type i=j+1 ; i < iming ; i++) {
	    patVariables::size_type iswap1    = index(i,j) ;
	    patVariables::size_type iswap2    = index(iming,i) ;
	    patReal temp      = data[iswap1]            ;
	    data[iswap1] = data[iswap2]            ;
	    data[iswap2] = temp                 ;
	  }

	  // Swap column j and column gersh

	  for (patVariables::size_type i=iming+1 ; i < n ; i++) {
	    patVariables::size_type iswap1    = index(i,j) ;
	    patVariables::size_type iswap2    = index(i,iming) ;
	    patReal temp      = data[iswap1]            ;
	    data[iswap1] = data[iswap2]            ;
	    data[iswap2] = temp                 ;
	  }

	  // Swap diagonal elements

	  patVariables::size_type iswap1    = index(iming,iming) ;
	  patReal temp      = data[jj]                    ;
	  data[jj]     = data[iswap1]                ;
	  data[iswap1] = temp                     ;

	  // Swap pivot
	  //	  DEBUG_MESSAGE("Swap pivot") ;
	  
	  patVariables::size_type itemp  = pivot[j]     ;
	  pivot[j]     = pivot[iming] ;
	  pivot[iming] = itemp        ;

	  // Swap gershgorin bounds

	  temp     = g[j]     ;
	  g[j]     = g[iming] ;
	  g[iming] = temp     ;  

	}

	// Compute the perturbation to the diagonal

	patReal normj = 0.0 ;
	for (patVariables::size_type i = j+1 ; i < n ; i++)  {
	  patReal temp   = data[index(i,j)] ;
	  normj += patAbs(temp) ;
	}
	
	patReal temp    = patMax(normj,taugamma) ;
	patReal delta1  = temp - data[jj]        ;
	temp    = 0.0                ;
	delta1  = patMax(temp,delta1)    ;
	delta   = patMax(delta1,delta)   ;
/*
 *      e[j]    = delta               ;
 */
	data[jj]  += delta               ;

	if (delta != 0.0) defPos = patFALSE ;

	// Update Gershgorin bounds estimates

	if (data[jj] != normj) {
	  temp = (normj / data[jj]) - 1.0 ;
	  for (patVariables::size_type i = j+1 ; i < n ; i++) {
	    g[i] += patAbs(data[index(i,j)]) * temp ; 
	  }
	}

	//Cholesky update

	data[jj] = sqrt(data[jj]) ;
	for (patVariables::size_type i = j+1 ; i < n ; i++) 
	  data[index(i,j)] /= data[jj] ;
	
	for (patVariables::size_type i = j+1 ; i < n ; i++) 
	  for (patVariables::size_type k = j+1 ; k <= i ; k++) 
	    data[index(i,k)] -= data[index(i,j)] * data[index(k,j)] ;
	
      }
      else {
	patBoolean r = 
	  final2by2(tolSchnabelEskow,&delta,gamma) ;
	if (!r) {
	  defPos = patFALSE ;
	}
      }
    }
  }
       
  // Compute inverse pivot

  //  DEBUG_MESSAGE("Compute inverse pivot") ;
  for (patVariables::size_type i = 0 ; i < n ; i++) { 
    pivotInverse[pivot[i]] = i ;
  }


  setLower() ;



  return(defPos) ;


}

patVariables patHybridMatrix::calcgersch(patVariables::size_type j) {

  patVariables::size_type n = getSize() ;
  patVariables g(n) ;
  for (patVariables::size_type i = j ; i < n  ; i++) {
    patReal offrow = 0.0 ;
    for (patVariables::size_type k = j ; k < i ; k++) {
      offrow += patAbs(data[index(i,k)]) ;
    }
    for (patVariables::size_type k = i+1 ; k < n ; k++) { 
      offrow += patAbs(data[index(k,i)]) ;
    }
    g[i] = offrow - data[index(i,i)] ;
  }
  return g;
  
}


patBoolean patHybridMatrix::final2by2(patReal tau2,
					       patReal *delta,
					       patReal gamma) {

  patBoolean output = patTRUE  ;
  // Find eigenvalues of final 2x2 matrix

  patVariables::size_type n = getSize() ;
  patVariables::size_type i1 = index(n-2,n-2) ;
  patVariables::size_type i2 = index(n-1,n-1) ;
  patVariables::size_type i3 = index(n-1,n-2) ;

  patReal t1 = data[i1] + data[i2] ;
  patReal t2 = data[i1] - data[i2] ;
  patReal temp = t2*t2 + 4.0*data[i3]*data[i3] ;
  patReal t3 = patReal(sqrt(temp)) ;
  patReal lambda1 = (t1 - t3) / 2.0 ;
  patReal lambda2 = (t1 + t3) / 2.0 ;
  patReal lambdahi = patMax(lambda1,lambda2) ;
  patReal lambdalo = patMin(lambda1,lambda2) ;


  //Update delta

  patReal delta1 = (lambdahi-lambdalo)/(1.0-tau2) ;
  delta1 = patMax(delta1,gamma) ;
  delta1 = tau2 * delta1 - lambdalo ;
  temp = 0.0 ;
  *delta = patMax((*delta),temp) ;
  *delta = patMax(delta1,(*delta)) ;

  if (*delta > 0.0) {
    data[i1] += *delta ;
    data[i2] += *delta ;

    output = patFALSE ;
    
  }
  data[i1] = sqrt(data[i1]) ;
  data[i3] /= data[i1] ;
  data[i2] -= data[i3]*data[i3] ;
  data[i2] = sqrt(data[i2]) ;

  return(output) ;

}


ostream& operator<<(ostream &str, const patHybridMatrix& x) {
  str << endl ;
  str << "Size = " << x.getSize() << endl ;
  str << "Data size = " << x.data.size() << endl ;
  patError*  err = NULL ;
  str << "[ " ;
  for (vector<patReal>::size_type i = 0 ; i < x.getSize() ; ++i) {
    for (vector<patReal>::size_type j = 0 ; j < x.getSize() ; ++j) {
      str << x.getElement(i,j,err) << " " ; 
    }
    str<< ";" << endl ;
  }
  str << "]" << endl ;
  return (str) ;
}


patHybridMatrix* 
patHybridMatrix::getSubMatrix(list<vector<patReal>::size_type> indices,
			      patError*&  err)  {
  
  vector<patBoolean> considerVariable(getSize(),
				      patFALSE) ;


  unsigned long nbrFree = 0 ;

  for (list<vector<patReal>::size_type>::iterator i = indices.begin() ;
       i != indices.end() ;
       ++i) {
    if (*i >= getSize()) {
      err = new patErrOutOfRange<vector<patReal>::size_type>(*i,0,getSize()-1) ;
      WARNING(err->describe()) ;
      return NULL;
    }
    considerVariable[*i] = patTRUE ;
    ++nbrFree ;
  }

  if (submatrix != NULL) {
    DELETE_PTR(submatrix) ;
  }

  submatrix = new patHybridMatrix(nbrFree) ;

  submatrix->setType(*this) ;
  


  unsigned long indexi = 0 ;
  for (unsigned long i = 0 ; i < getSize() ; ++i) {
    if (considerVariable[i]) {
      unsigned long indexj = 0 ;
      for (unsigned long j = 0 ; j <= i ; ++j) {
	if (considerVariable[j]) {
	  patReal elem = getElement(i,j,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	  submatrix->setElement(indexi, 
				indexj,
				elem,
				err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	  ++indexj ;
	}
      }
      ++indexi ;
    }
  }
  return submatrix ;
  
}

void patHybridMatrix::dumpOnFile(const patString& fileName, 
				 patError*&  err) {
  // On SGI, it seems that there are no way to create binaroy files.
  // THerefore, the matrix is dumped in ascii.

#ifdef SGI_COMPILER
  ofstream binFile(fileName.c_str(),ios::out) ;
  binFile << type << " " ;
  binFile << dim << " " ;
  unsigned long dataSize = data.size() ;
  binFile << dataSize << endl ;
  for (vector<patReal>::iterator i = data.begin() ;
       i != data.end() ;
       ++i) {
    binFile << *i << endl ;
  }
  binFile.close() ;
#else
  ofstream binFile(fileName.c_str(),ios::binary | ios::out) ;
  binFile.write((char*)&type,sizeof(type)) ;
  binFile.write((char*)&dim,sizeof(dim)) ;
  unsigned long dataSize = data.size() ;
  binFile.write((char*)&dataSize,sizeof(dataSize)) ;
  for (vector<patReal>::iterator i = data.begin() ;
       i != data.end() ;
       ++i) {
    patReal x = *i ;
    binFile.write((char*)&x,sizeof(x)) ;
  }
  binFile.close() ;
#endif
}

void patHybridMatrix::loadFromDumpFile(const patString& fileName, 
				       patError*&  err) {
#ifdef SGI_COMPILER
  ifstream binFile(fileName.c_str(), ios::in) ;
  binFile >> type ;
  if (binFile.bad()) {
    err = new patErrMiscError("Error in reading type from dump file") ;
    WARNING(err->describe()) ;
    binFile.close() ;
    return ;
  }
  binFile >> dim ;
  if (binFile.bad()) {
    err = new patErrMiscError("Error in reading dimension from dump file") ;
    WARNING(err->describe()) ;
    binFile.close() ;
    return ;
  }
  unsigned long dataSize ;
  binFile >> dataSize ;
  data.resize(dataSize) ;
  for (vector<patReal>::iterator i = data.begin() ;
       i != data.end() ;
       ++i) {
    patReal x ;
    binFile >> x ;
    if (binFile.bad()) {
      stringstream str ;
      str << "Error in reading entry " << i-data.begin() 
	  << " from dump file" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      binFile.close() ;
      return ;
    }
    *i = x ;
  }
  binFile.close() ;
#else
  ifstream binFile(fileName.c_str(),ios::binary | ios::in) ;
  binFile.read((char*)&type,sizeof(type)) ;
  if (binFile.bad() || binFile.gcount() != sizeof(type)) {
    DEBUG_MESSAGE("gcount = " << binFile.gcount()) ;
    DEBUG_MESSAGE("size   = " << sizeof(type)) ;
    err = new patErrMiscError("Error in reading type from dump file") ;
    WARNING(err->describe()) ;
    binFile.close() ;
    return ;
  }
  binFile.read((char*)&dim,sizeof(dim)) ;
  if (binFile.bad() || binFile.gcount() != sizeof(dim)) {
    err = new patErrMiscError("Error in reading dimension from dump file") ;
    WARNING(err->describe()) ;
    binFile.close() ;
    return ;
  }
  unsigned long dataSize ;
  binFile.read((char*)&dataSize,sizeof(dataSize)) ;
  data.resize(dataSize) ;
  for (vector<patReal>::iterator i = data.begin() ;
       i != data.end() ;
       ++i) {
    patReal x ;
    binFile.read((char*)&x,sizeof(x)) ;
    if (binFile.bad() || binFile.gcount() != sizeof(x)) {
      DEBUG_MESSAGE("gcount = " << binFile.gcount()) ;
      DEBUG_MESSAGE("size   = " << sizeof(type)) ;
      stringstream str ;
      str << "Error in reading entry " << i-data.begin() 
	  << " from dump file"  ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      binFile.close() ;
      return ;
    }
    *i = x ;
  }
  binFile.close() ;
#endif
}


//  patVariables patHybridMatrix::eigenValues(vector<patVariables>* eigenVectors,
//  					  patError*&  err) {

//    //  DEBUG_MESSAGE("Compute eigenvalues of the following matrix") ;
//    //cout << *this << endl ;

//    if (type == patSymmetric) {
//      LaSymmMatDouble A(getSize(),getSize()) ;
//      for (unsigned long i = 0 ; i < getSize() ; ++i) {
//        for (unsigned long j = 0 ; j < getSize() ; ++j) {
//  	A(i,j) = getElement(i,j,err) ;
//        }
//      }
    
//      int N = getSize() ;
//      LaVectorDouble v(N) ;
//      LaGenMatDouble Eigenvectors(N,N) ;
    
//      Eigenvectors(0,0) = 0.0 ;

//      if (eigenVectors != NULL) {
//        LaEigSolve(A,v,Eigenvectors) ;
//      }
//      else {
//        LaEigSolve(A,v) ;
//      }

//      patVariables result(getSize()) ;
//      for (unsigned long i = 0 ; i < getSize() ; ++i) {
//        result[i] = v(i) ;
//        if (eigenVectors != NULL) {
//  	for (unsigned long j = 0 ; j < getSize() ; ++j) {
//  	  (*eigenVectors)[i][j] = Eigenvectors(i,j) ;
//  	}
//        }
//      }      
//      return result ;
//    }
//    else if (type == patLower) {
//      WARNING("Not yet implemented") ;
//      return patVariables() ;
//    }
//    else if (type == patUpper) {
//      WARNING("Not yet implemented") ;
//      return patVariables() ;
//    }
//    else if (type == patDiagonal) {
//      patVariables eigVal(getSize()) ;
//      for (unsigned long i = 0 ; i < getSize() ; ++i) {
//        eigVal[i] = getElement(i,i,err) ;
//        if (eigenVectors !=  NULL) {
//  	for (unsigned long j = 0 ; j < getSize() ; ++j) {
//  	  (*eigenVectors)[i][j] = 0.0 ;
//  	}
//  	(*eigenVectors)[i][i] = 1.0 ;
//        }
//      }
//      return eigVal ;
//    }
//  }

void patHybridMatrix::add(const patHybridMatrix& M,
			  patError*&  err) {
  if (data.size() != M.data.size()) {
    stringstream str ;
    str << "Cannot add a matrix with " << data.size()
	<< " entries with a matrix with " << M.data.size() 
	<< " entries" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }

  for (unsigned long j = 0 ; j < data.size() ; ++j) {
    data[j] += M.data[j] ;
  }
}


void patHybridMatrix::set(const patHybridMatrix& M,
			  patError*&  err) {
  if (data.size() != M.data.size()) {
    stringstream str ;
    str << "Cannot set a matrix with " << data.size()
	<< " entries from a matrix with " << M.data.size() 
	<< " entries" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  
  for (unsigned long j = 0 ; j < data.size() ; ++j) {
    data[j] = M.data[j] ;
  }
}



void patHybridMatrix::addAlpha(patReal alpha, 
			  const patHybridMatrix& M,
			  patError*&  err) {
  if (data.size() != M.data.size()) {
    stringstream str ;
    str << "Cannot add a matrix with " << data.size()
	<< " entries with a matrix with " << M.data.size() 
	<< " entries" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }

  for (unsigned long j = 0 ; j < data.size() ; ++j) {
    data[j] += alpha * M.data[j] ;
  }
}

void patHybridMatrix::multAlpha(patReal alpha) {
  for (unsigned long j = 0 ; j < data.size() ; ++j) {
    data[j] = alpha * data[j] ;
  }
}

ostream& operator<<(ostream &str, const patHybridMatrix::patMatrixType& x) {
  switch(x) {
  case patHybridMatrix::patSymmetric :
    str << 0 ;
    return str ;
  case patHybridMatrix::patLower :
    str << 1 ;
    return str ;
  case patHybridMatrix::patUpper:
    str << 2 ;
    return str ;
  case patHybridMatrix::patDiagonal:
    str << 3 ;
    return str ;
  }
  return str ;
}
istream& operator>>(istream &str,  patHybridMatrix::patMatrixType& x) {
  short i ;
  str >> i ;
  switch (i) {
  case 0 :
    x = patHybridMatrix::patSymmetric ;
    return str ;
  case 1 :
    x = patHybridMatrix::patLower ;
    return str ;
  case 2: 
    x = patHybridMatrix::patUpper ;
    return str ;
  case 3:
    x = patHybridMatrix::patDiagonal ;
    return str ;
  }
  return str ;
}


patVariables patHybridMatrix::solve(const patVariables& b, patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patVariables() ;
  }
  if (!isLower()) {
    err = new patErrMiscError("Matrix must be lower triangular") ;
    WARNING(err->describe()) ;
    return patVariables() ;
  }
  
  if (b.size() != dim) {
      stringstream str ;
      str << "Incompatible sizes: " << b.size() << "<>" << dim ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patVariables() ;
  }

  patVariables z(b.size()) ;
  for (vector<patReal>::size_type i = 0 ; i < dim ; ++i) {
    patReal sum = b[i] ;
    for (vector<patReal>::size_type j = 0 ; j < i ; ++j) {
      sum -= getElement(i,j,err) * z[j] ;  
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patVariables();
      }
    }
    z[i] = sum / getElement(i,i,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patVariables();
    }
  }
  
  patVariables result(b.size()) ;
  for (vector<patReal>::size_type ip = dim ; ip >= 1 ; --ip) {
    vector<patReal>::size_type i = ip - 1 ;
    patReal sum = z[i] ;
    for (vector<patReal>::size_type j = i+1 ; j < dim ; ++j) {
      sum -= getElement(j,i,err) * result[j] ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patVariables();
      }
    }
    result[i] = sum / getElement(i,i,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patVariables();
    }
  }

  return result ;
}


patBoolean patHybridMatrix::correctForSingularity(int svdMaxIter, 
						  patReal threshold,
						  patError*& err) {

  DEBUG_MESSAGE("####### singularity param: " << singularityPenalty) ;
  patEigenVectors ev(this,svdMaxIter,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patFALSE ;
  }
  eigenVectors = ev.getEigenVector(threshold) ;
  if (eigenVectors.empty()) {
    return patFALSE ;
  }
  DEBUG_MESSAGE("--> size of sing. subspace: " << eigenVectors.size()) ;
  Q = new patMyMatrix(ev.getRows(),eigenVectors.size()) ;

  unsigned int j(0) ;
  for (map<patReal,patVariables>::iterator iter = eigenVectors.begin() ;
       iter != eigenVectors.end() ;
       ++iter) {
    for (unsigned int i = 0 ; i < ev.getRows() ; ++i) {
      (*Q)[i][j] = iter->second[i] ;
    }
    ++j ;
  }

  if (singularityPenalty != 0.0) {
    for (unsigned int i = 0 ; i < getSize() ; ++i) {
      for (unsigned int j = i ; j < getSize() ; ++j) {
	patReal add(0.0) ;
	for (unsigned int k = 0 ; k < eigenVectors.size() ; ++k) {
	  add += singularityPenalty * (*Q)[i][k] * (*Q)[j][k] ;
	}
	addElement(i,j,add,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return patFALSE ;
	}
      }    
    }
    return patTRUE ;
  }
  else {
    return patFALSE ;
  }
}
  

void patHybridMatrix::updatePenalty(patReal singularityThreshold, const patVariables& step,patError*& err) {
  if (Q == NULL) {
    return ;
  }
  if (step.size() != Q->nRows()) {
    stringstream str ;
    str << "Incompatible size " << step.size() << " and " << Q->nRows() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  patVariables res(eigenVectors.size(),0.0) ;
  
  multTranspVec(*Q,step,res,err) ;
   if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }


  DEBUG_MESSAGE("Q=" << *Q) ;
  DEBUG_MESSAGE("s=" << step) ;
  DEBUG_MESSAGE("res=" << res) ;
  patReal norm = inner_product(res.begin(),res.end(),res.begin(),0.0) ;

  DEBUG_MESSAGE("--> ||Qs||^2 = " << norm) ;
  if (norm >= singularityThreshold) {
    singularityPenalty *= 10.0 ;
  }
  else {
    singularityPenalty = 1.0 ;
  }


  DEBUG_MESSAGE("--> singularPenalty set to " << singularityPenalty) ;
}

patBoolean patHybridMatrix::isEmpty() const {
  return (dim == 0) ;
}

patBoolean patHybridMatrix::straightCholesky(patError*& err) {
  patVariables::size_type n = getSize() ;


  for (patVariables::size_type j = 0 ; j < n ; ++j) {
    
    for (patVariables::size_type k = 0 ; k < j ; ++k) {
      data[index(j,j)] -= data[index(j,k)] * data[index(j,k)] ;
    }
    if (data[index(j,j)] < 0) {
      WARNING("Matrix is not positive definite A(" << j << "," << j << ")=" << data[index(j,j)]) ;
      return patFALSE ;
    }
    data[index(j,j)] = sqrt(data[index(j,j)]) ;
    for (patVariables::size_type i = j+1 ; i < n ; ++i) {
      for (patVariables::size_type k = 0 ; k < j ; ++k) {
	data[index(i,j)] -= data[index(i,k)] * data[index(j,k)] ;
      }
      data[index(i,j)] /= data[index(j,j)] ; 
    }
  }
  setLower() ;
  return patTRUE ;
}
