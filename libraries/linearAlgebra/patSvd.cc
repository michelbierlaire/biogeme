//-*-c++-*------------------------------------------------------------
//
// File name : patSvd.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Mon Aug 15 11:25:47 2005
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "patSvd.h"
#include "patMyMatrix.h"
#include "patNrSgn.h"
#include "patMath.h"
#include "patPythag.h"
#include "patErrMiscError.h"

patSvd::patSvd(patMyMatrix* aMat) : theMatrix(NULL), 
				    V(NULL), 
				    W(NULL),
				    theInverse(NULL),
				    rv1(NULL),
				    svdComputed(patFALSE) {
  if (aMat == NULL) return ;
  m = aMat->nRows() ;
  n = aMat->nCols() ;
  if (m == 0) return ;
  if (n == 0) return ;
  theMatrix = aMat ;
  V = new patMyMatrix(n,n) ;
  W = new patVariables(n) ;
  if (m == n) {
    theInverse = new patMyMatrix(n,n) ;
  }
}

patSvd::~patSvd() {
  DELETE_PTR(W) ;
  DELETE_PTR(rv1) ;
  DELETE_PTR(theInverse) ;
  DELETE_PTR(V) ;
}

void patSvd::computeSvd(patULong maxIter, patError*& err) {
  patBoolean flag ;
  rv1 = new patVariables(n) ;

  patReal c,f,g,h,s,x,y,z,scale, anorm ;
  
  g = 0.0 ;
  scale = 0.0 ;
  anorm = 0.0 ;
  
  // indices
  int i, its, j, jj, k, l, nm ;
  
  for (i = 0 ; i < n ; ++i) {
    l = i + 2 ;
    (*rv1)[i] = scale*g ;
    g = s = scale = 0.0 ;
    if (i < m) {
      for (k = i; k < m ; ++k) {
	scale += patAbs((*theMatrix)[k][i]) ;
      }
      if (scale != 0.0) {
	for (k = i; k < m ; ++k) {
	  (*theMatrix)[k][i] /= scale ;
	  s +=  (*theMatrix)[k][i] * (*theMatrix)[k][i] ;
	}
	f = (*theMatrix)[i][i] ;
	g = -patNrSgn(sqrt(s),f) ;
	h = f * g - s ;
	(*theMatrix)[i][i] = f - g ;
	for (j = l-1 ; j < n ; ++j) {
	  for (s = 0.0, k = i ; k < m ; ++k) {
	    s += (*theMatrix)[k][i] * (*theMatrix)[k][j] ;
	  }
	  f = s / h ;
	  for (k = i ; k < m ; ++k) {
	    (*theMatrix)[k][j] += f * (*theMatrix)[k][i] ;
	  }
	}
	for (k = i ; k < m ; ++k) {
	  (*theMatrix)[k][i] *= scale ;
	}
      }    
    }
    (*W)[i] = scale * g ;
    g = s = scale = 0.0 ;
    if (i+1  <= m && i != n) {
      for (k=l-1; k < n; ++k) {
	scale += patAbs((*theMatrix)[i][k]) ;
      }
      if (scale != 0.0) {
	for (k=l-1; k < n ; ++k) {
	  (*theMatrix)[i][k] /= scale ;
	  s += (*theMatrix)[i][k] * (*theMatrix)[i][k] ;
	}
	f = (*theMatrix)[i][l-1] ;
	g = -patNrSgn(sqrt(s),f) ;
	h = f * g - s ;
	(*theMatrix)[i][l-1] = f - g ;
	for (k=l-1; k < n ; ++k) {
	  (*rv1)[k] = (*theMatrix)[i][k] / h ;
	}
	for (j=l-1; j < m ; ++j) {
	  for (s=0.0, k=l-1 ; k < n ; ++k) {
	    s += (*theMatrix)[j][k] * (*theMatrix)[i][k] ;
	  }
	  for (k=l-1; k < n ; ++k) {
	    (*theMatrix)[j][k] += s * (*rv1)[k] ;
	  }
	}
	for (k = l-1 ; k < n ; ++k) {
	  (*theMatrix)[i][k] *= scale ;
	}
      }
    }
    anorm = patMax(anorm,(patAbs((*W)[i])+patAbs((*rv1)[i]))) ;
  }
  
  for (i=n-1 ; i>=0 ; --i) {
    if (i < n-1) {
      if (g != 0.0) {
	for (j=l ; j < n ; ++j) {
	  (*V)[j][i] = ((*theMatrix)[i][j]/(*theMatrix)[i][l])/g ;
	}
	for (j=l ; j < n ; ++j) {
	  for (s=0.0, k=l ; k < n ; ++k) {
	    s += (*theMatrix)[i][k] * (*V)[k][j] ;
	  }
	  for (k=l ; k < n ; ++k) {
	    (*V)[k][j] += s * (*V)[k][i] ;
	  }
	}
      }
      for (j=l ; j < n ; ++j) {
	(*V)[i][j] = (*V)[j][i] = 0.0 ;
      }
    }
    (*V)[i][i] = 1.0 ;
    g = (*rv1)[i] ;
    l = i ;
  }
  for (i=patMin<int>(m,n)-1; i>=0;--i) {
    l = i + 1 ;
    g = (*W)[i] ;
    for (j = l ; j < n ; ++j) {
      (*theMatrix)[i][j] = 0.0 ;
    }
    if (g != 0.0) {
      g = 1.0 / g ;
      for (j = l ; j < n ; ++j) {
	for (s=0.0, k=l ; k <m ; ++k) {
	  s += (*theMatrix)[k][i] * (*theMatrix)[k][j] ;
	}
	f = (s / (*theMatrix)[i][i]) * g ;
	for (k = i ; k < m ; ++k) {
	  (*theMatrix)[k][j] += f * (*theMatrix)[k][i] ;
	}
      }
      for (j=i ; j < m ; ++j) {
	(*theMatrix)[j][i] *= g ;
      }
    }
    else {
      for (j=i ; j < m ; ++j) {
	(*theMatrix)[j][i] = 0.0 ;
      }
    }
    ++(*theMatrix)[i][i] ;
  }
  for (k=n-1; k >= 0; --k) {
    for (its = 0 ; its < maxIter ; ++its) {
      flag = patTRUE ;
      for (l = k ; l >= 0 ; --l) {
	nm = l-1 ;
	if (patAbs((*rv1)[l])+anorm == anorm) {
	  flag = patFALSE ;
	  break ;
	}
	if (patAbs((*W)[nm]) + anorm == anorm) break ;
      }
      if (flag) {
	c = 0.0 ;
	s = 1.0 ;
	for (i= l-1; i < k+1; ++i) {
	  f = s * (*rv1)[i] ;
	  (*rv1)[i] = c * (*rv1)[i] ;
	  if (patAbs(f) + anorm == anorm) break ;
	  g = (*W)[i] ;
	  h = patPythag(f,g) ;
	  (*W)[i] = h ;
	  h = 1.0 / h ;
	  c = g * h ;
	  s = -f * h ;
	  for (j = 0; j < m ; ++j) {
	    y = (*theMatrix)[j][nm] ;
	    z = (*theMatrix)[j][i] ;
	    (*theMatrix)[j][nm] = y * c + z * s ;
	    (*theMatrix)[j][i] = z * c - y * s ;
	  }
	}
      }
      z = (*W)[k] ;
      if (l == k) {
	if (z < 0.0) {
	  (*W)[k] = -z ;
	  for (j=0 ; j < n ; ++j) {
	    (*V)[j][k] = -(*V)[j][k] ;
	  }
	}
	break ;
      }
      if (its == (maxIter - 1)) {
	stringstream str ;
	str << "No convergence in " << maxIter << " SVD iterations" ;
	WARNING(patString(str.str())) ;
	svdComputed = patFALSE ;

	return ;
      }
      x = (*W)[l] ;
      nm = k-1 ;
      y = (*W)[nm] ;
      g = (*rv1)[nm] ;
      h = (*rv1)[k] ;
      f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y) ;
      g = patPythag(f,1.0) ;
      f = ((x-z)*(x+z)+h*((y/(f+patNrSgn(g,f)))-h))/x ;
      c = s = 1.0 ;
      for (j=l ; j <= nm ; ++j) {
	i = j+1 ;
	g = (*rv1)[i] ;
	y = (*W)[i] ;
	h = s*g ;
	g = c*g ;
	z = patPythag(f,h) ;
	(*rv1)[j] = z ;
	c = f / z ;
	s = h / z ;
	f = x * c + g * s ;
	g = g * c - x * s ;
	h = y * s ;
	y *= c ;
	for (jj = 0 ; jj < n ; ++jj) {
	  x = (*V)[jj][j] ;
	  z = (*V)[jj][i] ;
	  (*V)[jj][j] = x * c + z * s ;
	  (*V)[jj][i] = z * c - x * s ;
	}
	z = patPythag(f,h) ;
	(*W)[j] = z ;
	if (z) {
	  z = 1.0 / z ;
	  c = f * z ;
	  s = h * z ;
	}
	f = c * g + s * y ;
	x = c * y - s * g ;
	for (jj = 0 ; jj < m ; ++jj) {
	  y = (*theMatrix)[jj][j] ;
	  z = (*theMatrix)[jj][i] ;
	  (*theMatrix)[jj][j] = y * c + z * s ;
	  (*theMatrix)[jj][i] = z * c - y * s ;
	}
      }
      (*rv1)[l] = 0.0 ;
      (*rv1)[k] = f ;
      (*W)[k] = x ;
    }
  }

  svdComputed = patTRUE ;
}

const patMyMatrix* patSvd::getU() const {
  if (svdComputed) {
    return theMatrix ;
  }
  else {
    return NULL ;
  }
}

const patMyMatrix* patSvd::getV() const {
  if (svdComputed) {
    return V ;
  }
  else {
    return NULL ;
  }
}

const patVariables* patSvd::getSingularValues() const {
  if (svdComputed) {
    return W ;
  }
  else {
    return NULL ;
  }
}

patVariables patSvd::backSubstitution(const patVariables& b) {
  if (!svdComputed) {
    return patVariables() ;
  }
  patVariables x(n) ;
  int jj,j,i ;
  patReal s ;
  patVariables tmp(n) ;
  for (j = 0 ; j < n ; ++j) {
    s = 0.0 ;
    for (i = 0 ; i < m ; ++i) {
      s += (*theMatrix)[i][j] * b[i] ;
    }
    if (patAbs((*W)[j]) >= patMinReal) {
      s /= (*W)[j] ;
    }
    else {
      s = patSgn(s) * sqrt(patMaxReal) ;
    }
    tmp[j] = s ;
  }
  for (j = 0 ; j < n ; ++j) {
    s = 0.0 ;
    for (jj = 0 ; jj < n ; ++jj) {
      s += (*V)[j][jj] * tmp[jj] ;
    }
    x[j] = s ;
  }
  return x ;
}

const patMyMatrix* patSvd::computeInverse() {
  if (!svdComputed || (n != m)) {
    return NULL ;
  }

  for (int i = 0 ; i < n ; ++i) {
    for (int j = 0 ; j < n ; ++j) {
      (*theInverse)[i][j] = 0.0 ;
      for (int k = 0 ; k < n ; ++k) {
	(*theInverse)[i][j] += (*theMatrix)[j][k] * (*V)[i][k] / (*W)[k] ;
      }
      if (!patFinite((*theInverse)[i][j])) {
	(*theInverse)[i][j] = patSgn((*theInverse)[i][j]) * sqrt(patMaxReal) ;
      }
    }
  }
  return theInverse ;
}

const patMyMatrix* patSvd::getInverse() const {
  if (!svdComputed || (n != m)) {
    return NULL ;
  }
  else {
    return theInverse ;
  }
  
}

map<patReal,patVariables>  patSvd::getEigenVectorsOfZeroEigenValues(patReal threshold) {
  if (n != m) {
    return map<patReal,patVariables>() ;
  }
  map<patReal,patVariables> theMap ;
  patVariables aVector(n) ;
  for (int i = 0 ; i < n ; ++i) {
    if (patAbs((*W)[i]) <= threshold) {
      for (int j = 0 ; j < n ; ++j) {
	aVector[j] = (*V)[j][i] ;
      }
      theMap[(*W)[i]] = aVector ;
    }
  }
  return theMap ;
}

patReal patSvd::getSmallestSingularValue() {
  if (!svdComputed) {
    return patMaxReal ;
  }
  smallestSingularValue = patMaxReal ;
  for (int i = 0 ; i < W->size() ; ++i) {
    if (patAbs((*W)[i]) < patAbs(smallestSingularValue)) {
      smallestSingularValue = (*W)[i] ;
    }
  }
  return smallestSingularValue ;
}
