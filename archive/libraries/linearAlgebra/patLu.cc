//-*-c++-*------------------------------------------------------------
//
// File name : patLu.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Thu Jun 16 09:26:55 2005
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patLu.h"
#include "patMyMatrix.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"
#include "patMath.h"

patLu::patLu(patMyMatrix* aMat) : theMatrix(aMat), luPerformed(patFALSE), n(0) {
  if (aMat == NULL) {
    return ;
  }
  n = theMatrix->nRows() ;
  index.resize(n) ;
  solution.resize(n) ;
}
 
void patLu::computeLu(patError*& err) {
  if (theMatrix == NULL) {
    err = new patErrNullPointer("patMatrix") ;
    WARNING(err->describe()) ;
    return ;
  }

  if (n != theMatrix->nCols()) {
    stringstream str ;
    str << "Cannot perform LU on a " << n << "x" << theMatrix->nCols() << " matrix" ;
    err = new patErrMiscError(str.str()) ;
  }


  unsigned long i ;
  unsigned long imax ;
  unsigned long j ;
  unsigned long k ;
  
  patReal big ;
  patReal dum ;
  patReal sum ;
  patReal temp ;

  vector<patReal> vv(n) ;
  interchanges = 1 ;

  success = patTRUE ;

  for (i = 0 ; i < n ; ++i) {
    big = 0.0 ;
    for (j = 0 ; j < n ; ++j) {
      temp = patAbs((*theMatrix)[i][j]) ;
      if (temp > big) {
	big = temp ;
      }
    }
    if (big == 0.0) {
      success = patFALSE ;
      return ;
    }
    vv[i] = 1.0 / big ;
  }
  for (j = 0 ; j < n ; ++j) {
    for (i = 0 ; i < j ; ++i) {
      sum = (*theMatrix)[i][j] ;
      for (k = 0 ; k < i ; ++k) {
	sum -= (*theMatrix)[i][k] * (*theMatrix)[k][j] ;
      }
      (*theMatrix)[i][j] = sum ;
    }
    big = 0.0 ;
    for (i = j ; i < n ; ++i) {
      sum = (*theMatrix)[i][j] ;
      for (k = 0 ; k < j ; ++k) {
	sum -= (*theMatrix)[i][k] * (*theMatrix)[k][j] ;
      }
      (*theMatrix)[i][j] = sum ;
      dum = vv[i] * patAbs(sum) ;
      if (dum >= big) {
	big = dum ;
	imax = i ;
      }
    }
    if (j != imax) {
      for (k = 0 ; k < n ; ++k) {
	dum = (*theMatrix)[imax][k] ;
	(*theMatrix)[imax][k] = (*theMatrix)[j][k] ;
	(*theMatrix)[j][k] = dum ;
      }
      interchanges = -interchanges ;
      vv[imax] = vv[j] ;
    }
    index[j] = imax ;
    if (patAbs((*theMatrix)[j][j]) <= patEPSILON) {
      dum = (*theMatrix)[j][j] ;
      (*theMatrix)[j][j] = patSgn(dum) * patEPSILON ;
      success = patFALSE ;
    }
    if (j != n-1) {
      dum = 1.0 / ((*theMatrix)[j][j]) ;
      for (i = j+1 ; i < n ; ++i) {
	(*theMatrix)[i][j] *= dum ;
      }
    }
  }
  luPerformed = patTRUE ;
}

const vector<unsigned long>* patLu::getPermutation() const {
  return &index ;
}

const patVariables* patLu::solve(const patVariables& b,patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  if (b.size() != n) {
    stringstream str ;
    str << "Inconsistent sizes: " << b.size() << " and " << n ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL;
  }
  if (!luPerformed) {
    computeLu(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL;
    }
  }

  solution = b ;

  patReal sum ;

  long i ;
  unsigned long j ;
  unsigned long ii(0) ;
  unsigned long ip ;

  for (i = 0 ; i < n ; ++i) {
    ip = index[i] ;
    sum = solution[ip] ;
    solution[ip] = solution[i] ;
    if (ii != 0) {
      for (j = ii-1 ; j < i ; ++j) {
	sum -= (*theMatrix)[i][j] * solution[j] ;
      }
    }
    else if (sum != 0.0) {
      ii = i+1 ;
    }
    solution[i] = sum ;
  }
  for (i = n - 1 ; i >= 0 ; --i) {
    sum = solution[i] ;
    for (j=i+1; j < n ; ++j) {
      sum -= (*theMatrix)[i][j] * solution[j] ;
    }
    solution[i] = sum / (*theMatrix)[i][i] ;
  }
  return &solution ;
}

patBoolean patLu::isSuccessful() const {
  return success ;
}
