#include <algorithm>
#include <functional>
#include "trSchnabelEskow.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"

trSchnabelEskow::trSchnabelEskow(const patHybridMatrix& pr) :
  precond(pr) {

}
  
void trSchnabelEskow::factorize(patReal tolerance,
				patError*& err) {

  //  DEBUG_MESSAGE("Before : " << precond) ;
  if (precond.isSymmetric()) {
    //        DEBUG_MESSAGE("Perform Cholesky factorization") ;
    // Perform the modified Cholesky factorization
    precond.cholesky(tolerance, err) ;
    //        DEBUG_MESSAGE("Done") ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
  //  DEBUG_MESSAGE("After : " << precond) ;

}

trVector trSchnabelEskow::solve(const trVector* b,
				     patError*& err) const {

  //DEBUG_MESSAGE("We are here") ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return trVector();
  }

  if (b == NULL) {
    err = new patErrNullPointer("trVector") ;
    WARNING(err->describe()) ;
    return trVector() ;
  } 

  if (b->size() != precond.getSize()) {
    err = new patErrMiscError("Incompatible sizes") ;
    WARNING(err->describe()) ;
    return (*b) ;
  }

  if (precond.isSymmetric()) {
    err = new patErrMiscError("Matrix must be lower triangular") ;
    WARNING(err->describe()) ;
    return(*b);
  }
  
//      DEBUG_MESSAGE("After Cholesky") ;
//      DEBUG_MESSAGE("Pivot : ") ;
//      copy(precond.pivot.begin(),
//           precond.pivot.end(),
//           ostream_iterator<unsigned long>(cout,",")) ;
//      DEBUG_MESSAGE("Pivot inverse: ") ;
//        copy(precond.pivotInverse.begin(),
//    	 precond.pivotInverse.end(),
//    	 ostream_iterator<unsigned long>(cout,",")) ;


  // We solve here a linear system of equations Ax = b where we know L such
  // that P'AP = LL', where P is a permutation matrix, and L a lower
  // triangular matrix (the Cholesky factor of A)
  //
  // Reference : "Matrix Computations" by G.H. Golub and C. F. Van Loan, North
  // Oxford Academic, 1983 store the result in x.

  // First solve Lz = P'b and store the result in x
  //  DEBUG_MESSAGE("First solve Lz = P'b and store the result in x") ;

//     DEBUG_MESSAGE("L is " << precond) ;
//     DEBUG_MESSAGE("b is " << *b) ;
//     DEBUG_MESSAGE("P'b is ") ;
//     for (patVariables::size_type i = 0 ; i < b->size() ; i++) {
//       cout << (*b)[precond.pivotInverse[i]] << " " ;
//     }
//    cout << endl ;

  patVariables z(b->size()) ;

  for (patVariables::size_type i = 0 ; i < b->size() ; i++) {
    z[i] = (*b)[precond.pivot[i]] ;
    for (patVariables::size_type j = 0 ; j < i ; j++)
      z[i] -= precond(i,j,err) * z[j] ;
    z[i] /= precond(i,i,err) ;
  }

  //    DEBUG_MESSAGE("z is " << z) ;
//    //   Then solve L'y = z 
//     DEBUG_MESSAGE("Then solve L'y = z ") ;
//     DEBUG_MESSAGE("L is " << precond) ;
//     DEBUG_MESSAGE("z is " << z) ;

  patVariables y(z.size()) ;
  
  for (patVariables::size_type i = b->size()  ; i > 0 ; --i) {
    y[i-1] = z[i-1] ;
    for (patVariables::size_type j = i  ; j < b->size() ; ++j) {
      y[i-1] -= precond(j,i-1,err) * y[j] ;
    }
    y[i-1] /= precond(i-1,i-1,err) ;
  }

  //    DEBUG_MESSAGE("y is " << y) ;
//    // Finally, x = Py
//    DEBUG_MESSAGE("Finally, x = Py") ;
    //       DEBUG_MESSAGE("Pivot : ") ;
    //     copy(precond.pivot.begin(),
//          precond.pivot.end(),
//          ostream_iterator<unsigned long>(cout,",")) ;
//     DEBUG_MESSAGE("Pivot inverse: ") ;
//       copy(precond.pivotInverse.begin(),
//   	 precond.pivotInverse.end(),
//   	 ostream_iterator<unsigned long>(cout,",")) ;

  patVariables x(y.size()) ;

  for (patVariables::size_type i = 0 ; i < b->size() ; i++) {
    x[i] = y[precond.pivotInverse[i]] ;
  }
  //    DEBUG_MESSAGE("x is " << x) ;
  return(x) ;
}


ostream& operator<<(ostream &str, const trSchnabelEskow& x) {
  str << x.precond ;
  return str ;
}
