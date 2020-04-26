//-*-c++-*------------------------------------------------------------
//
// File name : patQR.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Jun 16 11:15:53 1999
// Recoded :   Fri Jan 20 13:40:29 2006
//
//--------------------------------------------------------------------

#ifndef patQR_h
#define patQR_h

#include "patMyMatrix.h"
#include "patHouseholder.h"

/**
   This object is in charge of the QR decomposition based on Householder matrices.
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Fri Jan 20 13:40:29 2006)
 */
class patQR {
  
public:
  /**
   */
  patQR(patMyMatrix* _patA = NULL) ;

  /**
   */
  void setMatrix(patMyMatrix *_patA) ;

  /** returns R
   */
  patMyMatrix* getR() const ;

  /**
     Compute QR and return Q and R such that A = Q'R
  */
  pair<patMyMatrix*,patMyMatrix*> QR(patError*& err) ; 
  /**
   */
  unsigned long getRank() const {return rank ;}

  /**
     Multiply Q by a vector without explicitly forming it
   */
  patVariables Qtimes(const patVariables& x,
		      patError*& err) const ;
  /**
     @param i an index
     @return perm[i], where perm is the permutation used during the QR factorization
   */
  unsigned long p(unsigned long i) {return perm[i] ;}
  /**
     @return  permutation used during the QR factorization
   */
  vector<unsigned long> getPermutation() {return perm;}

  /** Compute QR 
   */
  patMyMatrix* computeQR(patError*& err)  ;


private:
  patMyMatrix* A ;
  patMyMatrix* Q ;
  vector<patHouseholder> pk ;
  vector<unsigned long> perm ;
  vector<patReal> gamma ;
  unsigned long rank ;
  patBoolean factorized ;
  patBoolean qComputed ;
};
#endif
