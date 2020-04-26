//-*-c++-*------------------------------------------------------------
//
// File name : patVariables.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri Jan 29 16:46:40 1999
//
//--------------------------------------------------------------------

#ifndef patVariables_h
#define patVariables_h

#include <vector>
#include "patType.h"
#include <iostream>

/**
 */
typedef vector<patReal> patVariables;

/**
 */
patVariables operator-(const patVariables& x,
		       const patVariables& y) ;

/**
 */
patVariables operator/(const patVariables& x,const patReal& y) ;

/** 
 */
patVariables operator-(const patVariables& x) ;

/** 
 */
patVariables operator+(const patVariables& x,
		       const patVariables& y) ;

/** 
 */
patVariables operator*(patReal alpha,
		       const patVariables& y) ;

/**
 */
patVariables &operator+=(patVariables& x, const patVariables& y) ;
/**
 */
patVariables &operator-=(patVariables& x, const patVariables& y) ;

/**
 */
patVariables &operator*=(patVariables& x, const float alpha) ;

/**
 */
patReal norm2(patVariables& x) ;

/**
 */
patVariables &operator/=(patVariables& x, const float alpha) ;


/**
 */
void x_plus_ay(patVariables& x, float a, const patVariables& y) ;


/**
 @doc Function object checking if two variables are symmetric
*/
class isSymmetric {
public:
  /**
   */
  isSymmetric(const patVariables& x) : ref(x) {};
  /**
   */
  patBoolean operator()(const patVariables& x) ;

private:
  const patVariables& ref ;
} ;

  /** @doc Function object comparing two patVariables. The absolute value of each
      component is considered. Therefore, two symmetric patVariables are
      considered equal. It is designed to allow the pattern search algorithm
      to consider symmetric directions in sequence, as described by Torczon
      (97).  
      @see Torczon, V. (1997) "On the convergence of pattern search
      algorithms" */

  class compSymmetric {
  public:
    /**
     */
  patBoolean operator()(const patVariables& x, const patVariables& y) ;
};

/**
 */
ostream& operator<<(ostream &str, const patVariables& x) ;

#endif
