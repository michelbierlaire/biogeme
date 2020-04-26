//-*-c++-*------------------------------------------------------------
//
// File name : bioTypes.h
// @date   Wed Apr 11 12:58:48 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioTypes_h
#define bioTypes_h

#include <limits>
//#include <cfloat>

#ifdef LONGDOUBLE
typedef long double bioReal ;
#else
typedef double bioReal ;
#endif
const bioReal bioMinReal = std::numeric_limits<bioReal>::min() ;
const bioReal bioMaxReal = std::numeric_limits<bioReal>::max() ;
const bioReal bioEpsilon = std::numeric_limits<bioReal>::epsilon() ;

typedef bool bioBoolean ;
typedef unsigned long bioUInt ;
typedef long bioInt ;

#endif
