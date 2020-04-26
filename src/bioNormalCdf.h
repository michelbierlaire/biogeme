//-*-c++-*------------------------------------------------------------
//
// File name : bioNormalCdf.h
// @date   Wed May 30 15:52:02 2018
// @author Michel Bierlaire
// @version Revision 1.0 (modified from pythonbiogeme)
//
//--------------------------------------------------------------------

#ifndef bioNormalCdf_h
#define bioNormalCdf_h

#include <vector>
#include "bioTypes.h"

class bioNormalCdf {

 public:
  bioNormalCdf() ;
  bioReal compute(bioReal gdx) const ;
  bioReal derivative(bioReal x) const ;
 private:
  bioReal gammp(bioReal a, bioReal x) const ;
  bioReal gser(bioReal a, bioReal x) const ;
  bioReal gcf(bioReal a, bioReal x) const ;
  bioReal gammln(bioReal xx) const ; 
  bioReal oneDivSqrtTwoPi ;
  unsigned short itmax ;
  bioReal eps ;
  bioReal fpmin ;
  std::vector<bioReal> cof ;
  bioReal stp ;
};

#endif
