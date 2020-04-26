//-*-c++-*------------------------------------------------------------
//
// File name : patSampleEnuGetIndices.h
// Author :    Michel Bierlaire
// Date :      Sun Dec 16 15:47:22 2007
//
//--------------------------------------------------------------------

#ifndef patSampleEnuGetIndices_h
#define patSampleEnuGetIndices_h

#include "patType.h"
#include "patError.h"
#include "patOneZhengFosgerau.h"

/**
   This class determines in which columns results of the sample
   enumeration are stored

   The convention with utilities 

   Column 0 : id of the chosen alternative
   Column 1 : probability of the chosen alternative
   Column 2 + alt: utility of alternative alt
   Column 2 + nAlt + 2 * alt: proba of alternative alt
   Column 3 + nAlt + 2 * alt: residual of alternative alt
   Column 2 + 3 * nAlt: total (checksum)
   Column 3 + 3 * nAlt + e: expression e for Zheng-Fosgerau test

   Total number of columns: 3 + 3 * nAlt + nExpression

   Example with 3 alternatives with utilities and 2 expressions

   0   1  2  3  4  5  6  7  8  9 10 11 12 13
   C  PC U0 U1 U2 P0 R0 P1 R1 P2 R2  1 e1 e2

   Example with 4 alternatives with utilities and 2 expressions

   0   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
   C  PC U0 U1 U2 U3 P0 R0 P1 R1 P2 R2 P3 R3  1 e1 e2

   The convention without utilities 

   Column 0 : id of the chosen alternative
   Column 1 : probability of the chosen alternative
   Column 2 + 2 * alt: proba of alternative alt
   Column 3 + 2 * alt: residual of alternative alt
   Column 2 + 2 * nAlt: total (checksum)
   Column 3 + 2 * nAlt + e: expression e for Zheng-Fosgerau test

   Total number of columns: 3 + 2 * nAlt + nExpression

   Example with 3 alternatives without utilites and 2 expressions

   0   1  2  3  4  5  6  7  8  9 10
   C  PC P0 R0 P1 R1 P2 R2  1 e1 e2

   Example with 4 alternatives without utilites and 2 expressions

   0   1  2  3  4  5  6  7  8  9 10 11 12
   C  PC P0 R0 P1 R1 P2 R2 P3 R3  1 e1 e2

 */
class patSampleEnuGetIndices {

 public:
  patSampleEnuGetIndices(patBoolean util, patULong na, patULong ne) ;
  patULong getIndexProba(patULong alt, patError*& err) const ;
  patULong getIndexResid(patULong alt, patError*& err) const ;
  patULong getIndexUtil(patULong alt, patError*& err) const ;
  patULong getIndexExpr(patULong expr, patError*& err) const ;
  patULong getNbrOfColumns() const ;
  patULong getIndexZhengFosgerau(patOneZhengFosgerau* t, 
				 patError*& err) const ;
 private:
  const patBoolean withUtilities ;
  const patULong nAlt ;
  const patULong nExpressions ;

};

#endif
