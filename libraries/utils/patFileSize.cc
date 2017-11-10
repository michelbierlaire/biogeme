//-*-c++-*------------------------------------------------------------
//
// File name : patFileSize.cc
// Author :    Michel Bierlaire
// Date :      Mon Dec 19 08:59:39 2011
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include "patFileSize.h"
#include <cmath>

patFileSize::patFileSize(patULong n) : theNumber(n) {
  
}
ostream& operator<<(ostream &str, const patFileSize& x) {
  patReal GIGA(1073741824.0) ;
  patReal MEGA(1048576.0) ;
  patReal KILO(1024.0) ;

  if (x.theNumber == 1) {
    str << "1 byte" ;
  }
  else if (x.theNumber < KILO) {
    str << x.theNumber << " bytes" ;
  }
  else if (x.theNumber < MEGA) {
    str << floor(patReal(x.theNumber) / KILO) << " Kbytes" ;
  }
  else if (x.theNumber < GIGA) {
    str << floor(patReal(x.theNumber) / MEGA) << " Mbytes" ;
  }
  else {
    str << floor(patReal(x.theNumber) / GIGA) << " Gbytes" ;
  }
  return str ;
  
}

