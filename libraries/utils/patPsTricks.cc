//-*-c++-*------------------------------------------------------------
//
// File name : patPsTricks.cc
// Author :    \URL[Michel Bierlaire]{http://people.epfl.ch/michel.bierlaire}
// Date :      Mon Dec 24 09:59:15 2007
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patPsTricks.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"
#include "patMath.h"
#include "patFormatRealNumbers.h"

patPsTricks::patPsTricks(patVariables* x,
			 patVariables* y,
			 patError*& err) :
  lx(x),ly(y),lup(NULL),ldown(NULL) {

  if (lx == NULL || ly == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return ;
  }

  if (lx->size() != ly->size()) {
    stringstream str ;
    str << "Incompatible sizes: x(" << lx->size() << "), y(" << ly->size() << ")" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
    
  }
}

patPsTricks::patPsTricks(patVariables* x,
			 patVariables* y,
			 patVariables* up,
			 patVariables* down,
			 patError*& err)  :
  lx(x),ly(y),lup(up),ldown(down) {
  if (lx == NULL || ly == NULL || lup == NULL || ldown == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return ;
  }

  if (lx->size() != ly->size() || 
      lx->size() != lup->size() || 
      lx->size() != ldown->size()) {
    stringstream str ;
    str << "Incompatible sizes: x(" << lx->size() << "), y(" << ly->size() << "), up(" << lup->size() << "), down(" << ldown->size() << ")" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  
}
  
patString patPsTricks::getCode(patReal maxY,
			       patReal xSize,
			       patReal ySize,
			       patReal xMinSize,
			       patReal yMinSize,
			       patError*& err) {

  patFormatRealNumbers theNumber ;

  // Compute min and max

  patReal xmin = 0.0 ;
  patReal xmax = 0.0 ;
  patReal ymin = 0.0 ;
  patReal ymax = 0.0 ;

  for (patULong i = 0 ; i < lx->size() ; ++i) {
    if ((*lx)[i] > xmax) {
      xmax = (*lx)[i] ;
    }
    if ((*lx)[i] < xmin) {
      xmin = (*lx)[i] ;
    }

    // We project high and low values to maxY  and -maxY

    if ((*ly)[i] > maxY) {
      (*ly)[i] = maxY ;
    }
    if ((*ly)[i] < -maxY) {
      (*ly)[i] = -maxY ;
    }

    if ((*ly)[i] > ymax) {
      ymax = (*ly)[i] ;
    }
    if ((*ly)[i] < ymin) {
      ymin = (*ly)[i] ;
    }
    if (!singlePlot()) {
      if ((*lup)[i] > maxY) {
	(*lup)[i] = maxY ;
      }
      if ((*lup)[i] < -maxY) {
	(*lup)[i] = -maxY ;
      }
      if ((*lup)[i] > ymax) {
	ymax = (*lup)[i] ;
      }
      if ((*lup)[i] < ymin) {
	ymin = (*lup)[i] ;
      }
      if ((*ldown)[i] > maxY) {
	(*ldown)[i] = maxY ;
      }
      if ((*ldown)[i] < -maxY) {
	(*ldown)[i] = -maxY ;
      }
      if ((*ldown)[i] > ymax) {
	ymax = (*ldown)[i] ;
      }
      if ((*ldown)[i] < ymin) {
	ymin = (*ldown)[i] ;
      }
    }
  }

  stringstream str ;

  str << "\\psset{xunit=" << theNumber.format(patFALSE,
					      patTRUE,
					      3,
					      patMax(xSize/(xmax-xmin),xMinSize))  
      << "cm,yunit="<< theNumber.format(patFALSE,
					      patTRUE,
					      3,
					patMax(ySize/(ymax-ymin),yMinSize)) << "cm}" << endl ;
  str << "\\pspicture(" 
      << theNumber.format(patFALSE,
			  patTRUE,
			  3,
			  xmin) 
      << "," << theNumber.format(patFALSE,
				 patTRUE,
				 3,
				 ymin) << ")(" 
      << theNumber.format(patFALSE,
			  patTRUE,
			  3,
			  xmax) 
      << "," << theNumber.format(patFALSE,
				 patTRUE,
				 3,
				 ymax) <<")" << endl ;

  patReal dx = pow(10,round(log10(xmax-xmin)-0.5)) ;
  patReal dy = pow(10,round(log10(ymax-ymin)-0.5)) ;

  str << "\\psaxes[Dx=" << theNumber.format(patFALSE,
					      patTRUE,
					      3,
					    dx) 
      << ",Dy=" << theNumber.format(patFALSE,
					      patTRUE,
					      3,
				    dy) << "]{<->}(0,0)(" 
      << xmin << "," << ymin << ")(" 
      << xmax << "," << ymax <<")" << endl ;

  str << "\\savedata{\\maindata}[" << endl ;
  str << "{" << endl ;
  for (patULong i = 0 ; i < lx->size() ; ++i) {
    if (patFinite((*lx)[i]) && patFinite((*ly)[i])) {
      str << "{" << (*lx)[i] << "," << (*ly)[i] << "}," << endl;
    }
  }
  str << "}" << endl ;
  str << "]" << endl ;
  str << "\\dataplot[plotstyle=line]{\\maindata}" << endl ;

  if (!singlePlot()) {
    str << "\\savedata{\\updata}[" << endl ;
    str << "{" << endl ;
    for (patULong i = 0 ; i < lx->size() ; ++i) {
      if (patFinite((*lx)[i]) && patFinite((*lup)[i])) {
	str << "{" << (*lx)[i] << "," << (*lup)[i] << "}," << endl;
      }
    }
    str << "}" << endl ;
    str << "]" << endl ;
    str << "\\dataplot[plotstyle=line,linestyle=dashed]{\\updata}" << endl ;
    str << "\\savedata{\\downdata}[" << endl ;
    str << "{" << endl ;
    for (patULong i = 0 ; i < lx->size() ; ++i) {
      if (patFinite((*lx)[i]) && patFinite((*ldown)[i])) {
	str << "{" << (*lx)[i] << "," << (*ldown)[i] << "}," << endl;
      }
    }
    str << "}" << endl ;
    str << "]" << endl ;
    str << "\\dataplot[plotstyle=line,linestyle=dashed]{\\downdata}" << endl ;

  }
  str << "\\endpspicture" << endl ;
  return patString(str.str()) ;
}

patBoolean patPsTricks::singlePlot() const {
  
  return (lup == NULL && ldown == NULL) ;
}
