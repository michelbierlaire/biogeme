//-*-c++-*------------------------------------------------------------
//
// File name : patPsTricks.h
// Author :    \URL[Michel Bierlaire]{http://transp-or2.epfl.ch}
// Date :      Mon Dec 24 09:54:55 2007
//
//--------------------------------------------------------------------

#ifndef patPsTricks_h
#define patPsTricks_h

#include "patError.h"
#include "patVariables.h"

class patPsTricks {

 public:
  patPsTricks(patVariables* x,
	      patVariables* y,
	      patError*& err) ;

  patPsTricks(patVariables* x,
	      patVariables* y,
	      patVariables* up,
	      patVariables* down,
	      patError*& err) ;
  
  patString getCode(patReal maxY, // patParameters::the()->getgevNonParamPlotMaxY() ;
		    patReal xSize, // patParameters::the()->getgevNonParamPlotXSizeCm()
		    patReal ySize,  // patParameters::the()->getgevNonParamPlotYSizeCm() 
		    patReal xMinSize, //patParameters::the()->getgevNonParamPlotMinXSizeCm()
		    patReal yMinSize, // patParameters::the()->getgevNonParamPlotMinYSizeCm() 
		    patError*& err) ;

  patBoolean singlePlot() const;

private:
  
  patVariables* lx ;
  patVariables* ly ;
  patVariables* lup ;
  patVariables* ldown ;

};

#endif
