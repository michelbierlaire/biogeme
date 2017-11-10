//-*-c++-*------------------------------------------------------------
//
// File name : bioFixedParameter.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Mon Apr 27 16:00:25 2009
//
//--------------------------------------------------------------------

#ifndef bioFixedParameter_h
#define bioFixedParameter_h

/*!
Class defining a fixed parameter of the model (that is, not
distributed), to be estimated from data
*/

#include "bioLiteral.h"

class bioFixedParameter : public bioLiteral {

  friend class bioLiteralRepository ;
  friend ostream& operator<<(ostream &str, const bioFixedParameter& x) ;

public:
  patReal getValue() const ;
  void setValue(patReal val) ;
  bioLiteralType getType() const ;
  patReal getLowerBound() const ;
  patReal getUpperBound() const ;
  patBoolean isFixedParam() const ;
  patString printPythonCode(patBoolean estimation=patFALSE) const ;
  patULong getParameterId() const ;
  patULong getEstimatedParameterId() const ;
  patString getLaTeXRow(patReal stdErr, patError*& err) const ;
 protected:
  /*!
    Only the repository can create a parameter
  */
  bioFixedParameter(patString theName, 
		    patULong uniqueId,
		    patULong pId,
		    patULong eId,
		    patReal def, 
		    patReal lb, 
		    patReal ub, 
		    patBoolean f,
		    patString latex) ;


  /*!
    lowerBound is set to -1e20, upperBound is set to 1e20, isFixed is
    set to FALSE
   */
  bioFixedParameter(patString theName, 
		    patULong id, 
		    patULong pId,
		    patULong eId,
		    patReal def = 0.0) ;



 protected:
  patULong theParameterId ;
  patULong theEstimatedId ;
  patReal defaultValue ;
  patReal currentValue ;
  patReal lowerBound ;
  patReal upperBound ;
  patBoolean isFixed ;
  // Consecutive id for all parameters
  // Consecutive id for estimated parameters
  patString latexName ;
};

#endif
