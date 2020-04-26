//-*-c++-*------------------------------------------------------------
//
// File name : patGEV.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Jan 10 14:36:56 2001
//
//--------------------------------------------------------------------

#ifndef patGEV_h
#define patGEV_h

#include "patError.h"

#include "patVariables.h"

/**
  @doc This class defines the common interface shared  by all GEV models. 
  @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Jan 10 14:36:56 2001)
 */

class patGEV {

public :

  virtual ~patGEV() ;

  /** Evaluate the function. Purely virtual.
   *  @param x value of the variables. Most of the time, exponential of 
   *           utilities.
   *  @param param value of the structural parameters
   *  @param mu value of the homogeneity factor
   *  @param available table with as many entries as the number of variables.
   *                   If an entry is patFALSE, the corresponding variables 
   *                   will not be considered in the sums.
   *  @param err ref. of the pointer to the error object.
   */

  virtual patReal evaluate(const patVariables* x,
			   const patVariables* param,
			   const patReal* mu,
			   const vector<patBoolean>& available,
			   patError*& err) 
    = PURE_VIRTUAL ;
  
  /** Compute the partial derivative with respect to one variable.
   *  Purely virtual.
   *  @param index index of partial derivative (variable index)
   *  @param x value of the variables. Most of the time, exponential of
   *           utilities.
   *  @param param value of the structural parameters
   *  @param mu value of the homogeneity factor
   *  @param available table with as many entries as the number of variables.
   *                   If an entry is patFALSE, the corresponding variables 
   *                   will not be considered in the sums.
   *  @param err ref. of the pointer to the error object.
   **/

  virtual patReal getDerivative_xi(unsigned long index,
				   const patVariables* x,
				   const patVariables* param,
				   const patReal* mu,
				   const vector<patBoolean>& available,
				   patError*& err) 
    = PURE_VIRTUAL ;

  /**
     Same by finite difference
   */

  virtual patReal getDerivative_xi_finDiff(unsigned long index,
					   const patVariables* x,
					   const patVariables* param,
					   const patReal* mu,
					   const vector<patBoolean>& available,
					   patError*& err) ;

  /** Compute the partial derivative with respect to mu, the homogeneity factor
   *      Purely virtual.
   *  @param x value of the variables. Most of the time, exponential of
   *           utilities.
   *  @param param value of the structural parameters
   *  @param mu value of the homogeneity factor
   *  @param available table with as many entries as the number of variables.
   *      If an entry is patFALSE, the corresponding variables will not be
   *      considered in the sums.
   *  @param err ref. of the pointer to the error object.
   */
  virtual patReal getDerivative_mu(const patVariables* x,
				   const patVariables* param,
				   const patReal* mu,
				   const vector<patBoolean>& available,
				   patError*& err) 
    = PURE_VIRTUAL ;

  /**
     Same by finite difference
   */

  virtual patReal getDerivative_mu_finDiff(const patVariables* x,
				   const patVariables* param,
				   const patReal* mu,
				   const vector<patBoolean>& available,
				   patError*& err) ;

  /** Compute the partial derivative with respect to one structural parameter
   *  Purely virtual.
   *  @param index index of partial derivative (parameter index)
   *  @param x value of the variables. Most of the time, exponential of
   *           utilities.
   *  @param param value of the structural parameters
   *  @param mu value of the homogeneity factor
   *  @param available table with as many entries as the number of variables.
   *      If an entry is patFALSE, the corresponding variables will not be
   *      considered in the sums.
   *  @param err ref. of the pointer to the error object.
   */

  virtual patReal getDerivative_param(unsigned long index,
				      const patVariables* x,
				      const patVariables* param,
				      const patReal* mu, 
				      const vector<patBoolean>& available,
				      patError*& err) 
    = PURE_VIRTUAL ;

  /**
     Same by finite difference
  */
  
  virtual patReal getDerivative_param_finDiff(unsigned long index,
					      const patVariables* x,
					      const patVariables* param,
					      const patReal* mu, 
					      const vector<patBoolean>& available,
					      patError*& err) ;

  /** Compute the second partial derivative with respect to two variables.
   *  Purely virtual.
   *  @param index1 index of first partial derivative (variable index)
   *  @param index2 index of second partial derivative (variable index)
   *  @param x value of the variables. Most of the time, exponential of
   *           utilities.
   *  @param param value of the structural parameters
   *  @param mu value of the homogeneity factor
   *  @param available table with as many entries as the number of variables.
   *                   If an entry is patFALSE, the corresponding variables 
   *                   will not be considered in the sums.
   *  @param err ref. of the pointer to the error object.
   **/

  virtual patReal getSecondDerivative_xi_xj(unsigned long index1,
					    unsigned long index2,
					    const patVariables* x,
					    const patVariables* param,
					    const patReal* mu,
					    const vector<patBoolean>& available,
					    patError*& err) 
    = PURE_VIRTUAL ;

  /**
     Same by finite difference
   */

  virtual patReal getSecondDerivative_xi_xj_finDiff(unsigned long index1,
						    unsigned long index2,
						    const patVariables* x,
						    const patVariables* param,
						    const patReal* mu,
						    const vector<patBoolean>& available,
						    patError*& err) ;

  /** Compute the second partial derivative with respect to one variable and mu.
   *  Purely virtual.
   *  @param index index of partial derivative (variable index)
   *  @param x value of the variables. Most of the time, exponential of
   *           utilities.
   *  @param param value of the structural parameters
   *  @param mu value of the homogeneity factor
   *  @param available table with as many entries as the number of variables.
   *                   If an entry is patFALSE, the corresponding variables 
   *                   will not be considered in the sums.
   *  @param err ref. of the pointer to the error object.
   **/

  virtual patReal getSecondDerivative_xi_mu(unsigned long index,
					    const patVariables* x,
					    const patVariables* param,
					    const patReal* mu,
					    const vector<patBoolean>& available,
					    patError*& err) 
    = PURE_VIRTUAL ;

  /**
     Same by finite difference
   */

  patReal getSecondDerivative_xi_mu_finDiff(unsigned long index,
					    const patVariables* x,
					    const patVariables* param,
					    const patReal* mu,
					    const vector<patBoolean>& available,
					    patError*& err)  ;

  /** Compute the second partial derivative with respect to one structural parameter and one variable
   *  Purely virtual.
   *  @param indexVar index of partial derivative (variable index)
   *  @param indexParameter index of partial derivative (parameter index)
   *  @param x value of the variables. Most of the time, exponential of
   *           utilities.
   *  @param param value of the structural parameters
   *  @param mu value of the homogeneity factor
   *  @param available table with as many entries as the number of variables.
   *      If an entry is patFALSE, the corresponding variables will not be
   *      considered in the sums.
   *  @param err ref. of the pointer to the error object.
   */

  virtual patReal getSecondDerivative_param(unsigned long indexVar,
					    unsigned long indexParam,
					    const patVariables* x,
					    const patVariables* param,
					    const patReal* mu, 
					    const vector<patBoolean>& available,
					    patError*& err) 
    = PURE_VIRTUAL ;
   
  /**
     Same by finite difference
   */
  virtual patReal getSecondDerivative_param_finDiff(unsigned long indexVar,
						    unsigned long indexParam,
						    const patVariables* x,
						    const patVariables* param,
						    const patReal* mu, 
						    const vector<patBoolean>& available,
						    patError*& err) ;

  /**
   *Purely virtual.
  */
  virtual patString getModelName() = PURE_VIRTUAL ;

  /**
   * Purely virtual 
   */

  virtual unsigned long getNbrParameters() = PURE_VIRTUAL ;


  /**
   * For the sake of efficiency, all derivatives can be computed first and stored.
   */
  
  virtual void compute(const patVariables* x,
		       const patVariables* param,
		       const patReal* mu, 
		       const vector<patBoolean>& available,
		       patBoolean computeSecondDerivatives,
		       patError*& err) = PURE_VIRTUAL ;

  /**
   */  
  virtual void generateCppCode(ostream& cppFile, 
			       patBoolean derivatives, 
			       patError*& err) = PURE_VIRTUAL ;
};

#endif
