//-*-c++-*------------------------------------------------------------
//
// File name : patCNL.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Aug 10 22:38:32 2000
//
//--------------------------------------------------------------------

#ifndef patCNL_h
#define patCNL_h


#include "patGEV.h"

#include "patIterator.h"

/**
   @doc This class implements the GEV generating function for the cross 
   nested logit model 
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Thu Aug 10 22:38:32 2000) 
*/

class patCNL : public patGEV {

public :

  /** 
   *Sole constructor
   */
  patCNL() ;

  /**
   * Destructor
   */
  virtual ~patCNL() ;

  /** 
   *Evaluate the function: 
   \[
   G = \sum_{m=1}^M \left( \sum_j (\alpha_{jm} x_j)^{\mu_m} \right)^{\frac{\mu}{\mu_m}}
   \]
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
			   patError*& err) ;


  /** 
   *Compute the partial derivative with respect to one variable.
   \[
   G_i=\frac{\partial G}{\partial x_i} = \mu \sum_m \alpha_{im}^{\mu_m} x_i^{\mu_m-1} \left( \sum_j (\alpha_{jm} x_j)^{\mu_m} \right)^{\frac{\mu}{\mu_m}-1}
   \]
   *  @param index index of partial derivative (variable index)
   *  @param x value of the variables. Most of the time, exponential of
   *           utilities.
   *  @param param value of the structural parameters
   *  @param mu value of the homogeneity factor
   *  @param available table with as many entries as the number of variables.
   *                   If an entry is patFALSE, the corresponding variables 
   *                   will not be considered in the sums.
   *  @param err ref. of the pointer to the error object.
      This routine has been successfully checked through a comparison with finite difference computation (Fri Jan 12 15:26:30 2001)  
   */
  virtual patReal getDerivative_xi(unsigned long index,
				   const patVariables* x,
				   const patVariables* param,
				   const patReal* mu,
				   const vector<patBoolean>& available,
				   patError*& err) ;


  /** 
   *Compute the partial derivative with respect to mu, the homogeneity factor
   \[
   \frac{\partial G}{\partial \mu} = \sum_m \frac{1}{\mu_m} y_m^{\frac{\mu}{\mu_m}} \ln(y_m)
   \]
   where
   \[
   y_m = \sum_{j\in C_m} (\alpha_{jm} x_j)^{\mu_m} 
   \]
   *  @param x value of the variables. Most of the time, exponential of
   *           utilities.
   *  @param param value of the structural parameters
   *  @param mu value of the homogeneity factor
   *  @param available table with as many entries as the number of variables.
   *      If an entry is patFALSE, the corresponding variables will not be
   *      considered in the sums.
   *  @param err ref. of the pointer to the error object.
      This routine has been successfully checked through a comparison with finite difference computation (Fri Jan 12 15:28:40 2001)  
   */
  virtual patReal getDerivative_mu(const patVariables* x,
				   const patVariables* param,
				   const patReal* mu,
				   const vector<patBoolean>& available,
				   patError*& err) ;
  

  /** 
   * Compute the partial derivative with respect to one structural parameter
   \[
   \frac{\partial G}{\partial \mu_m} = y_m^{\frac{\mu}{\mu_m}} \left( 
\frac{\mu}{\mu_m} \frac{1}{y_m} \sum_{j\in C} (\alpha_{jm} x_j)^{\mu_m} \ln (\alpha_{jm} x_j) - \frac{\mu}{\mu_m^2} \ln y_m
\right)
\]
   and
   \[
   \frac{\partial G}{\partial \alpha_{im}} = \mu y_m^{\frac{\mu}{\mu_m}-1} \alpha_{im}^{\mu_m-1}x_i^{\mu_m}
   \]
   where
   \[
    y_m = \sum_{j\in C_m} (\alpha_{jm} x_j)^{\mu_m} 
    \]
   *
   *  @param index index of partial derivative (parameter index)
   *  @param x value of the variables. Most of the time, exponential of
   *           utilities.
   *  @param param value of the structural parameters
   *  @param mu value of the homogeneity factor
   *  @param available table with as many entries as the number of variables.
   *      If an entry is patFALSE, the corresponding variables will not be
   *      considered in the sums.
   *  @param err ref. of the pointer to the error object. 
      This routine has been successfully checked through a comparison with finite difference computation (Mon Jan 15 11:56:30 2001)  
   */
  virtual patReal getDerivative_param(unsigned long index,
				      const patVariables* x,
				      const patVariables* param,
				      const patReal* mu, 
				      const vector<patBoolean>& available,
				      patError*& err) ;
   

  /** Compute the second partial derivative with respect to two variables.
      If $i=j$, we have
      \[
      \frac{\partial^2 G}{\partial x_i^2} = \frac{\partial G_i}{\partial x_i} = 
     \mu \sum_m \left(y_m^{\frac{\mu}{\mu_m}-2} \alpha_{im}^{\mu_m} x_i^{\mu_m-2}\left((\mu-\mu_m) \alpha_{im}^{\mu_m}x_i^{\mu_m} + y_m (\mu_m-1) \right)\right)
       \]
      and if $i\neq j$, we have
      \[
      \frac{\partial^2 G}{\partial x_i \partial x_j} = \frac{\partial G_i}{\partial x_j} = \mu \sum_m \left( (\mu-\mu_m) y_m^{\frac{\mu}{\mu_m}-2} \alpha_{im}^{\mu_m} \alpha_{jm}^{\mu_m} x_i^{\mu_m-1} x_j^{\mu_m-1}\right)
      \]
      where
      \[
      y_m = \sum_{j\in C_m} (\alpha_{jm} x_j)^{\mu_m} 
      \]
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
      This routine has been successfully checked through a comparison with finite difference computation (Mon Jan 15 12:55:09 2001)  
   **/

  virtual patReal getSecondDerivative_xi_xj(unsigned long index1,
					    unsigned long index2,
					    const patVariables* x,
					    const patVariables* param,
					    const patReal* mu,
					    const vector<patBoolean>& available,
					    patError*& err) ;


  /** Compute the second partial derivative with respect to one variable and mu.
      \[
      \frac{\partial^2 G}{\partial x_i \partial \mu} =  \frac{\partial G_i}{\partial \mu} = \sum_m \left( y_m^{\frac{\mu}{\mu_m}-1}\alpha_{im}^{\mu_m}x_i^{\mu_m-1}\left(1+\frac{\mu}{\mu_m}\ln y_m\right)\right)
      \]
      where
      \[
      y_m = \sum_{j\in C_m} (\alpha_{jm} x_j)^{\mu_m} 
      \]
   *  @param index index of partial derivative (variable index)
   *  @param x value of the variables. Most of the time, exponential of
   *           utilities.
   *  @param param value of the structural parameters
   *  @param mu value of the homogeneity factor
   *  @param available table with as many entries as the number of variables.
   *                   If an entry is patFALSE, the corresponding variables 
   *                   will not be considered in the sums.
   *  @param err ref. of the pointer to the error object.
      This routine has been successfully checked through a comparison with finite difference computation (Mon Jan 15 13:27:41 2001)  
   **/

  virtual patReal getSecondDerivative_xi_mu(unsigned long index,
					    const patVariables* x,
					    const patVariables* param,
					    const patReal* mu,
					    const vector<patBoolean>& available,
					    patError*& err) ;

  /** Compute the second partial derivative with respect to one structural parameter and one variable
      \[
      \frac{\partial^2 G}{\partial x_i \partial \mu_m} = \frac{\partial G_i}{\partial \mu_m} = \mu \alpha_{im} (\alpha_{im} x_i)^{\mu_m-1} y_m^{\frac{\mu}{\mu_m}-1} \left( \ln \alpha_{im} x_i + (\frac{\mu}{\mu_m}-1)\frac{1}{y_m}\frac{\partial y_m}{\partial \mu_m} - \frac{\mu}{\mu_m^2} \ln y_m\right),
      \]

      \[
      \frac{\partial^2 G}{\partial x_i \partial \alpha_{im}} = \mu \mu_m y_m^{\frac{\mu}{\mu_m}-1}  (\alpha_{im}x_i)^{\mu_m-1} \left(1+ (\frac{\mu}{\mu_m}-1) (\alpha_{im}x_i)^{\mu_m} \frac{1}{y_m}\right)
      \]
      and if $i\neq j$,
      \[
      \frac{\partial^2 G}{\partial x_i \partial \alpha_{jm}} = \mu (\frac{\mu}{\mu_m}-1) y_m^{\frac{\mu}{\mu_m}-2} \mu_m \alpha_{im}^{\mu_m}\alpha_{jm}^{\mu_m-1} x_i^{\mu_m-1}x_j^{\mu_m}
      \]
      where
      \[
      y_m = \sum_{j\in C_m} (\alpha_{jm} x_j)^{\mu_m} 
      \]
      and
      \[
      \frac{\partial y_m}{\partial \mu_m} = \sum_j (\alpha_{jm} x_j)^{\mu_m} \ln (\alpha_{jm} x_j).
      \]
  
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
					    patError*& err) ;

   


  /**
   *  Implements the pure virtual function from patGEV
   *  @return Cross-Nested Logit Model
   */
  virtual patString getModelName() ;

  /**
   */
   unsigned long getNbrParameters() ;

  /**
   */
  void compute(const patVariables* x,
	       const patVariables* param,
	       const patReal* mu, 
	       const vector<patBoolean>& available,
	       patBoolean computeSecondDerivatives,
	       patError*& err) ;

  /**
   */
  void generateCppCode(ostream& cppFile, 
			       patBoolean derivatives, 
			       patError*& err) ;

private:
  patBoolean firstTime ;
  unsigned long nNests ;
  unsigned long J ;
  vector<unsigned long> indexOfNestParam ; 
  vector<vector<unsigned long> > indexOfAlphaParam ;
  vector<unsigned long> nestForParamIndex ;
  vector<unsigned long> altForParamIndex ;

  vector<patReal> nestParams ;
  vector<vector<patReal> > alphas ;
  vector<vector<patReal> > alphasToMumOverMu ;
  vector<vector<patReal> > xToMum ;
  vector<patReal> Am ;
  vector<patReal> Bm ;
  vector<patReal> Delta ;
  vector<patReal> DAmDMum ;
  vector<patReal> DAmDMu ;
  vector<patReal> firstDeriv_xi ;
  vector<vector<patReal> > secondDeriv_xi_xj ;
  vector<vector<patReal> > secondDeriv_xi_param ;
  vector<patBoolean> computeDerivativeParam ;
  patBoolean computeMuDerivative ;
  vector<patIterator<unsigned long>*> alphaIter  ;
  vector<patReal> muDerivative ;
  patVariables currentPoint ;
};

#endif
