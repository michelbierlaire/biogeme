//-*-c++-*------------------------------------------------------------
//
// File name : patNL.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Sep  6 09:01:26 2000
//
//--------------------------------------------------------------------

#ifndef patNL_h
#define patNL_h


#include "patGEV.h"
#include <list>
#include <vector>

/**
  @doc This class implements the GEV generating function for the nested logit model
  @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Sep  6 09:01:26 2000)
 */

class patNL : public patGEV {

public :

  friend ostream& operator<<(ostream& str, const patNL&) ;

  /** Sole constructor
   */
  patNL(patError*& err) ;

  /**
     Destructor
   */
  virtual ~patNL() ;


  /** Evaluate the function for $M$ nests 
      \[
      G(x) = \sum_{m=1}^M \left( \sum_{i\in C_m} x_i^{\mu_m}\right)^{\frac{\mu}{\mu_m}}
      \]

      @param x value of the variables. Most of the time, exponential of 
               utilities.
      @param param value of the structural parameters
      @param mu value of the homogeneity factor
      @param available table with as many entries as the number of variables.
                       If an entry is patFALSE, the corresponding variables 
                       will not be considered in the sums.
      @param err ref. of the pointer to the error object.
   */

  virtual patReal evaluate(const patVariables* x,
			   const patVariables* param,
			   const patReal* mu,
			   const vector<patBoolean>& available,
			   patError*& err) ;


  /** Compute the partial derivative with respect to one variable.
      \[
      \frac{\partial G}{\partial x_i} = \mu x_i^{\mu_{m_i}-1}\left( \sum_{j\in C_{m_i}} x_j^{\mu_{m_i}}\right)^{(\frac{\mu}{\mu_{m_i}}-1)}
      \]
      where $m_i$ is the (unique) nest containing alternative $i$.

      @param index index of partial derivative (variable index)
      @param x value of the variables. Most of the time, exponential of
               utilities.
      @param param value of the structural parameters
      @param mu value of the homogeneity factor
      @param available table with as many entries as the number of variables.
                       If an entry is patFALSE, the corresponding variables 
                       will not be considered in the sums.
      @param err ref. of the pointer to the error object.
      This routine has been successfully checked through a comparison with finite difference computation (Fri Jan 12 14:14:01 2001)  
   */

  virtual patReal getDerivative_xi(unsigned long index,
				   const patVariables* x,
				   const patVariables* param,
				   const patReal* mu,
				   const vector<patBoolean>& available,
				   patError*& err) ;


  /** Compute the partial derivative with respect to mu, the homogeneity factor

      \[
      \frac{\partial G}{\partial \mu} = \sum_{m=1}^M \frac{1}{\mu_m} \left( \sum_{i\in C_m} x_i^{\mu_m}\right)^{\frac{\mu}{\mu_m}} \ln\left( \sum_{i\in C_m} x_i^{\mu_m}\right)
      \]
      @param x value of the variables. Most of the time, exponential of
               utilities.
      @param param value of the structural parameters
      @param mu value of the homogeneity factor
      @param available table with as many entries as the number of variables.
          If an entry is patFALSE, the corresponding variables will not be
          considered in the sums.
      @param err ref. of the pointer to the error object.
      This routine has been successfully checked through a comparison with finite difference computation (Fri Jan 12 14:15:01 2001)  
   */
  virtual patReal getDerivative_mu(const patVariables* x,
				   const patVariables* param,
				   const patReal* mu,
				   const vector<patBoolean>& available,
				   patError*& err) ;



  /** Compute the partial derivative with respect to one structural parameter $\mu_m$
      \[
      \frac{\partial G}{\partial \mu_m} = \frac{\mu}{\mu_m} (\sum_{i \in C_m} x_i^{\mu_m})^{\frac{\mu}{\mu_m}-1} (\sum_{i \in C_m} x_i^{\mu_m} \ln(x_i)) - \frac{\mu}{\mu_m^2} (\sum_{i \in C_m} x_i^{\mu_m})^{\frac{\mu}{\mu_m}} \ln (\sum_{i \in C_m} x_i^{\mu_m})
      \]

      @param index index of partial derivative (parameter index)
      @param x value of the variables. Most of the time, exponential of
               utilities.
      @param param value of the structural parameters
      @param mu value of the homogeneity factor
      @param available table with as many entries as the number of variables.
          If an entry is patFALSE, the corresponding variables will not be
          considered in the sums.
      @param err ref. of the pointer to the error object. 
      This routine has been successfully checked through a comparison with finite difference computation (Fri Jan 12 14:22:52 2001)  
   */

  virtual patReal getDerivative_param(unsigned long index,
				      const patVariables* x,
				      const patVariables* param,
				      const patReal* mu, 
				      const vector<patBoolean>& available,
				      patError*& err) ;

  /** Compute the second partial derivative with respect to two variables $i$ and $j$. If $i=j$, we have
      \[
      \frac{\partial^2 G}{\partial x_i^2} = \frac{\partial G_i}{\partial x_i} =
      \mu(\mu_m-1)x_i^{(\mu_m-2)}(\sum_{i \in C_m} x_i^{\mu_m})^{(\frac{\mu}{\mu_m}-1)} + \mu(\mu-\mu_m)x_i^{(2\mu_m-2)}(\sum_{i \in C_m} x_i^{\mu_m})^{(\frac{\mu}{\mu_m}-2)}  
      \]
      If $i \neq j$ and $i,j \in C_m$, we have
      \[
\frac{\partial^2 G}{\partial x_i \partial x_j} = \frac{\partial G_i}{\partial x_j}  = \mu(\mu-\mu_m)x_i^{\mu_m-1}x_j^{\mu_m-1}(\sum_{i \in C_m} x_i^{\mu_m})^{(\frac{\mu}{\mu_m}-2)}  
      \]
      If $i\in C_m$ and $j \not\in C_m$, we have 
      \[
\frac{\partial^2 G}{\partial x_i \partial x_j} = \frac{\partial G_i}{\partial x_j}  = 0
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
      This routine has been successfully checked through a comparison with finite difference computation (Fri Jan 12 14:40:10 2001)  
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
      \frac{\partial^2 G}{\partial x_i \partial \mu} = \frac{\partial G_i}{\partial \mu} = x_i^{\mu_m-1}(\sum_{i \in C_m} x_i^{\mu_m})^{\frac{\mu}{\mu_m}-1} \left(1+\frac{\mu}{\mu_m} \ln (\sum_{i \in C_m} x_i^{\mu_m})\right)
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
      This routine has been successfully checked through a comparison with finite difference computation (Fri Jan 12 14:45:36 2001)  
   **/

  virtual patReal getSecondDerivative_xi_mu(unsigned long index,
					    const patVariables* x,
					    const patVariables* param,
					    const patReal* mu,
					    const vector<patBoolean>& available,
					    patError*& err) ;

  /** Compute the second partial derivative with respect to one structural parameter and one variable
      If $i \in C_m$, we have
      \[
      \frac{\partial^2 G}{\partial x_i \partial \mu_m} = \frac{\partial G_i}{\partial \mu_m} = \mu y^{\frac{\mu}{\mu_m}-1} x_i^{\mu_m-1} \ln x_i + \mu x_i^{\mu_m-1} y^{\frac{\mu}{\mu_m}-1} \left( \frac{\frac{\mu}{\mu_m}-1}{y} \sum_j x_j^{\mu_m} \ln x_j - \frac{\mu}{\mu_m^2}\ln y\right)
      \]
      where
      \[
      y = \sum_{j\in C_m} x_j^{\mu_m} 
      \]
      If $i \not\in C_m$ we have
      \[
      \frac{\partial^2 G}{\partial x_i \partial \mu_m} = \frac{\partial G_i}{\partial \mu_m} = 0
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
      This routine has been successfully checked through a comparison with finite difference computation (Fri Jan 12 15:09:03 2001)  
   */

  virtual patReal getSecondDerivative_param(unsigned long indexVar,
					    unsigned long indexParam,
					    const patVariables* x,
					    const patVariables* param,
					    const patReal* mu, 
					    const vector<patBoolean>& available,
					    patError*& err) ;
   


  /**
     Read how alternatives are assigned to nests. The specifications are
     available from the global object patModelSpec (based on the singleton
     pattern).
     @param err ref. of the pointer to the error object.  
     @see patModelSpec
  */
  void readNestRepartition(patError*& err) ;
  
  /**
     Assign a name to a nest
     @param nestId nest identifier
     @param name   name to assign
   */
  void addNestName(unsigned long nestId, const patString& name, 
		   patError*& err) ;

  /**
     Assign an alternative to a nest
     @param altid identifier of the alternative
     @param name nest name
   */
  void assignAltToNest(unsigned long altid, const patString& name, 
		       patError*& err) ; 



  /**
     @return "Nested Logit Model"
   */
  virtual patString getModelName() {
    return patString("Nested Logit Model") ;
  }
  
  /**
   */
  virtual unsigned long getNbrParameters() ;

  /**
   * For the sake of efficiency, all derivatives can be computed first and stored.
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
  vector<patString> nestNames ;
  vector<list<unsigned long> > altPerNest ; 
  vector<unsigned long> nestOfAlt ;
  patBoolean firstTime ;
  unsigned long nNests ;
  unsigned long J ;
  vector<patReal> xToMum ;
  vector<patReal> Am ;
  vector<patReal> Bm ;
  vector<patReal> firstDeriv_xi ;
  vector<vector<patReal> > secondDeriv_xi_xj ;
  vector<vector<patReal> > secondDeriv_xi_param ;
  vector<patBoolean> computeDerivativeParam ;
  patBoolean computeMuDerivative ;
  vector<patReal> muDerivative ;

};

#endif
