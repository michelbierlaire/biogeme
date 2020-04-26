//-*-c++-*------------------------------------------------------------
//
// File name : patNetworkGevAlt.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Dec 12 12:48:09 2001
//
//--------------------------------------------------------------------

#ifndef patNetworkGevAlt_h
#define patNetworkGevAlt_h

#include <set>
#include "patError.h"
#include "patNetworkGevNode.h"
#include "patVariables.h"

/**
  @doc This class implement the interface for GEV models for the special case
  of a node of a Network GEV model corresponding to an alternative.
  @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Dec 12 12:48:30 2001) */

class patNetworkGevAlt : public patNetworkGevNode {

public :

  /**
   */
  patNetworkGevAlt(const patString& name,
		   unsigned long muIndex,
		   unsigned long altIndex) ;
  
  /** @memo Evaluate the function.
      @doc Evaluate the function.
      \[
        G^i(x) = x_i^{\mu_i}
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
		  patError*& err)  ;

  /** @memo Compute the partial derivative with respect to one variable.
      @doc Compute the partial derivative with respect to one variable.
      \[
      \frac{\partial G^i}{\partial x_i} = \mu_i x_i^{\mu_i-1}
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
   **/

 patReal getDerivative_xi(unsigned long index,
				   const patVariables* x,
				   const patVariables* param,
				   const patReal* mu,
				   const vector<patBoolean>& available,
				   patError*& err) ;

  /** @memo Compute the partial derivative with respect to mu, the homogeneity factor
      @doc Compute the partial derivative with respect to mu, the homogeneity factor, which is always 0 is this case.
   *  @param x value of the variables. Most of the time, exponential of
   *           utilities.
   *  @param param value of the structural parameters
   *  @param mu value of the homogeneity factor
   *  @param available table with as many entries as the number of variables.
   *      If an entry is patFALSE, the corresponding variables will not be
   *      considered in the sums.
   *  @param err ref. of the pointer to the error object.
   */
  patReal getDerivative_mu(const patVariables* x,
			   const patVariables* param,
			   const patReal* mu,
			   const vector<patBoolean>& available,
			   patError*& err) ;


  /** @memo Compute the partial derivative with respect to one structural parameter
      @doc Compute the partial derivative with respect to one structural parameter
      \[
      \frac{\partial G^i}{\partial \mu_i} = x_i^{\mu_i} \ln x_i
      \]
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

  patReal getDerivative_param(unsigned long index,
			      const patVariables* x,
			      const patVariables* param,
			      const patReal* mu, 
			      const vector<patBoolean>& available,
			      patError*& err)  ;   
  /** @memo Compute the second partial derivative with respect to two variables
      @doc Compute the second partial derivative with respect to two variables
      \[
      \frac{\partial^2 G^i(x)}{\partial x_i^2} = \mu_i (\mu_i-1) x_i^{\mu_i-1}
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
   **/

  patReal getSecondDerivative_xi_xj(unsigned long index1,
				    unsigned long index2,
				    const patVariables* x,
				    const patVariables* param,
				    const patReal* mu,
				    const vector<patBoolean>& available,
				    patError*& err) ;

  /** @memo Compute the second partial derivative with respect to one variable and mu.
      @doc Compute the second partial derivative with respect to one variable and mu, which is always 0 in this case.
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

  patReal getSecondDerivative_xi_mu(unsigned long index,
				    const patVariables* x,
				    const patVariables* param,
				    const patReal* mu,
				    const vector<patBoolean>& available,
				    patError*& err) ;

  /** @memo Compute the second partial derivative with respect to one structural parameter and one variable
      @doc Compute the second partial derivative with respect to one structural parameter and one variable
      \[
      \frac{\partial G^i(x)}{\partial x_i \partial \mu_i} = x_i^{\mu_i-1} (1+\mu_i \ln x_i)
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

  patReal getSecondDerivative_param(unsigned long indexVar,
				    unsigned long indexParam,
				    const patVariables* x,
				    const patVariables* param,
				    const patReal* mu, 
				    const vector<patBoolean>& available,
				    patError*& err) ;
   
  /**
   */
  patString getModelName()  ;
  
  /**
   */
  virtual patString nodeType() const ;

  /**
     Add a node to the list of successors
   */
  void addSuccessor(patNetworkGevAlt* aNode,unsigned long index) ;

  /**
     Add a node to the list of successors. It is here to comply with the general interface for a patNetworkGevNode. In this cintext, calling the function produces a warning and doe snothing else. 
     @param aNode pointer to the successor node
     @param index index of the alpha parameter in the model parameters vector
   */
  virtual void addSuccessor(patNetworkGevNode* aNode,unsigned long index) ;

  /**
   */
  unsigned long nSucc() ;

  /**
   */
  patBoolean isRoot() ;

  /**
   */
  unsigned long getMuIndex() ;


  /**
   */
  ostream& print(ostream& str) ;

  /**
     WARNING: the caller is responsible for releasing the allocated memory 
   */
  patIterator<patNetworkGevNode*>* getSuccessorsIterator() ;

  /**
   */
  unsigned long getNbrParameters() ;

  /**
   */
  patBoolean isAlternative() ;

  /**
   */
  patString getNodeName() ;

  /**
     The list contains the user id of relevant alternatives for the
     node. As the node is an alternative, the list contains only one
     element.
  */
  std::set<unsigned long> getRelevantAlternatives() ; 

private:
  patString nodeName ;
  unsigned long indexOfMu ;
  unsigned long altIndex ;
  std::set<unsigned long> relevant ;
};

#endif
