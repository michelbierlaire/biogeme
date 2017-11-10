//-*-c++-*------------------------------------------------------------
//
// File name : patNetworkGevNode.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Dec 10 18:26:48 2001
//
//--------------------------------------------------------------------

#ifndef patNetworkGevNode_h
#define patNetworkGevNode_h

#include <set>
#include "patError.h"
#include "patGEV.h"
#include "patVariables.h"
#include "patIterator.h"

/**
  @doc This class specializes the interface for GEV models for the special case
  of a node of a Network GEV model.
  class
  @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Mon
  Dec 10 18:27:40 2001) */

class patNetworkGevNode : public patGEV {

public :

  friend ostream& operator<<(ostream &str, const patNetworkGevNode* x) ;



  /** Evaluate the function.
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
		  patError*& err) = PURE_VIRTUAL ;

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
				   patError*& err) = PURE_VIRTUAL ;

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
			   patError*& err) = PURE_VIRTUAL ;


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
			      patError*& err) = PURE_VIRTUAL ;   

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
				    patError*& err) = PURE_VIRTUAL ;

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
				    patError*& err) = PURE_VIRTUAL ;

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
				    patError*& err) = PURE_VIRTUAL ;
   
  /**
   */
  virtual patString getModelName() =  PURE_VIRTUAL ;
  
  /**
   */
  virtual unsigned long getNbrParameters() ;

  /**
     Add a node to the list of successors
     @param aNode pointer to the successor node
     @param index index of the alpha parameter in the model parameters vector
   */
  virtual void addSuccessor(patNetworkGevNode* aNode,unsigned long index) 
    = PURE_VIRTUAL ;
  /**
   */
  virtual unsigned long nSucc() = PURE_VIRTUAL ;

  /**
   */
  virtual patBoolean isRoot() = PURE_VIRTUAL ;

  /**
   */
  virtual patString nodeType() const = PURE_VIRTUAL ;

  /**
   */
  virtual unsigned long getMuIndex() = PURE_VIRTUAL ;

  /**
   */
  virtual ostream& print(ostream& str) = PURE_VIRTUAL ;

  /**
   */
  virtual patIterator<patNetworkGevNode*>* getSuccessorsIterator() = PURE_VIRTUAL ;

  /**
   */
  virtual patBoolean isAlternative() = PURE_VIRTUAL ;

  /**
   */
  virtual patString getNodeName() = PURE_VIRTUAL ;

  /**
     The set contains the user id of relevant alternatives for the node
   */
  virtual std::set<unsigned long> getRelevantAlternatives() = PURE_VIRTUAL ; 

  /**
     Check if an alternative is relevant for the node
   */
  virtual patBoolean isAlternativeRelevant(unsigned long index) ;

  /**
   */
  void setNbrParameters(unsigned long n) ;

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
  unsigned long parameters ;
};

#endif
