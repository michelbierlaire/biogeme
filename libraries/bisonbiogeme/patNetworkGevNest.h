//-*-c++-*------------------------------------------------------------
//
// File name : patNetworkGevNest.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Tue Jan  8 05:40:00 2002
//
//--------------------------------------------------------------------

#ifndef patNetworkGevNest_h
#define patNetworkGevNest_h

#include "patError.h"
#include "patNetworkGevNode.h"
#include "patVariables.h"

/**
  @doc This class implement the interface for GEV models for the special case
  of a node of a Network GEV model. All nodes except those capturing
  alternatives are considered. For the alternatives, use the patNetworkAlt
  class
  @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Tue Jan  8 05:40:07 2002) */

class patNetworkGevNest : public patNetworkGevNode {

public :


  /**
     @param name Node name 
     @param ind Index of the node parameter in the model
     parameter vector. patBadId for the root, because mu is the parameter of
     the root node, and mu is handled separately 
 */
  patNetworkGevNest(const patString& name, unsigned long ind = patBadId) ;

  /** @memo Evaluate the function.
      @doc Evaluate the function.
   \[
G^i(x) = \sum_{j \in \text{succ}(i)} \alpha_{(i,j)} G^j(x)^{\frac{\mu_i}{\mu_j}}
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
      \frac{\partial G^i(x)}{\partial x_k} = \sum_j \alpha_{ij} \frac{\mu_i}{\mu_j} G^j(x)^{\frac{\mu_i}{\mu_j}-1} \frac{\partial G^j(x)}{\partial x_k}
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

 virtual patReal getDerivative_xi(unsigned long index,
				   const patVariables* x,
				   const patVariables* param,
				   const patReal* mu,
				   const vector<patBoolean>& available,
				   patError*& err) ;

  /** @memo Compute the partial derivative with respect to mu, the homogeneity factor
      @doc Compute the partial derivative with respect to mu, the homogeneity factor
      It is 0.0 if the node is not the root. It if it is the root, it is     
   \[
        \frac{\partial G^i}{\partial \mu} = \sum_j \frac{\alpha_{ij}}{\mu_j} G^j(x)^{\frac{\mu}{\mu_j}} \ln G^j(x)
        \]
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
			   patError*& err) ;


  /** @memo Compute the partial derivative with respect to one structural parameter
   @doc Compute the partial derivative with respect to one structural parameter
     \[
   \frac{\partial G^i(x)}{\alpha_{ij}} = G^j(x)^{\frac{\mu_i}{\mu_j}}
\]
 \[
   \frac{\partial G^i(x)}{\partial \mu_i} = \sum_j \frac{\alpha_{ij}}{\mu_j} G^j(x)^{\frac{\mu_i}{\mu_j}} \ln G^j(x)
 \]
   If node $k$ is a successor of node $i$, then  
   \[
    \frac{\partial G^i(x)}{\mu_k} =  \alpha_{ik} G^k(x)^{\frac{\mu_i}{\mu_k}-1} \frac{\mu_i}{\mu_k}\left( \frac{\partial G^k(x)}{\partial \mu_k} - \frac{1}{\mu_k} G^k(x)\ln G^k(x) \right)
   \]
   If not, then
   \[
\frac{\partial G^i(x)}{\mu_k} = \sum_j \alpha_{ij} \frac{\mu_i}{\mu_j} G^j(x)^{\frac{\mu_i}{\mu_j}-1} \frac{\partial G^j(x)}{\partial \mu_k}
   \]
   \[
    \frac{\partial G^i(x)}{\partial \alpha_{ij}} = G^j(x)^{\frac{\mu_i}{\mu_j}}
   \]
For any other parameter $\gamma$, we have
\[
\frac{\partial G^i(x)}{\partial \gamma} = \sum_j \alpha_{ij} \frac{\mu_i}{\mu_j} G^j(x)^{\frac{\mu_i}{\mu_j}-1} \frac{\partial G^j(x)}{\partial \gamma}
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

  virtual patReal getDerivative_param(unsigned long index,
			      const patVariables* x,
			      const patVariables* param,
			      const patReal* mu, 
			      const vector<patBoolean>& available,
			      patError*& err)  ;   
  /** @memo Compute the second partial derivative with respect to two variables.
     @doc Compute the second partial derivative with respect to two variables.
     \[
       \frac{\partial^2 G^i(x)}{\partial x_k \partial x_m} = \sum_j \alpha_{ij} \frac{\mu_i}{\mu_j}(\frac{\mu_i}{\mu_j}-1) G^j(x)^{\frac{\mu_i}{\mu_j}-2} \frac{\partial G^j(x)}{\partial x_k} \frac{\partial G^j(x)}{\partial x_m} + \alpha_{ij} \frac{\mu_i}{\mu_j}G^j(x)^{\frac{\mu_i}{\mu_j}-1} \frac{\partial^2 G^j(x)}{\partial x_k \partial x_m}
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

  virtual patReal getSecondDerivative_xi_xj(unsigned long index1,
				    unsigned long index2,
				    const patVariables* x,
				    const patVariables* param,
				    const patReal* mu,
				    const vector<patBoolean>& available,
				    patError*& err) ;

  /** @memo Compute the second partial derivative with respect to one variable and mu.    
      @doc Compute the second partial derivative with respect to one variable and mu.    
      \[
      \frac{\partial^2 G^i(x)}{\partial x_k \partial \mu} = \sum_j \frac{\alpha_{ij}}{\mu_j}  G^j(x)^{\frac{\mu}{\mu_j}-1} \frac{\partial G^j(x)}{\partial x_k}\left( \frac{\mu}{\mu_j} \ln G^j(x)  +  1  \right) 
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

  virtual patReal getSecondDerivative_xi_mu(unsigned long index,
				    const patVariables* x,
				    const patVariables* param,
				    const patReal* mu,
				    const vector<patBoolean>& available,
				    patError*& err) ;

  /** @memo Compute the second partial derivative with respect to one
      structural parameter and one variable 
      @doc Compute the second
      partial derivative with respect to one structural parameter and
      one variable \[ \frac{\partial^2 G^i(x)}{\partial x_k \partial
      \mu_i} = \sum_j \frac{\alpha_{ij}}{\mu_j}
      G^j(x)^{\frac{\mu_i}{\mu_j}-1} \frac{\partial G^j(x)}{\partial
      x_k}\left( \frac{\mu_i}{\mu_j} \ln G^j(x) + 1 \right) \]

\[
\frac{\partial^2 G^i(x)}{\partial x_k \partial \alpha_{ij}} = \frac{\mu_i}{\mu_j} G^j(x)^{\frac{\mu_i}{\mu_j}-1} \frac{\partial G^j(x)}{\partial x_k}
\]

If node $j$ is a successor of node $i$, then
\[
\frac{\partial^2 G^i(x)}{\partial x_k \partial \mu_j} = \alpha_{ij} \frac{\mu_i}{\mu_j} G^j(x)^{\frac{\mu_i}{\mu_j}-1} \left( \frac{\partial^2 G^j}{\partial x_k \partial \mu_j} - \frac{1}{\mu_j} \frac{\partial G^j(x)}{\partial x_k} + \frac{\partial G^j(x)}{\partial x_k} \left(G^j(x)^{-1} (\frac{\mu_i}{\mu_j}-1) - \ln G^j(x) \frac{\mu_i}{\mu_j^2} \right)\right)
\] 

If not, then 
\[
\frac{\partial^2 G^i(x)}{\partial x_k \partial \mu_\ell} = \sum_j \alpha_{ij} \frac{\mu_i}{\mu_j} G^j(x)^{\frac{\mu_i}{\mu_j}-2} \left((\frac{\mu_i}{\mu_j}-1) \frac{\partial G^j(x)}{\partial \mu_\ell} + G^j(x) \frac{\partial^2 G^j(x)}{\partial x_k \partial \mu_\ell} \right)
\]
For any other parameter $\gamma$, we have
\[
\frac{\partial^2 G^i(x)}{\partial x_k \partial \gamma} = \sum_j \alpha_{ij} \frac{\mu_i}{\mu_j} (\frac{\mu_i}{\mu_j}-1) G^j(x)^{\frac{\mu_i}{\mu_j}-2} \frac{\partial G^j(x)}{\partial x_k}\frac{\partial G^j(x)}{\partial \gamma} + \alpha_{ij} \frac{\mu_i}{\mu_j} G^j(x)^{\frac{\mu_i}{\mu_j}-1} \frac{\partial^2 G^j(x)}{\partial x_k \partial \gamma}
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
   */
  virtual patString getModelName()  ;
  
  /**
   */
  virtual patString nodeType() const ;

  /**
     Add a node to the list of successors
     @param aNode pointer to the successor node
     @param index index of the alpha parameter in the model parameters vector
   */
  virtual void addSuccessor(patNetworkGevNode* aNode,unsigned long index) ;

  /**
   */
  virtual unsigned long nSucc() ;

  /**
   */
  virtual unsigned long getMuIndex() ;

  /**
   */
  virtual patBoolean isRoot() ;

  /**
   */
  ostream& print(ostream& str) ;

  /**
     WARNING: the caller is responsible for releasing the allocated memory
   */
  virtual patIterator<patNetworkGevNode*>* getSuccessorsIterator() ;

  /**
   */
  virtual patBoolean isAlternative() ;

  /**
   */
  virtual patString getNodeName() ;
  /**
     The list contains the user id of relevant alternatives for the node
   */
  virtual std::set<unsigned long> getRelevantAlternatives() ; 

protected:
  // Index of the mu parameter for the node in the vector of parameters
  patString nodeName ;
  unsigned long indexOfMu ;
private:
  /**
   */
  virtual unsigned long getNbrParameters() ;

  // Indices of the alpha parameters in the vector of parameters
  vector<unsigned long> indexOfAlpha ;
  vector<patNetworkGevNode*> listOfSuccessors ;
  patBoolean relevantComputed ;
  std::set<unsigned long> relevant ;
  
};

#endif
