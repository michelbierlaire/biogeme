//-*-c++-*------------------------------------------------------------
//
// File name : patMaxLikeProblem.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri Apr 27 11:34:05 2001
//
//--------------------------------------------------------------------

#ifndef patMaxLikeProblem_h
#define patMaxLikeProblem_h

#include "patNonLinearProblem.h"
#include "trVector.h"

#include "patMyMatrix.h"

#include "trParameters.h"

class trBounds ;

class patSvd ;

/**
   @doc Defines an interface to a constrained maximum likelihood problem 
   
     Consider the following constrained maximum likelihood problem:
     \[
     \max_{\theta_1,\ldots,\theta_n} \mathcal{L}(\theta_1,\ldots,\theta_n)
     \]
     subject to
     \[
     \begin{array}{rclll}
     \theta_i &\geq& \ell_i & i=1,n &\text{(lower bounds)} \\
     \theta_i  &\leq& u_i & i=1,n &\text{(upper bounds)} \\
     \sum_{j=1}^n a_{ji} \theta_j  &=& b_i & i=1,m_{le} & \text{(linear equality)} \\
     \sum_{j=1}^n c_{ji} \theta_j  &\leq& d_i & i=1,m_{li} & \text{(linear inequality)} \\
     g_i(\theta_1,\ldots,\theta_n) & = & 0 & i=1,m_{ne} &\text{(non linear equality)} \\
     h_i(\theta_1,\ldots,\theta_n) & \leq & 0 & i=1,m_{ni} &\text{(non linear inequality)} \\
     \end{array}
     \]

The following slack variables are introduced:
\begin{itemize}
\item lower bounds: $\xi_i^{lb}, i=1,\ldots,n$
\item upper bounds: $\xi_i^{ub}, i=1,\ldots,n$
\item linear inequalities: $\xi_i^{li}, i=1,\ldots,m_{li}$
\item nonlinear inequalities: $\xi_i^{ni}, i=1,\ldots,m_{ni}$
\end{itemize}
We obtain the following problem with solely equality constraints:
     \[
     \max_{\theta_1,\ldots,\theta_n} \mathcal{L}(\theta_1,\ldots,\theta_n)
     \]
     subject to
     \[
     \begin{array}{rclll}
     \ell_i - \theta_i + (\xi_i^{lb})^2&=& 0 & i=1,n &\text{(lower bounds)} \\
     \theta_i - u_i + (\xi_i^{ub})^2&=& 0 & i=1,n &\text{(upper bounds)} \\
     \sum_{j=1}^n a_{ji} \theta_j - b_i &=& 0 & i=1,m_{le} & \text{(linear equality)} \\
     \sum_{j=1}^n c_{ji} \theta_j - d_i +(\xi_i^{li})^2&=& 0 & i=1,m_{li} & \text{(linear inequality)} \\
     g_i(\theta_1,\ldots,\theta_n) & = & 0 & i=1,m_{ne} &\text{(non linear equality)} \\
     h_i(\theta_1,\ldots,\theta_n) + (\xi_i^{ni})^2 & = & 0 & i=1,m_{ni} &\text{(non linear inequality)} \\
     \end{array}
     \]
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi} (Fri Apr 27 11:34:05 2001)
 */

class patMaxLikeProblem : public patNonLinearProblem {

  

public:

  typedef vector<pair<patVariables,patReal> > patLinearConstraints  ;

  patMaxLikeProblem(trFunction* _f, trBounds* _b, trParameters theTrParameters) ;

  virtual ~patMaxLikeProblem() ;
  /**
     number of variables ($n$ in the formulation above)
   */
  virtual unsigned long nVariables() ;
  /**
     number of non linear inequality constraints ($m_i$)
   */
  virtual unsigned long nNonLinearIneq() ;
  /**
     number of non linear equality constraints ($m_e$)
   */
  virtual unsigned long nNonLinearEq() ;
  /**
     number of linear inequality constraints ($k_i$)
   */
  virtual unsigned long nLinearIneq() ;
  /**
     number of linear equality constraints ($k_e$)
   */
  virtual unsigned long nLinearEq() ;

  /**
     vector of lower bounds
   */
  virtual patVariables getLowerBounds(patError*& err) ;

  /**
     vector of upper bounds
   */
  virtual patVariables getUpperBounds(patError*& err) ;

  /**
     objective function $f$
   */
  virtual trFunction* getObjective(patError*& err) ;

  /**
     nonlinear inequality constraint function $g$
   */
  virtual trFunction* getNonLinInequality(unsigned long i,
				    patError*& err) ;

  /**
     nonlinear equality constraint function $h$
   */
  virtual trFunction* getNonLinEquality(unsigned long i,
					patError*& err) ;

  /**
     linear inequality constraint
   */
  virtual pair<patVariables,patReal> 
  getLinInequality(unsigned long i,
		   patError*& err) ;
  /**
     linear equality constraint
   */
  virtual pair<patVariables,patReal> 
  getLinEquality(unsigned long i,
		 patError*& err) ;
  
  /**
     Add a non linear equality constraint
     @return number assigned to the constraint
   */
  int addNonLinEq(trFunction* f) ;

  /**
     Add a non linear inequality constraint
     @return number assigned to the constraint
   */
  int addNonLinIneq(trFunction* f) ;

  /**
     Add a linear equality constraint
     @return number assigned to the constraint
   */
  int addLinEq(const patVariables& a, patReal b) ;

  /**
     Add a linear inequality constraint
     @return number assigned to the constraint
   */
  int addLinIneq(const patVariables& a, patReal b) ;

  /**
   */
  patString getProblemName() ;
  
  /** 
   */
  patBoolean isFeasible(trVector& x, patError*& err) const ; 

  /**
   */ 
  unsigned long getSizeOfVarCovar() ;

  /**
The variance-covariance matrix is computed from the second derivatives matrix of the Lagrangian function.

The Lagrangian of the optimization problem is 
\[
\begin{array}{rcl}
L(\theta,\xi,\lambda) &=& \mathcal{L}(\theta_1,\ldots,\theta_n) \\*[2em]
  &+& \displaystyle\sum_{j=1}^n \lambda_j^{lb} (\ell_j - \theta_j + (\xi_j^{lb})^2) \\*[2em]
  &+& \displaystyle\sum_{j=1}^n \lambda_j^{ub} (\theta_j - u_j + (\xi_j^{ub})^2) \\*[2em]
  &+& \displaystyle\sum_{j=1}^{m_{le}} \lambda_j^{le} (\sum_{k=1}^n a_{kj} \theta_k - b_j) \\*[2em]
  &+& \displaystyle\sum_{j=1}^{m_{li}} \lambda_j^{li} (\sum_{k=1}^n c_{kj} \theta_k - d_j + (\xi_j^{li})^2) \\*[2em]
  &+& \displaystyle\sum_{j=1}^{m_{ne}} \lambda_j^{ne} g_j(\theta_1,\ldots,\theta_n) \\*[2em]
  &+& \displaystyle\sum_{j=1}^{m_{ni}} \lambda_j^{ne} (h_j(\theta_1,\ldots,\theta_n)+(\xi_j^{ni})^2) \\
\end{array}
\]
The first derivatives with respect to $\theta_k$ parameters are
\[
\frac{\partial L}{\partial \theta_k} = \frac{\partial \mathcal{L}}{\partial \theta_k} - \lambda_k^{lb} + \lambda_k^{ub} + \sum_{j=1}^{m_{le}} \lambda_j^{le} a_{kj} + \sum_{j=1}^{m_{li}}  \lambda_j^{li} c_{kj}+ \sum_{j=1}^{m_{ne}}  \lambda_j^{ne} \frac{\partial g_j}{\partial \theta_k}+ \sum_{j=1}^{m_{ni}}  \lambda_j^{ni} \frac{\partial h_j}{\partial \theta_k}
\]
The first derivatives with regard to $\xi_k^{\text{XX}}$, where $\text{XX} \in \{lb,ub,li,ni\}$ are
\[
\frac{\partial L }{\partial \xi_k^{\text{XX}}} = 2 \lambda_k^{\text{XX}} \xi_k^{\text{XX}}
\]
The first derivatives with respect to $\lambda$ are the constraints, that is
\[
\begin{array}{rcl}
\frac{\partial L}{\partial \lambda_k^{lb}} &=& \ell_k - \theta_k + (\xi_k^{lb})^2 \\*[2em]
\frac{\partial L}{\partial \lambda_k^{ub}} &=& \theta_k - u_k + (\xi_k^{ub})^2 \\*[2em]
\frac{\partial L}{\partial \lambda_k^{le}} &=& \displaystyle\sum_{j=1}^n a_{jk} \theta_j - b_k \\*[2em]
\frac{\partial L}{\partial \lambda_k^{li}} &=& \displaystyle\sum_{j=1}^n c_{jk} \theta_j - d_k + (\xi_k^{li})^2 \\*[2em]
\frac{\partial L}{\partial \lambda_k^{ne}} &=& g_k(\theta_1,\ldots,\theta_n) \\*[2em]
\frac{\partial L}{\partial \lambda_k^{ni}} &=& h_k(\theta_1,\ldots,\theta_n)+(\xi_k^{ni})^2
\end{array}
\]
     The second derivative matrix is 


\[
\begin{array}{c|ccccc|cccccc}
 & n & n & n & m_{li} & m_{ni} & n & n & m_{le} & m_{li} & m_{ne} & m_{ni} \\
\hline
n      & \Sigma & 0     & 0 & 0 & 0 & L     & U     & A & C     & G & H     \\
n      &   0    & \Lambda^{lb}     & 0 & 0 & 0 & \Xi^{lb} & 0     & 0 & 0     & 0 & 0     \\
n      &   0    & 0     & \Lambda^{ub} & 0 & 0 & 0     & \Xi^{ub} & 0 & 0     & 0 & 0     \\
m_{li} &   0    & 0     & 0 & \Lambda^{li} & 0 & 0     &   0   & 0 & \Xi^{li} & 0 & 0     \\
m_{ni} &   0    & 0     & 0 & 0 & \Lambda^{ni} & 0     &   0   & 0 & 0     & 0 & \Xi^{ni} \\
\hline
n      &   L^T  & \Xi^{lb} & 0 & 0 & 0 & 0     &   0   & 0 & 0     & 0 & 0     \\  
n      &   U^T  & 0 & \Xi^{ub} & 0 & 0 & 0     &   0   & 0 & 0     & 0 & 0     \\  
m_{le} &   A^T  & 0 &  0 & 0 & 0 & 0     &   0   & 0 & 0     & 0 & 0     \\  
m_{li} &   C^T  & 0 &  0 & \Xi^{li} & 0 & 0     &   0   & 0 & 0     & 0 & 0     \\  
m_{ne} &   G^T  & 0 &  0 & 0  & 0 & 0     &   0   & 0 & 0     & 0 & 0     \\  
m_{ni} &   H^T  & 0 &  0 & 0  & \Xi^{ni} & 0     &   0   & 0 & 0     & 0 & 0    
\end{array}
\]
     where
\[
\Sigma_{kp} = \frac{\partial^2 \mathcal{L}}{\partial \theta_k \partial \theta_p}+\sum_{j=1}^{m_{ne}} \lambda_j^{ne} \frac{\partial^2 g}{\partial \theta_k \partial \theta_p} + \sum_{j=1}^{m_{ni}} \lambda_j^{ni} \frac{\partial^2 h}{\partial \theta_k \partial \theta_p}
\]
\[
L_{ij} = \left\{   
\begin{array}{rl}
 -1 & \text{if } i=j, \\
  0 & \text{otherwise}.
\end{array}
\right. 
\]
\[
U_{ij} = \left\{   
\begin{array}{rl}
 1 & \text{if } i=j, \\
  0 & \text{otherwise}.
\end{array}
\right. 
\]
\[
A_{ij} = a_{ij}, \; i=1,\ldots,n, \; j=1,\ldots,m_{le}.
\]
\[
C_{ij} = c_{ij}, \; i=1,\ldots,n, \; j=1,\ldots,m_{le}.
\]
\[
G_{ij} = \frac{\partial g_j}{\partial \theta_i}, \; i=1,\ldots,n, \; j=1,\ldots,m_{ne}.
\]
\[
H_{ij} = \frac{\partial h_j}{\partial \theta_i}, \; i=1,\ldots,n, \; j=1,\ldots,m_{ni}.
\]
If $\text{XX}\in\{lb,ub,li,ni\}$, then
\[
\Xi^{\text{XX}}_{ij} = \left\{   
\begin{array}{rl}
 2 \xi_i^{\text{XX}} & \text{if } i=j, \\
  0 & \text{otherwise}.
\end{array}
\right. 
\]
\[
\Lambda^{\text{XX}}_{ij} = \left\{   
\begin{array}{rl}
 2 \lambda_i^{\text{XX}} & \text{if } i=j, \\
  0 & \text{otherwise}.
\end{array}
\right. 
\]

If
$
\mathcal{L}_A^{-1} = 
\left( 
\begin{array}{cc}
   P  &  Q \\
   Q^T  &  R  
\end{array}
\right)
$
then $P$ is the variance-covariance matrix of the parameters ($n$ parameters $\theta$ and $2n+m_{li}+m_{ni}$  slack variables), and $-R$ is the variance-covariance matrix of the $2n+m_{li}+m_{ni}+m_{le}+m_{ne}$ Lagrange multipliers.



 @see Silvey S. D. (1975) Statistical Inference, Monographs on Applied Probability and Statistics, Chapman and Hall, London. (pp. 80-81)

@see Schoenberg, R. (1997) Constrained Maximum Likelihood, Computational Economics, 10: 251-266.


     The robust variance covariance matrix is computed if the pointer
     robustVarCovar is non null
     . It is defined as 
     \[ H^{-1}
     \text{BHHH} H^{-1} 
     \] where $H$ is the second derivative matrix
     of the loglikelihood, and BHHH is \[ \sum_n \nabla g_n (\nabla
     g_n)^T \] and $g_n$ is the gradient of the probability model for
     individual $n$ in the sample. The constraints are ignored in this
     case

   */

  // Should return a LazyMatrix instead of a Matrix, but I am not sure
  // to fully understand how to use LazyMatrices. Should be
  // investigated in the future.  If the matrix is not invertible, the
  // routines computes the igenvector corresponding to the zero
  // eigenvalue.

patBoolean computeVarCovar(trVector* x,
			   patMyMatrix* varCovar,
			   patMyMatrix* robustVarCovar,
			   map<patReal,patVariables>* eigVec,
			   patReal* smallSV,
			   patError*& err) ;

  

protected:
private:
  trFunction* f ;
  vector<trFunction*> nonLinEqConstraints ;
  patLinearConstraints linEqConstraints ;
  patLinearConstraints linIneqConstraints ;
  trBounds* bounds ;
  patSvd* svd ;
  patMyMatrix* bigMatrix ;
  patMyMatrix* matHess ;
  
  trParameters theTrParameters ;
};

#endif 
