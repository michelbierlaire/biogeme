//-*-c++-*------------------------------------------------------------
//
// File name : patNonLinearProblem.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Apr 26 09:14:48 2001
//
//--------------------------------------------------------------------

#ifndef patNonLinearProblem_h
#define patNonLinearProblem_h

#include "patError.h"
#include "patVariables.h"

class trFunction ;
class trBounds ;

/**
@doc   Virtual object defining an interface for a Nonlinear Programming
   problem:
\[
\min_{x \in \mathbb{R}^n} f(x)
\]
subject to
\[
\begin{array}{rcl}
h(x) & = & 0  \\
g(x) &\leq & 0  \\
A x & = & b \\ 
C x  &\leq&  d \\
   x   & \geq & \ell \\
   x   & \leq & u \\
\end{array}
\]
where 
$f:\mathbb{R}^n \rightarrow \mathbb{R}$, $g:\mathbb{R}^n \rightarrow \mathbb{R}^{m_i}$ and
$h:\mathbb{R}^n \rightarrow \mathbb{R}^{m_e}$ are smooth differentiable nonlinear functions, 
$A \in \mathbb{R}^{k_e} \times \mathbb{R}^n$, 
$C \in \mathbb{R}^{k_i} \times \mathbb{R}^n$,
$b \in \mathbb{R}^{k_e}$,
$d \in \mathbb{R}^{k_i}$,
$\ell \in \mathbb{R}^{n}$, and
$u \in \mathbb{R}^{n}$,

   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Thu Apr 26 09:14:48 2001)
 */

class patNonLinearProblem {

public:
  /**
     number of variables ($n$ in the formulation above)
   */

  virtual unsigned long nVariables() = PURE_VIRTUAL ;

  /**
     number of constraints ($2n + m_i+m_e+k_i+k_e$). The $2n$ correspond to lower and upper bounds on variables.
   */

  virtual unsigned long nConstraints() ;

  /**
     number of non trivial constraints ($m_i+m_e+k_i+k_e$). 
  */
  virtual unsigned long nNonTrivialConstraints() ;

  /**
     number of non linear inequality constraints ($m_i$)
   */
  virtual unsigned long nNonLinearIneq() = PURE_VIRTUAL ;
  /**
     number of non linear equality constraints ($m_e$)
   */
  virtual unsigned long nNonLinearEq() = PURE_VIRTUAL ;
  /**
     number of linear inequality constraints ($k_i$)
   */
  virtual unsigned long nLinearIneq() = PURE_VIRTUAL ;
  /**
     number of linear equality constraints ($k_e$)
   */
  virtual unsigned long nLinearEq() = PURE_VIRTUAL ;

  /**
     vector of lower bounds
   */
  virtual patVariables getLowerBounds(patError*& err) = PURE_VIRTUAL ;

  /**
     vector of upper bounds
   */
  virtual patVariables getUpperBounds(patError*& err) = PURE_VIRTUAL ;

  /**
     objective function $f$
   */
  virtual trFunction* getObjective(patError*& err) = PURE_VIRTUAL ;

  /**
     nonlinear inequality constraint function $g$
   */
  virtual trFunction* getNonLinInequality(unsigned long i,
				    patError*& err) = PURE_VIRTUAL ;

  /**
     nonlinear equality constraint function $h$
   */
  virtual trFunction* getNonLinEquality(unsigned long i,
					patError*& err) = PURE_VIRTUAL ;

  /**
     linear inequality constraint
   */
  virtual pair<patVariables,patReal> 
  getLinInequality(unsigned long i,
		   patError*& err) = PURE_VIRTUAL ;
  /**
     linear equality constraint
   */
  virtual pair<patVariables,patReal> 
  getLinEquality(unsigned long i,
		 patError*& err) = PURE_VIRTUAL ;

  /**
   */
  virtual void setLagrangeLowerBounds(const patVariables& l,
			      patError*& err) ;

  /**
   */
  virtual void setLagrangeUpperBounds(const patVariables& l,
			      patError*& err) ;

  /**
   */
  virtual void setLagrangeNonLinEq(const patVariables& l,
			   patError*& err) ;
  /**
   */
  virtual void setLagrangeLinEq(const patVariables& l,
			patError*& err) ;
  /**
   */
  virtual void setLagrangeNonLinIneq(const patVariables& l,
			     patError*& err) ;
  /**
   */
  virtual void setLagrangeLinIneq(const patVariables& l,
			  patError*& err) ;

  /**
   */
  virtual patVariables getLagrangeLowerBounds() ;

  /**
   */
  virtual patVariables getLagrangeUpperBounds() ;
  /**
   */
  virtual patVariables getLagrangeNonLinEq() ;
  /**
   */
  virtual patVariables getLagrangeLinEq() ;
  /**
   */
  virtual patVariables getLagrangeNonLinIneq() ;
  /**
   */
  virtual patVariables getLagrangeLinIneq() ;

  /**
   */
  virtual patString getProblemName() = PURE_VIRTUAL ;

  /**
   */
  virtual patString getVariableName(unsigned long i,patError*& err) ;

  /**
   */
  virtual patString getNonLinIneqConstrName(unsigned long i,patError*& err) ;

  /**
   */
  virtual patString getNonLinEqConstrName(unsigned long i,patError*& err) ;

  /**
   */
  virtual patString getLinIneqConstrName(unsigned long i,patError*& err) ;

  /**
   */
  virtual patString getLinEqConstrName(unsigned long i,patError*& err) ;

  /**
     This routine uses the KKT conditions to estimate the Lagrange
     multipliers, given the optimal solution of the primal problem.  First, it
     identifies the non-active constraints, for which the associated Lagrange
     multipliers are set to 0. Therefore, all constraints can be considered as equality constraints. If
     \[
        c:\mathbb{R}^n \longrightarrow \mathbb{R}^m
     \]
     is the function representing the constraints, the Lagrange multiplier is obtained by
     \[
     \lambda^* = \left( \nabla c(x^*)^T \nabla c(x^*) \right)^{-1} \nabla c(x^*)^T \nabla f(x^*).
     \]
  */
  virtual void computeLagrangeMultipliers(patVariables& xOpt,
					  patError*& err) ;

  /**
     Compute the number of active constraints at x
   */ 
  virtual unsigned long getNumberOfActiveConstraints(patVariables& x,
					    patError*& err) ;

protected:
  
  patVariables lagrangeLowerBounds ;
  patVariables lagrangeUpperBounds ;
  patVariables lagrangeNonLinEqConstraints ;
  patVariables lagrangeLinEqConstraints ;
  patVariables lagrangeNonLinIneqConstraints ;
  patVariables lagrangeLinIneqConstraints ;


};

#endif

