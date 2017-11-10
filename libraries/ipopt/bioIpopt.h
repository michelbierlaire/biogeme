//-*-c++-*------------------------------------------------------------
//
// File name : bioIpopt.h
// Author :    Michel Bierlaire
// Date :      Mon Mar 20 09:50:31 2017
//
//--------------------------------------------------------------------

#ifndef bioIpopt_h
#define bioIpopt_h 

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

/**
@doc This class defines a C++ interface to the IPOPT routine, by Andreas Waechter et al.

*/


#include "patError.h"
#include "trNonLinearAlgo.h"

#ifdef IPOPT
#include "IpTNLP.hpp"
#include "IpIpoptApplication.hpp"
using namespace Ipopt;
#endif


class patNonLinearProblem ;
class patIterationBackup ;
class trHessian ;

class bioIpopt : public trNonLinearAlgo
#ifdef IPOPT
	       , public TNLP
#endif
{

public:
  /**
   */
  bioIpopt(patIterationBackup* i, patNonLinearProblem* aProblem = NULL) ;
  /**
   */
  virtual ~bioIpopt() ;
  
  // Methods required by IPOPT
#ifdef IPOPT  
  virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
			    Index& nnz_h_lag, IndexStyleEnum& index_style);
  
  /** Method to return the bounds for my problem */
  virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u,
			       Index m, Number* g_l, Number* g_u);
  
  /** Method to return the starting point for the algorithm */
  virtual bool get_starting_point(Index n, bool init_x, Number* x,
				  bool init_z, Number* z_L, Number* z_U,
				  Index m, bool init_lambda,
				  Number* lambda);
  
  /** Method to return the objective value */
  virtual bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value);
  
  /** Method to return the gradient of the objective */
  virtual bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f);
  
  /** Method to return the constraint residuals */
  virtual bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g);
  
  /** Method to return:
   *   1) The structure of the jacobian (if "values" is NULL)
   *   2) The values of the jacobian (if "values" is not NULL)
   */
  virtual bool eval_jac_g(Index n, const Number* x, bool new_x,
			  Index m, Index nele_jac, Index* iRow, Index *jCol,
			  Number* values);
  
  /** Method to return:
   *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
   *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
   */
  virtual bool eval_h(Index n, const Number* x, bool new_x,
		      Number obj_factor, Index m, const Number* lambda,
		      bool new_lambda, Index nele_hess, Index* iRow,
		      Index* jCol, Number* values);
  
  //@}
  
  /** @name Solution Methods */
  //@{
  /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
  virtual void finalize_solution(SolverReturn status,
				 Index n, const Number* x, const Number* z_L, const Number* z_U,
				 Index m, const Number* g, const Number* lambda,
				 Number obj_value,
				 const IpoptData* ip_data,
				 IpoptCalculatedQuantities* ip_cq);  

#endif
  
  // Methods required by Biogeme
  /**
     @return Diagnostic from algorithm
   */
  virtual patString run(patError*& err)  ;
  /**
     @return number of iterations. If there is any error, 0 is returned. 
   */
  virtual patULong nbrIter()  ;
  /**
   */
  virtual patVariables getSolution(patError*& err)  ;
  /**
   */
  virtual patReal getValueSolution(patError*& err)  ;
  /**
   */
  virtual patVariables getLowerBoundsLambda()  ;
  /**
   */
  virtual patVariables getUpperBoundsLambda()  ;
  /**
   */
  virtual void defineStartingPoint(const patVariables& x0)  ;

  /**
   */
  virtual patString getName()  ;

  virtual patBoolean isAvailable() const ;
  

  
private:
#ifdef IPOPT
  patString getFinalStatus() ;
  SmartPtr<IpoptApplication> app ;
  SolverReturn finalStatus ;
#endif
  patULong n ;
  patVariables startingPoint ;
  patBoolean startingPointDefined ;
  patVariables solution ;
  patReal solutionValue ;
  patVariables solutionGradient ;
  patVariables lowerLambda ;
  patVariables upperLambda ;

  patIterationBackup* theInteraction ;
  patString stopFile ;

  patBoolean exactHessian ;
  trHessian* theHessian ;
};



#endif
