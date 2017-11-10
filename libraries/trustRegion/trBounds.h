//-*-c++-*------------------------------------------------------------
//
// File name : trBounds.h
// Date :      Sun Nov 26 15:16:29 2000
//
//--------------------------------------------------------------------

#ifndef trBounds_h
#define trBounds_h

#include <utility>
#include <vector>
#include <queue>
#include "trVector.h"
#include "patError.h"

/**
   @doc Objects of this class define the bounds on the variables.  All variables
   must be subject to bounds constraints. If a bound is meaningless for a
   specific variable, specify a very large upper bound or a very large lower
   bound.
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Sun Nov 26 15:16:29 2000) 
*/
class trBounds {

public:

  /**
   */
  friend ostream& operator<<(ostream &str, const trBounds& x) ;
  /**
   */
  enum patActivityStatus {
    /**
       The variable is at its lower bound
     */
    patLower,
    /**
       The variable is at its upper bound
     */
    patUpper,
    /**
       The variable lies strictly between its bounds
     */
    patFree
  };

  /**
   */
  typedef pair<patReal,unsigned long> patBreakPoint ;

  /**
   */
  typedef priority_queue< patBreakPoint, deque<patBreakPoint > , 
    greater<patBreakPoint > > patBreakPointsContainer ;
 
  
  /**
     @param size number of variables
   */
  trBounds(unsigned long size) ;

  /**
     @param l vector of lower bounds
     @param u vector of upper bounds
   */
  trBounds(const patVariables& l, const patVariables& u) ;

  /**
     Creates a feasible domain wich is the intersection of a given feasible domain and an infinity-norm trust region around x 
     If the constraint on x is l <= x <= u, the bounds are on possible steps, that is l-x <= s <= u-x and on trus region radius -radius <= s <= radius. This domain is defined only in the subspace of the free variables, as defined by activity. 
     @param b initial feasible domain
     @param x center of the trust region
     @param activity activity status of each variable
     @param trustRegionRadius radius of the trust-region
   */
  trBounds(const trBounds& b, 
	   const trVector& x,
	   const vector<patActivityStatus>& activity,
	   patReal trustRegionRadius,
	   patError*& err) ;
  /**
     Defines the bounds
     @param variable index of the variable
     @param lower value of its lower bound
     @param upper value of its upper bound
     @param err ref. of the pointer to the error object.
   */
  void setBounds(unsigned long variable,
		 patReal lower,
		 patReal upper,
		 patError*& err) ;

  /**
     Computes the largest positive step $\alpha$ such that $x + \alpha d$ is
     feasible. If $x$ is not feasible, an error is generated.
     @return value of the largest $\alpha$
     @param x $x$ vector
     @param d $d$ vector
     @param result If NULL, only $\alpha$ is computed. If not NULL, the vector
     $x+\alpha d$ is also computed and stored in the trVector pointed by
     result.
     @param err ref. of the pointer to the error object.
  */
  patReal getMaxStep(const trVector& x,
		     const trVector& d,
		     trVector* result,
		     patError*& err) const ;

  /**
     @param x vector that is checked for feasibility
     @param err ref. of the pointer to the error object.
     @return patTRUE is $x$ is feasible, patFALSE if not.
   */
  patBoolean isFeasible(const trVector& x,
			patError*& err) const ;

  /**
     @return  value of its lower bound
     @param  index index of the variable
     @param err ref. of the pointer to the error object.
   */
  patReal getLower(unsigned long index,
		   patError*& err) const ;
  
  /**
     @return upper value of its upper bound
     @param index index of the variable
     @param err ref. of the pointer to the error object.
   */
  patReal getUpper(unsigned long index,
		   patError*& err) const ;
  
  /**
   */
  unsigned long getDimension() const ;

  /**
     Computes the projection of x onto the feasible domain
   */
  trVector getProjection(const trVector& x,
			 patError*& err) const ;

  /**
     Consider the projection of $x + t d$ on the intersection of the feasible
     domain, and a trust region based on $\|\cdot\|_{\infty}$. This routine
     computes all break points of this piecewise linear path.  It
     returns a vector with a heap structure, enabling to easily access the
     smallest breakpoint.
     @param x reference point (supposed to be feasible)
     @param d search direction
     @param radius trus-region radius
     @param err ref. of the pointer to the error object.
     @return vector containing the breakpoints, and having a heap structure. 
     Use STL routine pop_heap to access its elements in increasing order.
     The elements of this vector are structs with two entries:
     \begin{description}
     \item[first] value of the breakpoint
     \item[second] corresponding index
     \end{description}
  */
  patBreakPointsContainer 
  getBreakPoints(const trVector& x,
		 const trVector& d,
		 patReal radius,
		 patError*& err) const ;

  /**
     Define the activity status of the constraints at x. An error is produced if x is infeasible. A variable is considered at its bound is |x-bound| < patEPSILON. In that case, the value of x is put exactly at its bound.
     @see patActivityStatus
  */
  vector<patActivityStatus> getActivity(trVector& x,
					patError*& err) const ;

  /**
   */
  vector<patReal> getLowerVector() ;

  /**
   */
  vector<patReal> getUpperVector() ;

private:

  vector<patReal> lower ;
  vector<patReal> upper ;
} ;


#endif
