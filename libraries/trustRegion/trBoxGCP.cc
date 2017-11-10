//-*-c++-*------------------------------------------------------------
//
// File name : trBoxGCP.cc
// Date :      Mon Nov 27 12:50:53 2000
//
//--------------------------------------------------------------------

#include <fstream>
#include <functional>
#include <numeric>
#include <cassert>
#include <algorithm>
#include "patMath.h"
#include "patDisplay.h"
#include "trBoxGCP.h"
#include "patErrMiscError.h"

trBoxGCP::trBoxGCP(const trBounds& _bounds,
		   patReal _radius,
		   const trVector& _currentIterate,
		   const trVector& _direction,
		   const trVector& _g,
		    trMatrixVector& _H) :
  bounds(_bounds),
  radius(_radius),
  currentIterate(_currentIterate),
  direction(_direction),
  g(_g),
  H(_H)
{

}

trVector trBoxGCP::computeGCP(patULong maxGcpIter,patError*& err) {


//   DEBUG_MESSAGE("COMPUTE GCP") ;
//   DEBUG_MESSAGE("x=" << currentIterate) ;
//   DEBUG_MESSAGE("d=" << direction) ;


  //Step 0: Initialization 
  // Compute the breakpoints

  trBounds::patBreakPointsContainer bp = 
    bounds.getBreakPoints(currentIterate,
			  direction,
			  radius,
			  err) ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return trVector() ;
  }


  // Note that bp is a priority queue.

  trVector d(direction) ;
  trVector s(direction.size(),0.0) ;

  


  // Get the first breakpoint
  assert(!bp.empty()) ;
  trBounds::patBreakPoint nextBreakPoint = bp.top() ;
  bp.pop() ;

  trBounds::patBreakPoint  currentBreakPoint ;
  if (nextBreakPoint.first == 0.0) {
    currentBreakPoint = nextBreakPoint ;
    d[currentBreakPoint.second] = 0.0 ;
  }
  else {
    currentBreakPoint.first = 0.0 ;
    currentBreakPoint.second = d.size() ;
  }

  patReal gd = inner_product(d.begin(),d.end(),g.begin(),0.0) ;

//   DEBUG_MESSAGE("gd= " << gd) ;

  trVector Hd = H(d,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trVector() ;
  }

  //  DEBUG_MESSAGE("Hd= " << Hd) ;

  patULong gcpIter = 0 ;
  
  while (gcpIter <= maxGcpIter) {
    ++gcpIter ;

//     DEBUG_MESSAGE("d=" << d);
//     DEBUG_MESSAGE("s=" << s) ;
//     DEBUG_MESSAGE("x bp = " << currentIterate + s) ;
//     DEBUG_MESSAGE("x next bp = " << s + (nextBreakPoint.first-currentBreakPoint.first) * d) ;
    
    if (currentBreakPoint != nextBreakPoint) {

      // Step 1: compute the slope and curvature
      
      patReal slope = gd + inner_product(Hd.begin(),Hd.end(),s.begin(),0.0) ;
      patReal curvature = inner_product(Hd.begin(),Hd.end(),d.begin(),0.0) ;
      
//        DEBUG_MESSAGE("slope = " << slope) ;
//        DEBUG_MESSAGE("curv  = " << curvature) ;

      // Step 2: Find the next breakpoint
      
      // nextBreakPoint already contains the next breakpoint
      
      // Step 3: Check the current interval for the GCP
      if (slope >= 0.0) {

//  	DEBUG_MESSAGE("Positive slope") ;
// 	DEBUG_MESSAGE("gcp = " << currentIterate + s) ;

	return currentIterate + s ;
      }
      
      if (curvature > 0.0) {
	
	// 		DEBUG_MESSAGE("Positive curvature") ;
	patReal deltaT = -slope / curvature ;
// 	DEBUG_MESSAGE("deltaT = " << deltaT) ;
	if (deltaT < (nextBreakPoint.first - currentBreakPoint.first)) {
	  // 	  DEBUG_MESSAGE("Local minimum found...") ;
// 	  DEBUG_MESSAGE("gcp = " << currentIterate + s + deltaT * d) ;
	  return (currentIterate + s + deltaT * d) ;
	}
	
      }

      // Step 4: Prepare for the next interval
      
      s += (nextBreakPoint.first - currentBreakPoint.first) * d ;

      if (s[nextBreakPoint.second] < 0) {
  	s[nextBreakPoint.second] = 
  	  patMax(bounds.getLower(nextBreakPoint.second,err) - 
  	  currentIterate[nextBreakPoint.second],-radius);
  	if (err != NULL) {
  	  WARNING(err->describe()) ;
  	  return trVector() ;
  	}
      }
      else if (s[nextBreakPoint.second] > 0){
  	s[nextBreakPoint.second] = 
  	  patMin(bounds.getUpper(nextBreakPoint.second,err) - 
  	  currentIterate[nextBreakPoint.second],radius);
  	if (err != NULL) {
  	  WARNING(err->describe()) ;
  	  return trVector() ;
  	}
      }
      
//        if (!bounds.isFeasible(currentIterate+s)) {
//  	DEBUG_MESSAGE("x+s=" << currentIterate+s) ;
//  	DEBUG_MESSAGE("Bounds") ;
//  	cout << bounds ;
//  	err = new patErrMiscError("Unfeasible point generated") ;
//  	WARNING(err->describe()) ;
//  	return trVector();
//        }


      currentBreakPoint = nextBreakPoint ;

      // Update...
    
      // Remove the corresponding direction component
      d[nextBreakPoint.second] = 0.0 ;
      // A smart update of gd and Hd could be done, but it we be done later on,
      // it time allows...
      gd = inner_product(d.begin(),d.end(),g.begin(),0.0) ;
      Hd = H(d,err) ;
      
    }
    else if (bp.empty()) {
      //      DEBUG_MESSAGE("gcp = " << currentIterate + s) ;
      return currentIterate + s ;
    }
    
    // Get the next breakpoint
    
    if (bp.empty()) {
      nextBreakPoint = currentBreakPoint ;
    }
    else {
      assert(!bp.empty()) ;
      nextBreakPoint = bp.top() ;
      bp.pop() ;
    }
  }
  err = new patErrMiscError("This statement should not be reached") ;
  WARNING(err->describe()) ;
  return trVector() ;
}
