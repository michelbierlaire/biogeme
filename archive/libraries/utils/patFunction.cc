#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patFunction.h"
#include "patDisplay.h"

patReal patFunction::operator()(const patVariables& x, 
				patError*&  err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    // There is something magic with this number ;-)
    return 0.90267 ;
  }

  map<patVariables,patReal,less<patVariables> >::const_iterator f = 
    functionValue.find(x) ;
  if (f != functionValue.end()) {
    return ((*f).second) ;
  }
  patReal value = evaluate(x,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    // There is something magic with this number ;-)
    return 0.90267 ;
  }
  functionValue[x] = value ;
  return value ;
  
}

 void patFunction::reset() {
  functionValue.erase(functionValue.begin(),functionValue.end()) ;
}

 unsigned long patFunction::getNbrEval() {
  return functionValue.size() ;
}
