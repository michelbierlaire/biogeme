//-*-c++-*------------------------------------------------------------
//
// File name : patAbsTime.cc
// Author :    Michel Bierlaire
// Date :      Mon Dec 21 14:42:26 1998
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <locale>

#include "patConst.h"
#include "patDisplay.h"
#include "patAbsTime.h"


patAbsTime::patAbsTime() {
  setTimeOfDay() ;
}

patAbsTime::~patAbsTime() {
}

patAbsTime::patAbsTime(struct tm p) {
  setTime(p);
}

patAbsTime::patAbsTime(time_t p) {
  setTime(p) ;
}

void patAbsTime::setTime(time_t p) {
  theTime = p ; 
}

void patAbsTime::setTime(struct tm p) {
#if defined(HAVE_CONFIG_H) && ! defined(HAVE_MKTIME)
  cerr << "Your system does not support the time function mktime." << endl ;
#else
  theTime = mktime(&p) ;
#endif 

} 

patUnitTime patAbsTime::getSeconds() const {
  return (patUnitTime)time;
} 

patAbsTime& patAbsTime::operator+=(time_t t) {
  theTime += t ; 
  return *this;
}
 

void patAbsTime::setTimeOfDay() {

//#ifdef HAVE_TIME
  theTime = time(&theTime) ;
//#else
//  cannot use WARNING here, it uses the time and causes a seg fault
//  WARNING("Your system does not support the function time") ;
//#endif

};


patString patAbsTime::getTimeString(patTimeStringFormat format) const {
  struct tm * timeinfo;
//#ifdef HAVE_LOCALTIME
  timeinfo = localtime ( &theTime );
//#else 
//  cannot use WARNING here, it uses the time and causes a seg fault
//  WARNING("Your system does not support the function localtime") ;
//#endif
  struct tm timeb = *timeinfo;

  // For convenience
  typedef std::ostreambuf_iterator<char,  std::char_traits<char> > Iter;

  // Get a time_put facet
  const std::time_put<char, Iter> &tp = 
    std::use_facet<std::time_put<char, Iter> >( std::locale ());

  stringstream theAlogitTime ;
  Iter begin (theAlogitTime);
  const char hr[] = "%H";
  const char mn[] = "%M";
  const char sc[] = "%S";

  const char da[] = "%d";
  const char mo[] = "%m";
  const char ye[] = "%y";

  const char full[] = "%c";

  switch (format) {
  case patAlogit:
    tp.put (begin, theAlogitTime, ' ', &timeb, 
            mo + 0, mo + sizeof mo - 1);
    theAlogitTime << '/';
    tp.put (begin, theAlogitTime, ' ', &timeb, 
	    da + 0, da + sizeof da - 1);
    theAlogitTime << '/';
    tp.put (begin, theAlogitTime, ' ', &timeb, 
	    ye + 0, ye + sizeof ye - 1);
    theAlogitTime << ' ';
    tp.put (begin, theAlogitTime, ' ', &timeb, 
            hr + 0, hr + sizeof hr - 1);
    theAlogitTime << ':';
    tp.put (begin, theAlogitTime, ' ', &timeb, 
	    mn + 0, mn + sizeof mn - 1);
    theAlogitTime << ':';
    tp.put (begin, theAlogitTime, ' ', &timeb, 
	    sc + 0, sc + sizeof sc - 1);
    
    return patString(theAlogitTime.str()) ;
    break ;
  case patTsfHMS :
    
    tp.put (begin, theAlogitTime, ' ', &timeb, 
            hr + 0, hr + sizeof hr - 1);
    theAlogitTime << ':';
    tp.put (begin, theAlogitTime, ' ', &timeb, 
	    mn + 0, mn + sizeof mn - 1);
    theAlogitTime << ':';
    tp.put (begin, theAlogitTime, ' ', &timeb, 
	    sc + 0, sc + sizeof sc - 1);
    
    return patString(theAlogitTime.str()) ;
  case patTsfFULL:
    
    tp.put (begin, theAlogitTime, ' ', &timeb, 
            full + 0, full + sizeof full - 1);
    return patString(theAlogitTime.str()) ;
  }
  return patString() ;
}


ostream& operator<<(ostream& stream, const patAbsTime& t) {
  stream << t.getTimeString() ;
  return stream ;
}




patAbsTime operator+(patAbsTime t1, patAbsTime t2) {
  time_t t = t1.theTime + t2.theTime ;
  patAbsTime x(t) ;
  return x ;
}


patBoolean operator<(const patAbsTime& t1, const patAbsTime& t2) {
  return (t1.theTime < t2.theTime ) ;
}

patBoolean operator==(const patAbsTime& t1, const patAbsTime& t2) {
  return ( ((t1 < t2) == patFALSE) &&  ((t2 < t1) == patFALSE)) ;
}


patBoolean operator>(const patAbsTime& t1, const patAbsTime& t2) {
  return (t2 < t1) ;
}

patBoolean operator!=(const patAbsTime& t1, const patAbsTime& t2) {
  return !(t1 == t2) ;
}

patBoolean operator<=(const patAbsTime& t1, const patAbsTime& t2) {
  return !(t1 > t2) ;
}

patBoolean operator>=(const patAbsTime& t1, const patAbsTime& t2) {
  return !(t1 < t2) ;
}


time_t patAbsTime::getUnixFormat() const {
  return theTime ;
}

time_t operator-(const patAbsTime& t1, const patAbsTime& t2) {
  return t1.theTime - t2.theTime ;
}
