//-*-c++-*------------------------------------------------------------
//
// File name : patTimeInterval.cc
// Author :    Michel Bierlaire
// Date :      Mon Jun 12 16:13:58 2000
//
// Modification history:
//
// Date                     Author            Description
// ======                   ======            ============
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <iomanip>
#include "patDisplay.h"
#include "patAbsTime.h"
#include "patTimeInterval.h"



patString patTimeInterval::getTimeString(patTimeIntervalStringFormat format) const{
 
  patString output ;
  patAbsTime s(start) ;
  patAbsTime e(end) ;
  switch (format) {
    case patTIsfFULL:
    output = patTIOpenBracket ;
    output += s.getTimeString(patTsfFULL) ;
    output += patTISeparator ;
    output += e.getTimeString(patTsfFULL) ;
    output += patTICloseBracket ;
    cout << "--> time interval " << output << endl ;
    return output ;
    case patTIsfHMS:
    default:
    output = patTIOpenBracket ;
    output += s.getTimeString(patTsfHMS) ;
    output += patTISeparator ;
    output += e.getTimeString(patTsfHMS) ;
    output += patTICloseBracket ;
    cout << "--> time interval " << output << endl ;
    return output ;
  }
}


patBoolean patTimeInterval::IsContaining(const patAbsTime& t) const {
  time_t w = t.getSeconds() ;
  patBoolean res = (w >= start && w < end) ;
  if (!res) {
    // Check if the result would be true if we don't take the day into account
    patAbsTime s(start) ;
    patAbsTime e(end) ;
  }
  return(res) ;
}


void patTimeInterval::setTimeInterval(const patAbsTime& s, const patAbsTime& e)
{
  start= s.getUnixFormat() ; end = e.getUnixFormat() ;
}


void patTimeInterval::setTimeInterval(const patAbsTime& middle, patUnitTime eps) {
  start= middle.getSeconds()-eps ; end = middle.getSeconds()+eps ;

}


ostream& operator<<(ostream& stream, const patTimeInterval& ti) {

  stream << ti.getTimeString() ;
  return stream ;
}


patBoolean operator==(const patTimeInterval& t1, 
			     const patTimeInterval& t2) {
  if (t1.getStart() != t2.getStart())  return patFALSE ;
  return(t1.getEnd() == t2.getEnd()) ;
}


patBoolean operator<(const patTimeInterval& t1, 
		     const patTimeInterval& t2) {
  if (t1.getStart() == t2.getStart()) {
    return (t1.getEnd() < t2.getEnd()) ;
  }
  return (t1.getStart() < t2.getStart()) ;
}

patBoolean operator>(const patTimeInterval& t1, 
		     const patTimeInterval& t2) {
  if (t1.getStart() == t2.getStart()) {
    return (t1.getEnd() > t2.getEnd()) ;
  }
  return (t1.getStart() > t2.getStart()) ;
}


patAbsTime patTimeInterval::getAbsStart() const {
  patAbsTime res(start) ;
  return res ;
}

patAbsTime patTimeInterval::getAbsEnd() const {
  patAbsTime res(end) ;
  return res ;
}

patTimeInterval patTimeInterval::Previous() const {
  return(patTimeInterval(start+start-end, start)) ;
}

patTimeInterval patTimeInterval::Next() const {
  return (patTimeInterval(end,end+end-start)) ;
}

void patTimeInterval::setStart(const patAbsTime& t) {
  setStart(t.getSeconds()) ;
}

void patTimeInterval::setEnd(const patAbsTime& t) {
  setEnd(t.getSeconds()) ;
}


patString patTimeInterval::getLength() const {

  patReal tmp = difftime(end.getUnixFormat(),start.getUnixFormat()) ;

  int seconds = int(tmp) % 60 ;
  
  tmp -= seconds ;
  tmp /= 60 ;
  
  int minutes = int(tmp) % 60 ;
  tmp -= minutes ;
  tmp /= 60 ;

  int hours = int(tmp) % 24 ;
  tmp -= hours ;
  tmp /= 24 ;
    
  stringstream str ;
  
  patBoolean start = patFALSE; 
  if (tmp > 0) {
    start = patTRUE ;
  }
  if (start) {
    str << tmp << " day" ;
    if (tmp > 1) {
      str << "s" ;
    }
    str << " " ;
  }

  if (hours > 0) {
    start = patTRUE ;
  }
  if (start) {
    str << setfill('0') << setw(2) << hours << "h ";
  }

  str << setfill('0') << setw(2) << minutes 
      << ":" << setfill('0') << setw(2) << seconds;
  
  patString res(str.str()) ;
  return res ;
  

}

patUnitTime patTimeInterval::getLengthInSeconds() const {
  return (end - start) ;
}
