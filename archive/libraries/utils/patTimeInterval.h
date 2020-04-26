//-*-c++-*------------------------------------------------------------
//
// File name : patTimeInterval.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Jun 12 16:11:13 2000
//
// Modification history:
//
// Date            Author         Description
// ======          ======         ============
//
//--------------------------------------------------------------------

#ifndef patTimeInterval_h
#define patTimeInterval_h

#include "patAbsTime.h"

#include "patString.h"
#include "patDisplay.h"
#include "patConst.h"


#include <iostream>

/**
  @doc patTimeStringFormat describes possibles format for writing/reading a time
  as a string
 */
enum patTimeIntervalStringFormat {
  /**
     For instance [Mon Jun 12 16:11:40 2000,Mon Jun 12 16:11:49 2000]
   */
  patTIsfFULL,  
  /**
     For instance, [16:11:40,16:11:49] 
   */
  patTIsfHMS   
}  ;


/**
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Mon Jun 12 16:11:13 2000)
 */
class patTimeInterval {


      /**
       */
    friend ostream& operator<<(ostream& stream, const patTimeInterval& ti) ;
      /**
       */
    friend patBoolean operator==(const patTimeInterval& t1, 
				 const patTimeInterval& t2) ;
      /**
       */
    friend patBoolean operator<(const patTimeInterval& t1, 
				const patTimeInterval& t2) ;
      /**
       */
	friend patBoolean operator>(const patTimeInterval& t1, 
				const patTimeInterval& t2) ;

public:
  /**
     Default constructor
   */
  patTimeInterval() : patTIOpenBracket("["),
		      patTICloseBracket("]"),
		      patTISeparator(","),
		      start(0), end(0) {} ;
  /**
     Copy constructor
   */
  patTimeInterval(const patTimeInterval& ti) : patTIOpenBracket("["),
					       patTICloseBracket("]"),
					       patTISeparator(","),
					       start(ti.start), end(ti.end) {} ;
  /**
     @param s start of the interval
     @param e end of the interval
   */
    patTimeInterval(const patAbsTime& s, const patAbsTime& e):
      patTIOpenBracket("["),
      patTICloseBracket("]"),
      patTISeparator(",") {
      setTimeInterval(s,e) ;
    }
  /**
     @param s start of the interval
     @param e end of the interval
   */
    patTimeInterval(patUnitTime s, patUnitTime e) : patTIOpenBracket("["),
      patTICloseBracket("]"),
      patTISeparator(",")  {
      setTimeInterval(s,e) ;
    }
  /**
     Generate the interval [middle-epd,middle+eps]
     @param middle middle of the interval
     @param half the length of the interval
   */
    patTimeInterval(const patAbsTime& middle, patUnitTime eps) : patTIOpenBracket("["),
      patTICloseBracket("]"),
      patTISeparator(",")  {
      setTimeInterval(middle, eps) ;
    }

  /**
     Dtor
   */
  virtual ~patTimeInterval() {}

  /**
     Set the start of the interval
   */
    virtual void setStart(const patAbsTime& t) ;
  /**
     Set the end of the interval
   */
    virtual void setEnd(const patAbsTime& t) ;
  /**
   */
    virtual patAbsTime getStart() const {return start;}
  /**
   */
    virtual void setStart(patUnitTime x) { start = x ; } 
  /**
   */
    virtual patAbsTime getEnd() const {return end; }
  /**
   */
    virtual void setEnd(patUnitTime x) { end = x ; } 
  /**
   */
    virtual patAbsTime getAbsStart() const ;
  /**
   */
    virtual patAbsTime getAbsEnd() const ;
  /**
   */
  virtual patUnitTime getLengthInSeconds() const ;
  /**
   */
    virtual patString getTimeString(patTimeIntervalStringFormat format=patTIsfHMS) const;
  /**
   */
    virtual void setTimeInterval(patUnitTime s, patUnitTime e) {
  /**
   */
      start = time_t(s) ; end = time_t(e) ;
    }
  /**
   */
    virtual void setTimeInterval(const patAbsTime& s, const patAbsTime& e) ;

  /**
   */
  virtual patString getLength() const ;

  /**
   */
    virtual void setTimeInterval(const patAbsTime& middle, patUnitTime eps) ;
  /**
     @return patTRUE if t is within the interval
   */
    virtual patBoolean IsContaining(const patAbsTime& t) const ;

  /**
   * This function checks is the time interval is containing time t,
   * without considering the day. Only hours, minutes, seconds are
   * considered.
   */

//     virtual patBoolean IsWithinDayContaining(const patAbsTime& t) const ;


  /**
     @return Previous interval, that is, if current interval is [s,e], 
     return [2s-e,s]
   */
    patTimeInterval Previous() const ;
  /**
     @return Next interval, that is, if current interval is [s,e], returns [e,2e-s]
   */
  patTimeInterval Next() const ;
  
private: 
  /**
 */
  patString patTIOpenBracket  ;
  patString patTICloseBracket  ;
  /**
 */
  patString patTISeparator  ;

  /**
   */
  patAbsTime start ; 
  /**
   */
  patAbsTime end ; 
};



#endif // patTimeInterval_h

