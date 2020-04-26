//-*-c++-*------------------------------------------------------------
//
// File name : patAbsTime.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Dec 21 14:43:39 1998
//
//--------------------------------------------------------------------
//
//
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef patAbsTime_h
#define patAbsTime_h

#include <patString.h>

#ifdef TIME_WITH_SYS_TIME
#include <sys/time.h>
#include <time.h>
#else
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#else
#include <time.h>
#endif
#endif

#ifdef HAVE_IOSTREAM
#include <iostream>
#endif

#include "patConst.h"


class patTimeInterval ;

/**
 @doc patTimeStringFormat describes possibles format for writing/reading a time
 as a string
*/

enum patTimeStringFormat{
  /**
     Example: Thu Oct 17 17:30:20 1996
   */
  patTsfFULL,    
  /**
     Example : 17:30:20
   */
  patTsfHMS,
  /**
     08/12/08 13:20:17
   */
  patAlogit
}  ;



/**
   @doc Objectives: defines an absolute representation of time, using the Unix
   standard, that is the number of seconds since 00:00:00 UTC January 1st,
   1970. 
   Warning: because of the time difference, this can have different meanings
   depending on the time zone. For example, in MIT's time zone, it is the
   number of seconds since 19:00:00 on December 31st, 1969.
   
   This representation allows times between
     December 13th, 1901 at 20:45:52 and
     January  19th, 2038 at 03:14:07 
   represented respectively (in our time) zone by -2147465648 and -2147465649

   the longTime variable has the following structure:
   struct    tm {
               int  tm_sec;    seconds after the minute - [0, 61] 
                                    for leap seconds 
               int  tm_min;    minutes after the hour - [0, 59] 
               int  tm_hour;   hour since midnight - [0, 23] 
               int  tm_mday;   day of the month - [1, 31] 
               int  tm_mon;    months since January - [0, 11] 
               int  tm_year;   years since 1900 
               int  tm_wday;   days since Sunday - [0, 6] 
               int  tm_yday;   days since January 1 - [0, 365] 
               int  tm_isdst;  flag for alternate daylight savings time 
   };

   The value of tm_isdst is positive if daylight savings time is in effect,
     zero if daylight savings time is not in effect, and negative if the
     information is not available.

 Sources: Unix man page: ctime(3C) 
       @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Mon Dec 21 14:43:39 1998)
*/

class patAbsTime {
/**
 */
    friend patAbsTime operator+(patAbsTime t1, patAbsTime t2) ;
/**
 */
    friend patBoolean operator<(const patAbsTime& t1, const patAbsTime& t2) ;
/**
 */
    friend ostream& operator<<(ostream& stream, const patAbsTime& time) ;

friend patBoolean operator==(const patAbsTime& t1, const patAbsTime& t2)  ;
friend patBoolean operator>(const patAbsTime& t1, const patAbsTime& t2) ;
friend patBoolean operator!=(const patAbsTime& t1, const patAbsTime& t2) ;
friend patBoolean operator<=(const patAbsTime& t1, const patAbsTime& t2) ;
friend patBoolean operator>=(const patAbsTime& t1, const patAbsTime& t2) ;
friend time_t operator-(const patAbsTime& t1, const patAbsTime& t2) ;

public:


  /**
   */
  patAbsTime() ;

  /**
   */
  ~patAbsTime() ;
  
/**
 */
  patAbsTime(struct tm p) ;
/**
 */
  patAbsTime(time_t p) ;

/**
 */
    void setTimeOfDay() ;

/**
 */
    void setTime(long day, long month, long year, long hour, long min, long sec) ;

/**
 */
  void setTime(time_t p) ;
/**
 */
  void setTime(struct tm p) ;
/**
 */
  patUnitTime getSeconds() const ;

  /**
   */
  time_t getUnixFormat() const ;

/**
 */
    string getTimeString(patTimeStringFormat format=patTsfHMS) const;
/**
 */
  patAbsTime& operator+=(time_t t) ;

  private:
    time_t theTime ;
};


const patAbsTime patInvalidTime(0L) ;

#endif // patAbsTime_h
