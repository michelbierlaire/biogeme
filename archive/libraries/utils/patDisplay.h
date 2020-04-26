//-*-c++-*------------------------------------------------------------
//
// File name : patDisplay.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Jun  2 22:40:38 2003
//
//--------------------------------------------------------------------

#ifndef patDisplay_h
#define patDisplay_h

#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <memory>
#include <mutex>
using namespace std ;


#include "patString.h"
#include "patImportance.h"
#include "patMessage.h"

class patLogMessage ;

#define DEBUG_MESSAGE(message)  \
{stringstream __str1,__str2,__str3 ; __str1 << message ;\
  __str2 << __FILE__  ; __str3 << __LINE__ << " " ;				\
patDisplay::the().addMessage(patImportance::patDEBUG,patString(__str1.str()), \
			      patString(__str2.str()),patString(__str3.str())) ;} 

#define GENERAL_MESSAGE(message)  \
{stringstream __str1,__str2,__str3 ; __str1 << message ;\
__str2 << __FILE__  ; __str3 << __LINE__ << " ";\
patDisplay::the().addMessage(patImportance::patGENERAL,patString(__str1.str()), \
			      patString(__str2.str()),patString(__str3.str())) ;} 

#define DETAILED_MESSAGE(message)  \
{stringstream __str1,__str2,__str3 ; __str1 << message ;\
__str2 << __FILE__  ; __str3 << __LINE__ << " ";\
patDisplay::the().addMessage(patImportance::patDETAILED,patString(__str1.str()), \
			      patString(__str2.str()),patString(__str3.str())) ;} 

#define WARNING(message)  \
{stringstream __str1,__str2,__str3 ; __str1 << message ;\
__str2 << __FILE__ ; __str3 << __LINE__ << " ";\
patDisplay::the().addMessage(patImportance::patWARNING, patString(__str1.str()), \
patString(__str2.str()),patString(__str3.str())) ; }

#define FATAL(message)  \
{stringstream __str1,__str2,__str3 ; __str1 << message ;\
__str2 << __FILE__ ; __str3 << __LINE__ ;\
patDisplay::the().addMessage(patImportance::patFATAL, patString(__str1.str()), \
			      patString(__str2.str()),patString(__str3.str())) ; exit(-1); }




/**
@doc This class collects all messages to be displayed and dispatch
them to the screen and/or to files according to their importance. The following levels of importance are considered.

1 Main messages, describing the main steps of the program
2 Detailed messages
3 Debug messages

 @author Michel Bierlaire, EPFL (Mon Jun  2 22:40:38 2003)

*/



class patDisplay {
 public:

  /**
     @return pointer to the single instance of the class
   */
  static patDisplay& the() ;

  /**
   */
  ~patDisplay() ;

  /**
   */

  void addMessage(const patImportance& aType,
		  const patString& text,
		  const patString& filename,
		  const patString& lineNumber) ;

  /**
   */
  void initProgressReport(const patString message,
			  unsigned long upperBound) ;

  /**
   */
  patBoolean updateProgressReport(unsigned long currentValue) ;

  /**
   */
  void terminateProgressReport() ;

  /**
   */
  void setScreenImportanceLevel(const patImportance& aType) ;
  
  /**
   */
  void setLogImportanceLevel(const patImportance& aType) ;

  /**
   */
  void setLogMessage(patLogMessage* up) ;

 private:
  patDisplay() ;
  patDisplay(const patDisplay& td) ;
  patDisplay& operator=(const patDisplay& td) ;
  void initLogFile() ;

private:
  // static std::shared_ptr<patDisplay> singleInstance ;
  // static std::once_flag only_one ;
 
 private:
  patImportance screenImportance ;
  patImportance logImportance ;
  vector<patMessage> messages ;
  ofstream logFile ;
  patString logFileName ;

  patLogMessage* logMessage ;
};

#endif
