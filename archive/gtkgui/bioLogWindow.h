//-*-c++-*------------------------------------------------------------
//
// File name : bioLogWindow.h
// Author :    Michel Bierlaire
// Date :      Mon Apr  4 17:46:27 2016
//
//--------------------------------------------------------------------

#ifndef bioLogWindow_h
#define bioLogWindow_h

#include <gtkmm.h>

#include <gtkmm/window.h>
#include <gtkmm/textview.h>
#include <gtkmm/scrolledwindow.h>
#include <gtkmm/frame.h>

#include "patLogMessage.h"

class bioLogWindow: public patLogMessage {
 public: 
  bioLogWindow(Glib::RefPtr<Gtk::Builder> b) ;
  ~bioLogWindow() ;
  virtual void addLogMessage(patString m) ;
  void initLog() ;
  void show(Glib::ustring title,
	    Glib::ustring directory,
	    Glib::ustring modelFileName,
	    Glib::ustring dataFileName) ;
  void show() ;
  void hide() ;
  void activateOkButton() ;
protected:
			   
  Gtk::MessageDialog *pMessageDialog ;
  Gtk::TextView *theLog ;
  Glib::RefPtr<Gtk::TextBuffer> theBuffer ;
  Glib::RefPtr<Gtk::Builder> refBuilder ;
  Gtk::Label *pTitle ;
  Gtk::Label *pDirectory ;
  Gtk::Label *pModelFile ;
  Gtk::Label *pDataFile ;
  Gtk::Button *pOkButton ;
  Gtk::Switch* pSwitchButtonLogWindow ;
} ;

#endif
