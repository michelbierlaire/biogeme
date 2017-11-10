//-*-c++-*------------------------------------------------------------
//
// File name : bioGtkScriptDialog.h
// Author :    Michel Bierlaire
// Date :      Tue Mar 21 11:03:54 2017
//
//--------------------------------------------------------------------

#ifndef bioGtkScriptDialog_h
#define bioGtkScriptDialog_h

#include <vector>
#include <gtkmm.h>
#include <gdkmm/types.h>

#include "patError.h"

class bioGtkScriptDialog {
public:
  bioGtkScriptDialog(Glib::RefPtr<Gtk::Builder> b,
		     patError*& err) ;
  ~bioGtkScriptDialog() ;
  void show() ;
  void hide() ;
  void run() ;
  void scriptFileButtonSelected() ;
  void workingDirectoryButtonSelected() ;
  void setFile1() ;
  void setFile2() ;
  void setFile3() ;
protected:
  Glib::RefPtr<Gtk::Builder> refBuilder ;
  Gtk::Dialog *pDialog ;
  Gtk::Button *pCancelButton ;
  Gtk::Button *pRunButton ;
  Gtk::Entry *pArg1 ;
  Gtk::Entry *pArg2 ;
  Gtk::Entry *pArg3 ;
  Gtk::Label *pWdLabel ;
  Gtk::FileChooserButton *pScriptFileChooser ;
  Gtk::FileChooserButton *pFileArg1 ;
  Gtk::FileChooserButton *pFileArg2 ;
  Gtk::FileChooserButton *pFileArg3 ;
  Gtk::FileChooserButton *pWorkingDirectoryChooser ;
  Glib::ustring scriptFileName ;
  Glib::ustring arg1 ;
  Glib::ustring arg2 ;
  Glib::ustring arg3 ;
  Gtk::TextView *theLog ;
  Glib::RefPtr<Gtk::TextBuffer> theBuffer ;

  // File descriptors for pipe.
  int fds[2];
  
private:
  void initWindow() ;
  void printToBuffer(patString s) ;
  patString workingDirectory ;
  void setWorkingDirectory(patString s) ;
};
#endif
