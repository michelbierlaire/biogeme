//-*-c++-*------------------------------------------------------------
//
// File name : bioGtkScriptDialog.cc
// Author :    Michel Bierlaire
// Date :      Tue Mar 21 12:33:25 2017
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include <iostream>
#include <giomm.h>
#include <gtkmm/messagedialog.h>
#include "patDisplay.h"
#include "patVersion.h"
#include "patExecScript.h"
#include "patGetWorkingDir.h"
#include "patErrNullPointer.h"

#include <gtkmm/label.h>
#include "bioGtkScriptDialog.h"

bioGtkScriptDialog::bioGtkScriptDialog(Glib::RefPtr<Gtk::Builder> b,
				       patError*& err) :
  refBuilder(b),
  pDialog(NULL),
  pCancelButton(NULL),
  pRunButton(NULL),
  pArg1(NULL),
  pArg2(NULL),
  pArg3(NULL),
  pWdLabel(NULL),
  pScriptFileChooser(NULL),
  pFileArg1(NULL),
  pFileArg2(NULL),
  pFileArg3(NULL),
  pWorkingDirectoryChooser(NULL),
  theLog(NULL) {

  refBuilder->get_widget("scriptDialog", pDialog);
  if (pDialog == NULL){
    err = new patErrNullPointer("Gtk::Dialog") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("scriptCancel",pCancelButton) ;
  if (pCancelButton == NULL){
    err = new patErrNullPointer("Gtk::Button") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("scriptRun",pRunButton) ;
  if (pRunButton == NULL){
    err = new patErrNullPointer("Gtk::Button") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("scriptFileChooser",pScriptFileChooser);
  if (pScriptFileChooser == NULL){
    err = new patErrNullPointer("Gtk::FileChooserButton") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("fileArg1",pFileArg1);
  if (pFileArg1 == NULL){
    err = new patErrNullPointer("Gtk::FileChooserButton") ;
    WARNING(err->describe()) ;
    return ;
  }

  refBuilder->get_widget("fileArg2",pFileArg2);
  if (pFileArg2 == NULL){
    err = new patErrNullPointer("Gtk::FileChooserButton") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("fileArg3",pFileArg3);
  if (pFileArg3 == NULL){
    err = new patErrNullPointer("Gtk::FileChooserButton") ;
    WARNING(err->describe()) ;
    return ;
  }


refBuilder->get_widget("workingDirectoryChooser",pWorkingDirectoryChooser);
  if (pWorkingDirectoryChooser == NULL){
    err = new patErrNullPointer("Gtk::FileChooserButton") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("wdLabel",pWdLabel);
  if (pWdLabel == NULL){
    err = new patErrNullPointer("Gtk::Label") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("arg1",pArg1) ;
  if (pArg1 == NULL) {
    err = new patErrNullPointer("Gtk::Entry") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("arg2",pArg2) ;
  if (pArg2 == NULL) {
    err = new patErrNullPointer("Gtk::Entry") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("arg3",pArg3) ;
  if (pArg3 == NULL) {
    err = new patErrNullPointer("Gtk::Entry") ;
    WARNING(err->describe()) ;
    return ;
  }

  refBuilder->get_widget("scriptTextView", theLog);
  if (theLog == NULL) {
    err = new patErrNullPointer("Gtk::TextView") ;
    WARNING(err->describe()) ;
    return ;
  }
  theBuffer = Gtk::TextBuffer::create() ;
  if (theLog) {
    theLog->set_buffer(theBuffer) ;
  }

  if (pDialog) {
    Gtk::Assistant *pAssistant ;
    refBuilder->get_widget("mainAssistant", pAssistant);
    pDialog->set_transient_for(*pAssistant) ;

    stringstream str ; 
    str << patVersion::the()->getVersionInfo() << ": script" ;
    pDialog->set_title(str.str()) ;
  }
  
  pCancelButton->signal_clicked().connect(sigc::mem_fun(*this,&bioGtkScriptDialog::hide)) ;
  pRunButton->signal_clicked().connect(sigc::mem_fun(*this,&bioGtkScriptDialog::run)) ;
  pRunButton->set_sensitive(false) ;
  pScriptFileChooser->signal_file_set().connect(sigc::mem_fun(*this,&bioGtkScriptDialog::scriptFileButtonSelected)) ;
  pFileArg1->signal_file_set().connect(sigc::mem_fun(*this,&bioGtkScriptDialog::setFile1)) ;
  pFileArg2->signal_file_set().connect(sigc::mem_fun(*this,&bioGtkScriptDialog::setFile2)) ;
  pFileArg3->signal_file_set().connect(sigc::mem_fun(*this,&bioGtkScriptDialog::setFile3)) ;

#ifndef HAVE_CHDIR
  Glib::ustring label("This version has been compiled on a system without chdir. Impossible to modify the working directory.") ;
  pWdLabel->set_label(label) ;
  pWorkingDirectoryChooser->set_sensitive(false) ;
#endif
  
    pWorkingDirectoryChooser->signal_file_set().connect(sigc::mem_fun(*this,&bioGtkScriptDialog::workingDirectoryButtonSelected)) ;

    patGetWorkingDir g ;
    workingDirectory = g() ;
    pWorkingDirectoryChooser->set_current_folder(workingDirectory) ;

    // Check if the execution of a scripts kills the current
    // executable, in order to warn the user
    vector<patString> dummyCommand ;
    patExecScript dummyScript(dummyCommand) ;
    if (dummyScript.killAfterRun()) {
      stringstream  s ;
      s << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl ;
      s << "****            IMPORTANT WARNING        ****" << endl ;
      s << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl ;
      s << "This version has been compiled on a system (such as Windows) that does not allow to fork processes." << endl ;
      s << "Therefore, if you run a script, it will kill the current program before executing the script." << endl ;
      s << "Although the script wil be executed, it is therefore recommended to run the script from a terminal." << endl ;
      s << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl ;
      printToBuffer(s.str()) ;
    }
}

void bioGtkScriptDialog::hide() {
  if (pDialog) {
    pDialog->hide() ;
  }
}



bioGtkScriptDialog::~bioGtkScriptDialog() {
  hide() ;
}

void bioGtkScriptDialog::show() {
  if (pDialog) {
    pDialog->show() ;
  }
}

void bioGtkScriptDialog::printToBuffer(patString s) {

  Gtk::TextBuffer::iterator endIter ;
  if (theBuffer) {
    endIter = theBuffer->end() ; 
    theBuffer->insert(endIter,Glib::ustring(s)) ;
    endIter = theBuffer->end() ; 
    theBuffer->insert(endIter,Glib::ustring("\n")) ;
    endIter = theBuffer->end() ;
  }

  if (theLog) {
    theLog->scroll_to(endIter) ;
  }
  // The following loop is necessary to allow the display to be
  // updated as the program is running
  
  while (Gtk::Main::events_pending ()) {
    Gtk::Main::iteration();
  }

}

void bioGtkScriptDialog::run() {
  scriptFileName = pScriptFileChooser->get_filename() ;
  arg1 = pArg1->get_text() ;
  arg2 = pArg2->get_text() ;
  arg3 = pArg3->get_text() ;

#ifdef HAVE_CHDIR
  chdir(workingDirectory.c_str()) ;
#endif
  patAbsTime now ;
  vector<patString> theCommand ;
  theCommand.push_back(scriptFileName) ;
  if (arg1 != "") {
    theCommand.push_back(arg1) ;
  }
  if (arg2 != "") {
    theCommand.push_back(arg2) ;
  }
  if (arg3 != "") {
    theCommand.push_back(arg3) ;
  }
  stringstream command ;
  command << scriptFileName << " " << arg1 << " " << arg2 << " " << arg3 ;
  stringstream str ;
  str << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl ;
  str << "Running '" << command.str() << "'" << endl ;
  str << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl ;
  str << now.getTimeString(patTsfFULL) << endl ;
  patGetWorkingDir g ;
  str << "Directory: " << g() << endl ;
  str << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl ;
  printToBuffer(str.str()) ;
  patExecScript theScript(theCommand) ;
  theScript.run() ;
  printToBuffer(theScript.getOutput()) ;
}

void bioGtkScriptDialog::scriptFileButtonSelected() {
  pRunButton->set_sensitive(true) ;
}

void bioGtkScriptDialog::workingDirectoryButtonSelected() {
  workingDirectory = pWorkingDirectoryChooser->get_filename() ;
}


void bioGtkScriptDialog::setFile1() {
  pArg1->set_text(pFileArg1->get_filename()) ;
  setWorkingDirectory (Glib::path_get_dirname(pFileArg1->get_filename())) ;
}

void bioGtkScriptDialog::setFile2() {
  pArg2->set_text(pFileArg2->get_filename()) ;
  setWorkingDirectory (Glib::path_get_dirname(pFileArg2->get_filename())) ;
}

void bioGtkScriptDialog::setFile3() {
  pArg3->set_text(pFileArg3->get_filename()) ;
  setWorkingDirectory (Glib::path_get_dirname(pFileArg3->get_filename())) ;
}

void bioGtkScriptDialog::setWorkingDirectory(patString s) {
  workingDirectory = s ;
  pWorkingDirectoryChooser->set_current_folder(workingDirectory) ;
}
