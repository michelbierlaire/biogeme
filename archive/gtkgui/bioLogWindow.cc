//-*-c++-*------------------------------------------------------------
//
// File name : bioLogWindow.cc
// Author :    Michel Bierlaire
// Date :      Mon Apr  4 17:49:58 2016
//
//--------------------------------------------------------------------

#include <iostream>
#include "bioLogWindow.h"
#include "patDisplay.h"
#include "patVersion.h"
#include <pangomm/fontdescription.h>
#include <gtkmm/main.h>

bioLogWindow::bioLogWindow(Glib::RefPtr<Gtk::Builder> b) :
  pMessageDialog(NULL),
  theLog(NULL),
  refBuilder(b),
  pTitle(NULL),
  pDirectory(NULL),
  pModelFile(NULL),
  pDataFile(NULL),
  pOkButton(NULL),
  pSwitchButtonLogWindow(NULL) {
  
  refBuilder->get_widget("logMessageDialog", pMessageDialog);

  if (pMessageDialog) {
    stringstream str ; 
    str << patVersion::the()->getVersionInfo() << " running" ;
    pMessageDialog->set_title(str.str()) ;
  }
  Gtk::Assistant *pAssistant ;
  refBuilder->get_widget("mainAssistant", pAssistant);
  pMessageDialog->set_transient_for(*pAssistant) ;
  refBuilder->get_widget("logTextView", theLog);
  theBuffer = Gtk::TextBuffer::create() ;
  if (theLog) {
    theLog->set_buffer(theBuffer) ;
  }
  refBuilder->get_widget("switchButtonLogWindow",pSwitchButtonLogWindow) ;
  
  stringstream str ;
  // str << patVersion::the()->getVersionInfo() << ": log messages" ;
  // set_title(str.str()) ;

  //set_policy(Gtk::POLICY_AUTOMATIC,Gtk::POLICY_AUTOMATIC) ;
  //  set_hexpand(TRUE) ;

  refBuilder->get_widget("okButtonRun", pOkButton);
  if (pOkButton) {
    pOkButton->set_sensitive(FALSE) ;
    pOkButton->signal_clicked().connect(sigc::mem_fun(*this,&bioLogWindow::hide)) ;
  }

  if (pSwitchButtonLogWindow) {
    pSwitchButtonLogWindow->set_sensitive(FALSE) ;
  }

}

void bioLogWindow::activateOkButton() {
  if (pOkButton) {
    pOkButton->set_sensitive(TRUE) ;
  }
}

bioLogWindow::~bioLogWindow() {

  hide() ;
  DELETE_PTR(theLog);
  DELETE_PTR(pMessageDialog) ;
}


void bioLogWindow::initLog() {
  patDisplay::the().setLogMessage(this) ;
}

void bioLogWindow::show(Glib::ustring title,
			Glib::ustring directory,
			Glib::ustring modelFileName,
			Glib::ustring dataFileName) {
  
  refBuilder->get_widget("pythonBisonConfirmLabel", pTitle);
  if (pTitle) {
    pTitle->set_label(title) ;
  }
  refBuilder->get_widget("directoryConfirm", pDirectory);
  if (pDirectory) {
    pDirectory->set_label(directory) ;
  }

  refBuilder->get_widget("modelFileConfirm", pModelFile);
  if (pModelFile) {
    pModelFile->set_label(modelFileName) ;
  }

  refBuilder->get_widget("dataFileConfirm", pDataFile);
  if (pDataFile) {
    pDataFile->set_label(dataFileName) ;
  }
  
  pSwitchButtonLogWindow->set_sensitive(TRUE) ;
  if (pMessageDialog) {
    pMessageDialog->show() ;
  }

}
void bioLogWindow::hide() {
  if (pMessageDialog) {
    pMessageDialog->hide() ;
    if (pSwitchButtonLogWindow) {
      pSwitchButtonLogWindow->set_state(false) ;
    }
  }
}

void bioLogWindow::show() {
  if (pMessageDialog) {
    pMessageDialog->show() ;
    if (pSwitchButtonLogWindow) {
      pSwitchButtonLogWindow->set_state(true) ;
    }
  }
}



void bioLogWindow::addLogMessage(patString m) {
  Gtk::TextBuffer::iterator endIter ;
  if (theBuffer) {
    endIter = theBuffer->end() ; 
    theBuffer->insert(endIter,Glib::ustring(m)) ;
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


