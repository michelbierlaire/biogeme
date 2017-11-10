//-*-c++-*------------------------------------------------------------
//
// File name : bioGtkFileViewer.cc
// Author :    Michel Bierlaire
// Date :      Wed Apr 27 08:22:28 2016
//
//--------------------------------------------------------------------

#include <giomm.h>
#include <gtkmm/messagedialog.h>
#include "patDisplay.h"
#include "patVersion.h"
#include "patErrNullPointer.h"
#include "bioFilePackage.h"
//
// WARNING. the content type "text/plain" is only valid in unix. In windows, ot should be "*.txt". 

// see https://developer.gnome.org/gio/stable/gio-GContentType.html

#include <gtkmm/label.h>
#include "bioGtkFileViewer.h"
bioGtkFileViewer::bioGtkFileViewer(Glib::RefPtr<Gtk::Builder> b, patError*& err) :
  refBuilder(b),
  pDialog(NULL),
  pCriticalGrid(NULL),
  pUsefulGrid(NULL),
  pDebugGrid(NULL),
  pCriticalFrame(NULL),
  pUsefulFrame(NULL),
  pDebugFrame(NULL),
  pOkButton(NULL),
  pSwitchButtonFileViewer(NULL) {

  refBuilder->get_widget("outputFileDialog", pDialog);
  if (pDialog == NULL) {
    err = new patErrNullPointer("Gtk::Dialog") ;
    WARNING(err->describe());
    return ;
  }
  
  refBuilder->get_widget("criticalGrid", pCriticalGrid);
  if (pCriticalGrid == NULL) {
    err = new patErrNullPointer("Gtk::Grid") ;
    WARNING(err->describe());
    return ;
  }
  
  refBuilder->get_widget("criticalFrame", pCriticalFrame);
  if (pCriticalFrame == NULL) {
    err = new patErrNullPointer("Gtk::Frame") ;
    WARNING(err->describe());
    return ;
  }

  pCriticalFiles = new bioFilePackage(pDialog,pCriticalGrid) ;

  refBuilder->get_widget("usefulGrid", pUsefulGrid);
  if (pUsefulGrid == NULL) {
    err = new patErrNullPointer("Gtk::Grid") ;
    WARNING(err->describe());
    return ;
  }

  refBuilder->get_widget("usefulFrame", pUsefulFrame);
  if (pUsefulFrame == NULL) {
    err = new patErrNullPointer("Gtk::Frame") ;
    WARNING(err->describe());
    return ;
  }

  pUsefulFiles = new bioFilePackage(pDialog,pUsefulGrid) ;

  refBuilder->get_widget("debugGrid", pDebugGrid);
  if (pDebugGrid == NULL) {
    err = new patErrNullPointer("Gtk::Grid") ;
    WARNING(err->describe());
    return ;
  }

  refBuilder->get_widget("debugFrame", pDebugFrame);
  if (pDebugFrame == NULL) {
    err = new patErrNullPointer("Gtk::Frame") ;
    WARNING(err->describe());
    return ;
  }

  pDebugFiles = new bioFilePackage(pDialog,pDebugGrid) ;

  refBuilder->get_widget("outputFilesDialogOk", pOkButton);
  if (pOkButton == NULL) {
    err = new patErrNullPointer("Gtk::Button") ;
    WARNING(err->describe());
    return ;
  }


  
  refBuilder->get_widget("switchButtonFileViewer",pSwitchButtonFileViewer) ;
  if (pSwitchButtonFileViewer == NULL) {
    err = new patErrNullPointer("Gtk::Switch") ;
    WARNING(err->describe());
    return ;
  }

  if (pDialog) {
    Gtk::Assistant *pAssistant ;
    refBuilder->get_widget("mainAssistant", pAssistant);
    if (pAssistant == NULL) {
      err = new patErrNullPointer("Gtk::Assistant") ;
      WARNING(err->describe());
      return ;
    }
    
    pDialog->set_transient_for(*pAssistant) ;

    stringstream str ; 
    str << patVersion::the()->getVersionInfo() << ": file viewer" ;
    pDialog->set_title(str.str()) ;
  }
  
  if (pOkButton) {
    pOkButton->signal_clicked().connect(sigc::mem_fun(*this,&bioGtkFileViewer::hide)) ;
  }


  

}

void bioGtkFileViewer::hide() {
  if (pDialog) {
    pDialog->hide() ;
    pSwitchButtonFileViewer->set_state(false) ;
  }
}

void bioGtkFileViewer::reset() {
  if (pCriticalFiles) {
    pCriticalFiles->reset() ;
  }
  if (pUsefulFiles) {
    pUsefulFiles->reset() ;
  }
  if (pDebugFiles) {
    pDebugFiles->reset() ;
  }
}


bioGtkFileViewer::~bioGtkFileViewer() {
  hide() ;
}

void bioGtkFileViewer::populateFiles() {
  patIterator<pair<patString,patString> > *theDebugIter =
    patOutputFiles::the()->createDebugIterator() ;  
  for (theDebugIter->first() ;
       !theDebugIter->isDone() ;
       theDebugIter->next()) {
    pair<patString,patString> f = theDebugIter->currentItem() ;
    pDebugFiles->addFile(f) ;
  }
  DELETE_PTR(theDebugIter) ;

  patIterator<pair<patString,patString> > *theUsefulIter =
    patOutputFiles::the()->createUsefulIterator() ;  
  for (theUsefulIter->first() ;
       !theUsefulIter->isDone() ;
       theUsefulIter->next()) {
    pair<patString,patString> f = theUsefulIter->currentItem() ;
    pUsefulFiles->addFile(f) ;
  }
  DELETE_PTR(theUsefulIter) ;

  patIterator<pair<patString,patString> > *theCriticalIter =
    patOutputFiles::the()->createCriticalIterator() ;  
  for (theCriticalIter->first() ;
       !theCriticalIter->isDone() ;
       theCriticalIter->next()) {
    pair<patString,patString> f = theCriticalIter->currentItem() ;
    pCriticalFiles->addFile(f) ;
  }
  DELETE_PTR(theCriticalIter) ;

}

void bioGtkFileViewer::show() {
  if (pDialog) {
    pDialog->show() ;
    pSwitchButtonFileViewer->set_state(true) ;
    
  }
}


void bioGtkFileViewer::hideCriticalFiles() {
  pCriticalFrame->hide() ;
}

void bioGtkFileViewer::hideUsefulFiles() {
  pUsefulFrame->hide() ;
}

void bioGtkFileViewer::hideDebugFiles() {
  pDebugFrame->hide() ;
}
