//-*-c++-*------------------------------------------------------------
//
// File name : bioGtkFileViewer.h
// Author :    Michel Bierlaire
// Date :      Sun Mar 12 15:34:27 2017
//
//--------------------------------------------------------------------

#ifndef bioGtkFileViewer_h
#define bioGtkFileViewer_h

#include "patOutputFiles.h"
#include <vector>
#include <gtkmm.h>

class bioFilePackage ;

class bioGtkFileViewer {
public:
  bioGtkFileViewer(Glib::RefPtr<Gtk::Builder> b, patError*& err) ;
  //  bioGtkFileViewer(std::vector<std::pair<Glib::ustring,Glib::ustring > > lof) ;
  ~bioGtkFileViewer() ;
  void show() ;
  void hide() ;
  void reset() ;
  void populateFiles() ;
  void hideCriticalFiles() ;
  void hideUsefulFiles() ;
  void hideDebugFiles() ;
protected:
  Glib::RefPtr<Gtk::Builder> refBuilder ;
  Gtk::Dialog *pDialog ;
  Gtk::Grid* pCriticalGrid ;
  Gtk::Grid* pUsefulGrid ;
  Gtk::Grid* pDebugGrid ;
  Gtk::Frame* pCriticalFrame ;
  Gtk::Frame* pUsefulFrame ;
  Gtk::Frame* pDebugFrame ;
  bioFilePackage* pCriticalFiles ;
  bioFilePackage* pUsefulFiles ;
  bioFilePackage* pDebugFiles ;
  Gtk::Button *pOkButton ;
  Gtk::Switch* pSwitchButtonFileViewer ;


  void openFile(std::size_t findex) ;
private:
  void initWindow() ;
};
#endif
