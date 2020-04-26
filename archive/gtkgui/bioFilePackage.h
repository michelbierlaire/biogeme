//-*-c++-*------------------------------------------------------------
//
// File name : bioFilePackage.h
// Author :    Michel Bierlaire
// Date :      Sun Mar 12 16:07:48 2017
//
//--------------------------------------------------------------------

#ifndef bioFilePackage_h
#define bioFilePackage_h

// Encapsulate everything related to a list of files. Typically, there
// will be three instances: critical, useful and debug

#include <gtkmm.h>

class bioFilePackage {
 public:
  bioFilePackage(Gtk::Dialog *pDialog, Gtk::Grid* aGrid) ;
  void addFile(std::pair<Glib::ustring,Glib::ustring > f) ;
  void reset() ;
 protected:
  void openFile(std::size_t findex) ;
 private:
  
  Gtk::Dialog *pDialog ;
  Gtk::Grid* pGrid ;
  std::vector<Gtk::Label> listOfLabels ;
  std::vector<Gtk::Label> listOfFileNames ;
  std::vector<Gtk::Button> listOfButtons ;
  std::vector<Gtk::LinkButton> listOfLinkButtons ;
  std::vector<Glib::ustring> listOfFiles ;
  
};
#endif
