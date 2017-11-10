//-*-c++-*------------------------------------------------------------
//
// File name : bioFilePackage.cc
// Author :    Michel Bierlaire
// Date :      Sun Mar 12 16:12:04 2017
//
//--------------------------------------------------------------------

#include <iostream>
#include <sstream>
#include "bioFilePackage.h"

bioFilePackage::bioFilePackage(Gtk::Dialog *pDialog,
			       Gtk::Grid* aGrid) : pGrid(aGrid) {

}

void bioFilePackage::openFile(std::size_t findex) {
  Glib::RefPtr<Gio::File> theFile = 
    Gio::File::create_for_path(listOfFiles[findex]);
  
  Glib::RefPtr<Gio::FileInfo> theFileInfo = theFile->query_info("*") ;

  Glib::ustring uri = theFile->get_uri() ;

  auto refClipboard = Gtk::Clipboard::get() ;
  refClipboard->set_text(uri);
  return ;


  GError *g = NULL;
  GError anerr = { 0, 0, NULL } ; 
  gtk_show_uri_on_window(NULL,uri.c_str(),GDK_CURRENT_TIME,&g) ;
  
  if (g != NULL) {
    std::stringstream str1 ;
    std::stringstream str2 ;
    str1 << "Error: unable to open  " 
	 << uri ;
    Gtk::MessageDialog dialog(*pDialog,Glib::ustring(str1.str())) ;
    anerr = *g;
    if( !anerr.message || !*anerr.message ) {
      str2 << "an unknown error has occurred" ;
      dialog.set_secondary_text(str2.str());
      dialog.run();
    }
  }
  if (anerr.message) {
    std::stringstream str1 ;
    std::stringstream str2 ;
    str1 << "Error: unable to open  " 
	 << uri ;
    Gtk::MessageDialog dialog(*pDialog,Glib::ustring(str1.str())) ;
    str2 << " Reason: " 
	 << anerr.message ;
    dialog.set_secondary_text(str2.str());
    dialog.run();
  }    

}
void bioFilePackage::addFile(std::pair<Glib::ustring,Glib::ustring > f) {
  std::size_t newposition = listOfButtons.size() ;
  listOfFiles.push_back(f.first) ;
  listOfFileNames.push_back(Gtk::Label()) ;
  std::stringstream boldtext ;
  boldtext << "<b>" << f.first << "</b>" ;
  listOfFileNames[newposition].set_markup(boldtext.str()) ;
  listOfLabels.push_back(Gtk::Label(f.second)) ;
  
  listOfButtons.push_back(Gtk::Button("Copy URI")) ;


  Glib::RefPtr<Gio::File> theFile = Gio::File::create_for_path(f.first);
  Glib::ustring uri = theFile->get_uri() ;
  listOfLinkButtons.push_back(Gtk::LinkButton(uri,f.first)) ;

  listOfButtons[newposition].signal_clicked().connect(sigc::bind<std::size_t>(sigc::mem_fun(*this,&bioFilePackage::openFile),newposition)) ;
  std::stringstream str ;
  str << f.first << std::endl << f.second << std::endl << "Click to copy the Uniform Resource Identifier (URI) of the file to be pasted in the browser" ;
  listOfButtons[newposition].set_tooltip_text(str.str()) ;
  listOfLabels[newposition].set_halign(Gtk::ALIGN_START);
  listOfLinkButtons[newposition].set_halign(Gtk::ALIGN_START);
  listOfFileNames[newposition].set_halign(Gtk::ALIGN_START);
  if (pGrid) {
    pGrid->attach(listOfLabels[newposition],2,newposition+1,1,1) ;
    pGrid->attach(listOfFileNames[newposition],1,newposition+1,1,1) ;
    pGrid->attach(listOfButtons[newposition],0,newposition+1,1,1) ;
    //pGrid->attach(listOfLinkButtons[newposition],1,newposition+1,1,1) ;
    pGrid->show_all() ;
  }
}

void bioFilePackage::reset() {
  listOfLabels.erase(listOfLabels.begin(),listOfLabels.end()) ;
  listOfFileNames.erase(listOfFileNames.begin(),listOfFileNames.end()) ;
  listOfButtons.erase(listOfButtons.begin(),listOfButtons.end()) ;
  listOfLinkButtons.erase(listOfLinkButtons.begin(),listOfLinkButtons.end()) ;
  listOfFiles.erase(listOfFiles.begin(),listOfFiles.end()) ;
}
