#include <gtkmm.h>
#include <glibmm/fileutils.h>
#include <glibmm/markup.h>
#include "patError.h"
#include "patDisplay.h"
#include "patVersion.h"
#include <iostream>
#include "bioGtkGui.h"









int main(int argc, char *argv[])
{

  Glib::RefPtr<Gtk::Application> app = Gtk::Application::create(argc,argv,"biogeme.app");
  
  Glib::ustring thePythonPath("") ;

  patError* err(NULL) ;

  bioGtkGui theGui(app,thePythonPath,err) ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    exit(-1) ;
  }

  return theGui.run() ;
  
  
}
