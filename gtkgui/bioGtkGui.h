//-*-c++-*------------------------------------------------------------
//
// File name : bioGtkGui.h
// Author :    Michel Bierlaire
// Date :      Wed Mar  8 18:30:26 2017
//
//--------------------------------------------------------------------

#ifndef bioGtkGui_h
#define bioGtkGui_h

#include <gtkmm.h>

#include "patType.h"
#include "patError.h"

#include "bioGtkGui.h"
#include "bioLogWindow.h"
#include "bioGtkFileViewer.h"
//#include "bioGtkScriptDialog.h"

class bioGtkGui {

  enum bioBiogemeType {
    bioPython,
    bioBison,
    bioBiosim,
    bioMod2Py
  } ;
 public: 
  bioGtkGui(Glib::RefPtr<Gtk::Application> app,
	    Glib::ustring pp,
	    patError*& err) ;
  ~bioGtkGui() ;
public:
  int run() ;

protected:
  void cancelButtonClicked() ;
  void applyButtonClicked() ;
  void aboutButtonClicked() ;
  //  void scriptButtonClicked() ;
  void setBiogemeType() ;
  void prepareAssistantPage( Gtk::Widget* theWidget) ;
  void runBiogeme() ;
  patBoolean readyToRun() ;

  void modelFileButtonSelected() ;
  void dataFileButtonSelected() ;
  void getModelFileName() ;
  void getDataFileName() ;

  void resetAll() ;
  void runPythonBiogeme() ;
  void runBisonBiogeme() ;
  void runBiosim() ;
  void runMod2Py() ;
  
  void manageFileViewer() ;
  void manageLogWindow() ;

  void freezeButtons() ;
protected:
  // return the nameof the target file
   Glib::ustring  copyFile(Glib::ustring fileName, Glib::ustring destDir, patError*& err) ;
  void prepareRefBuilder() ;
protected:
  Glib::RefPtr<Gtk::Application> theApplication ;
  Glib::RefPtr<Gtk::Builder> refBuilder ;

  
  
  bioBiogemeType theBiogemeType ;
  
  Gtk::Button *pAboutButton ;
  //  Gtk::Button *pScriptButton ;
  Gtk::AboutDialog *pAboutDialog ;
  Gtk::Assistant* pAssistant ;
  Gtk::Label* pPythonDescription ;
  Gtk::Label* pPythonBiogemeLabel ;
  Gtk::Label* pBisonBiogemeLabel ;
  Gtk::Label* pBiosimLabel ;
  Gtk::Label* pMod2PyLabel ;
  Gtk::Label* pVersionLabel ;
  Gtk::RadioButton* pPythonSelect ;
  Gtk::RadioButton* pBisonSelect ;
  Gtk::RadioButton* pBiosimSelect ;
  Gtk::RadioButton* pMod2pySelect ;
  Gtk::FileChooserButton *pModelFileChooser ;
  Gtk::FileChooserButton *pDataFileChooser ;
  Gtk::Frame *pDataFileFrame ;
  Gtk::Box* pPageFileSelection ;
  Gtk::Switch* pSwitchButtonFileViewer ;
  Gtk::Switch* pSwitchButtonLogWindow ;
  bioLogWindow *theLog ;
  bioGtkFileViewer* theFileViewer ;
  //  bioGtkScriptDialog *theScript ;
  Gtk::Label *pWarningLabel ;
  Gtk::Image *pImage ;
  
  Glib::RefPtr<Gtk::FileFilter>  allFilter ;
  Glib::RefPtr<Gtk::FileFilter>  modFilter ;
  Glib::RefPtr<Gtk::FileFilter>  pyFilter ;
  Glib::RefPtr<Gtk::FileFilter>  datFilter ;

  Glib::ustring modelFileName ;
  Glib::ustring modelFileName_file ;
  Glib::ustring modelFileName_dir ;
  patBoolean modelFileSelected ;
  
  Glib::ustring dataFileName ;
  Glib::ustring dataFileName_file ;
  Glib::ustring dataFileName_dir ;
  patBoolean dataFileSelected ;

  Glib::ustring thePythonPath ;
  patBoolean bisonExecuted ;

  int assistantApplyPage ;

  Glib::RefPtr< Gdk::Pixbuf >  theLogo ;

  patError* err ;
};

#endif
