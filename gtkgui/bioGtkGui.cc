//-*-c++-*------------------------------------------------------------
//
// File name : bioGtkGui.cc
// Author :    Michel Bierlaire
// Date :      Fri Mar 25 18:51:58 2016
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include <gtkmm.h>
#include <glibmm/fileutils.h>
#include <glibmm/markup.h>
#include "patErrMiscError.h"
#include "patDisplay.h"
#include "patVersion.h"
#include "logo.h"
#include <iostream>
#include "patErrNullPointer.h"
#include "patVersion.h"
#include "bioMain.h"

#ifdef BISONBIOGEME
#include "patBiogeme.h"
#include "patBisonSingletonFactory.h"
#endif

#ifdef PYTHONBIOGEME
#include "bioPythonSingletonFactory.h"
#endif

#include "patEnvPathVariable.h"
#include "patSingletonFactory.h"

#include "bioGtkGui.h"
#include "bioLogWindow.h"
#include "bioUiFile.h"

bioGtkGui::bioGtkGui(Glib::RefPtr<Gtk::Application> app,
		     Glib::ustring pp,
		     patError*& e) :
  theApplication(app),
  theBiogemeType(bioPython),
  pAboutButton(NULL),
  //  pScriptButton(NULL),
  pAboutDialog(NULL),
  pAssistant(NULL),
  pPythonDescription(NULL),
  pPythonBiogemeLabel(NULL),
  pBisonBiogemeLabel(NULL),
  pBiosimLabel(NULL),
  pMod2PyLabel(NULL),
  pVersionLabel(NULL),
  pPythonSelect(NULL),
  pBisonSelect(NULL),
  pBiosimSelect(NULL),
  pMod2pySelect(NULL),
  pModelFileChooser(NULL),
  pDataFileChooser(NULL),
  pPageFileSelection(NULL),
  pSwitchButtonFileViewer(NULL),
  pSwitchButtonLogWindow(NULL),
  theLog(NULL),
  theFileViewer(NULL),
  //  theScript(NULL),
  pWarningLabel(NULL),
  pImage(NULL),
  modelFileSelected(patFALSE),
  dataFileSelected(patFALSE),
  bisonExecuted(patFALSE),
  assistantApplyPage(-1),
  theLogo(Gdk::Pixbuf::create_from_inline (262144,my_pixbuf)),
  err(e) {


  prepareRefBuilder() ;

  allFilter = Gtk::FileFilter::create();
  allFilter->set_name("All files") ;
  allFilter->add_pattern("*.*") ;

  
  modFilter = Gtk::FileFilter::create();
  modFilter->set_name("Bison Biogeme files") ;
  modFilter->add_pattern("*.mod") ;

  pyFilter = Gtk::FileFilter::create() ;
  pyFilter->set_name("Python files") ;
  pyFilter->add_pattern("*.py") ;

  datFilter = Gtk::FileFilter::create() ;
  datFilter->set_name("Data files") ;
  datFilter->add_pattern("*.dat") ;
  datFilter->add_pattern("*.csv") ;
  datFilter->add_pattern("*.txt") ;
  datFilter->add_pattern("*.*") ;

  
  refBuilder->get_widget("mainAssistant", pAssistant);
  if (pAssistant == NULL){
    err = new patErrNullPointer("Gtk::Assistant") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("aboutButton", pAboutButton);
  if (pAboutButton == NULL) {
    err = new patErrNullPointer("Gtk::Button") ;
    WARNING(err->describe()) ;
    return ;
  }
  // refBuilder->get_widget("scriptButton", pScriptButton);
  // if (pScriptButton == NULL) {
  //   err = new patErrNullPointer("Gtk::Button") ;
  //   WARNING(err->describe()) ;
  //   return ;
  // }
  refBuilder->get_widget("aboutDialog", pAboutDialog);
  if (pAboutDialog == NULL) {
    err = new patErrNullPointer("Gtk::AboutDialog") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("pythonBisonFileLabel", pPythonDescription);
  if (pPythonDescription == NULL) {
    err = new patErrNullPointer("Gtk::Label") ;
    WARNING(err->describe()) ;
    return ;
  }

  refBuilder->get_widget("versionLabel", pVersionLabel);
  if (pVersionLabel == NULL) {
    err = new patErrNullPointer("Gtk::Label") ;
    WARNING(err->describe()) ;
    return ;
  }

  refBuilder->get_widget("selectPython", pPythonSelect);
  if (pPythonSelect == NULL) {
    err = new patErrNullPointer("Gtk::RadioButton") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("selectBison", pBisonSelect);
  if (pBisonSelect == NULL) {
    err = new patErrNullPointer("Gtk::RadioButton") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("selectBiosim", pBiosimSelect);
  if (pBiosimSelect == NULL) {
    err = new patErrNullPointer("Gtk::RadioButton") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("selectmod2py", pMod2pySelect);
  if (pMod2pySelect == NULL) {
    err = new patErrNullPointer("Gtk::RadioButton") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("modelFileChooser", pModelFileChooser);
  if (pModelFileChooser == NULL) {
    err = new patErrNullPointer("Gtk::FileChooserButton") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("dataFileChooser", pDataFileChooser);
  if (pDataFileChooser == NULL) {
    err = new patErrNullPointer("Gtk::FileChooserButton") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("pageFileSelection", pPageFileSelection);
  if (pPageFileSelection == NULL) {
    err = new patErrNullPointer("Gtk::Box") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("switchButtonFileViewer",pSwitchButtonFileViewer) ;
  if (pSwitchButtonFileViewer == NULL) {
    err = new patErrNullPointer("Gtk::Switch") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("switchButtonLogWindow",pSwitchButtonLogWindow) ;
  if (pSwitchButtonLogWindow == NULL) {
    err = new patErrNullPointer("Gtk::Switch") ;
    WARNING(err->describe()) ;
    return ;
  }
refBuilder->get_widget("applicationMustClose", pWarningLabel);
  if (pWarningLabel == NULL) {
    err = new patErrNullPointer("Gtk::Label") ;
    WARNING(err->describe()) ;
    return ;
  }
  refBuilder->get_widget("welcomeImage", pImage);
  if (pImage == NULL) {
    err = new patErrNullPointer("Gtk::Image") ;
    WARNING(err->describe()) ;
    return ;
  }
  
  
  refBuilder->get_widget("pythonBiogemeLabel", pPythonBiogemeLabel);
  if (pPythonBiogemeLabel == NULL) {
    err = new patErrNullPointer("Gtk::Label") ;
    WARNING(err->describe()) ;
    return ;
  }
  
  refBuilder->get_widget("bisonBiogemeLabel", pBisonBiogemeLabel);
  if (pBisonBiogemeLabel == NULL) {
    err = new patErrNullPointer("Gtk::Label") ;
    WARNING(err->describe()) ;
    return ;
  }

  refBuilder->get_widget("biosimLabel", pBiosimLabel);
  if (pBiosimLabel == NULL) {
    err = new patErrNullPointer("Gtk::Label") ;
    WARNING(err->describe()) ;
    return ;
  }

  refBuilder->get_widget("mod2pyLabel", pMod2PyLabel);
  if (pMod2PyLabel == NULL) {
    err = new patErrNullPointer("Gtk::Label") ;
    WARNING(err->describe()) ;
    return ;
  }

  if (pPythonSelect) {
    pPythonSelect->set_active() ;
  }
  
  if (pAboutButton){ 
    pAboutButton->signal_clicked().connect(sigc::mem_fun(*this,&bioGtkGui::aboutButtonClicked))  ;
  }

  // if (pScriptButton){ 
  //   pScriptButton->signal_clicked().connect(sigc::mem_fun(*this,&bioGtkGui::scriptButtonClicked))  ;
  // }

  if (pPythonSelect) {
    pPythonSelect->signal_toggled().connect(sigc::mem_fun(*this,&bioGtkGui::setBiogemeType)) ;
  }

  if (pBisonSelect) {
    pBisonSelect->signal_toggled().connect(sigc::mem_fun(*this,&bioGtkGui::setBiogemeType)) ;
  }

  if (pBiosimSelect) {
    pBiosimSelect->signal_toggled().connect(sigc::mem_fun(*this,&bioGtkGui::setBiogemeType)) ;
  }

    if (pMod2pySelect) {
    pMod2pySelect->signal_toggled().connect(sigc::mem_fun(*this,&bioGtkGui::setBiogemeType)) ;
  }



  if (pModelFileChooser) {
    pModelFileChooser->signal_file_set().connect(sigc::mem_fun(*this,&bioGtkGui::modelFileButtonSelected)) ;

  }

  if (pDataFileChooser) {
    pDataFileChooser->signal_file_set().connect(sigc::mem_fun(*this,&bioGtkGui::dataFileButtonSelected)) ;
  }

  if (pVersionLabel) {
    pVersionLabel->set_label( bioVersion::the()->getVersionInfoDate()) ;
  }
  if (pSwitchButtonFileViewer) {
pSwitchButtonFileViewer->property_active().signal_changed().connect(sigc::mem_fun(*this,&bioGtkGui::manageFileViewer)) ;
  }
  if (pSwitchButtonLogWindow) {
pSwitchButtonLogWindow->property_active().signal_changed().connect(sigc::mem_fun(*this,&bioGtkGui::manageLogWindow)) ;
  }
    // About dialog
  if (pAboutDialog) {
    pAboutDialog->set_version(patVersion::the()->getVersion()) ;
    pAboutDialog->set_copyright(patVersion::the()->getCopyright()) ;
    pAboutDialog->set_license(patVersion::the()->getLicense()) ;
    std::vector< Glib::ustring > authors ;
    authors.push_back(patVersion::the()->getVersionInfoAuthor()) ;
    pAboutDialog->set_authors(authors) ;
    pAboutDialog->set_comments(patVersion::the()->getVersionInfoCompiledDate()) ;
    pAboutDialog->set_transient_for(*pAssistant) ;
    pAboutDialog->set_logo(theLogo) ;
  }

  if (pImage) {
    pImage->set(theLogo) ;
  }

  if(pAssistant)  {
    stringstream str ; 
    str << patVersion::the()->getVersionInfo() ;
    pAssistant->set_title(str.str()) ;

    pAssistant->signal_cancel().connect(sigc::mem_fun(*this,&bioGtkGui::cancelButtonClicked));    
    pAssistant->signal_apply().connect(sigc::mem_fun(*this,&bioGtkGui::applyButtonClicked));    
    pAssistant->signal_prepare().connect(sigc::mem_fun(*this,&bioGtkGui::prepareAssistantPage));
  }

  theLog = new bioLogWindow(refBuilder) ;
  theFileViewer = new bioGtkFileViewer(refBuilder,err) ;
  if (err != NULL) {
    WARNING(err->describe());
    return ;
  }
  //  theScript = new bioGtkScriptDialog(refBuilder,err) ;
  // if (err != NULL) {
  //   WARNING(err->describe());
  //   return ;
  // }
  
#ifndef PYTHONBIOGEME
  pPythonSelect->hide() ;
  pPythonBiogemeLabel->set_label("Pythonbiogeme [NOT AVAILABLE]") ;
#endif
#ifndef BISONBIOGEME
  pBisonSelect->hide() ;
  pBisonBiogemeLabel->set_label("Bisonbiogeme [NOT AVAILABLE]") ;
  pBiosimSelect->hide() ;
  pBiosimLabel->set_label("Biosim [NOT AVAILABLE]") ;
  pMod2pySelect->hide() ;
  pMod2PyLabel->set_label("mod2py [NOT AVAILABLE]") ;
#endif
}

int bioGtkGui::run() {
    return theApplication->run(*pAssistant);
}

bioGtkGui::~bioGtkGui() {
  DELETE_PTR(pAboutButton) ;
  //  DELETE_PTR(pScriptButton) ;
  DELETE_PTR(pAboutDialog) ;
  DELETE_PTR(pAssistant) ;
  DELETE_PTR(pPythonDescription) ;

}

void bioGtkGui::cancelButtonClicked() {
  if (theApplication) {
    theApplication->quit() ;
  }

}

void bioGtkGui::applyButtonClicked() {
  runBiogeme() ;

}

void bioGtkGui::aboutButtonClicked() {
  if (pAboutDialog) {
    pAboutDialog->run() ;
    pAboutDialog->hide() ;
  }
}

// void bioGtkGui::scriptButtonClicked() {
// #ifdef HAVE_STDLIB_H
//   if (theScript) {
//     theScript->show() ;
//   }
// #else
//   err = new patErrMiscError("This version has been compiled on a system without stdlib.h and cannot execute scripts.") ;
//   Gtk::MessageDialog dialog(*pAssistant,Glib::ustring(err->describe())) ;
//   dialog.run() ;
// #endif
// }




void bioGtkGui::setBiogemeType() {
  if (pPythonSelect) {
    if (pPythonSelect->get_active()) {
      theBiogemeType = bioPython ;
      if (pDataFileChooser) {
	pDataFileChooser->show() ;
      }
    }
  }
  if (pBisonSelect) {
    if (pBisonSelect->get_active()) {
      theBiogemeType = bioBison ;
      if (pDataFileChooser) {
	pDataFileChooser->show() ;
      }

    }
  }
  if (pBiosimSelect) {
    if (pBiosimSelect->get_active()) {
      theBiogemeType = bioBiosim ;
      if (pDataFileChooser) {
	pDataFileChooser->show() ;
      }
    }
  }
  if (pMod2pySelect) {
    if (pMod2pySelect->get_active()) {
      theBiogemeType = bioMod2Py ;
      if (pDataFileChooser) {
	pDataFileChooser->hide() ;
      }
    }
  }
}

void bioGtkGui::prepareAssistantPage( Gtk::Widget* theWidget) {
  if (pPythonDescription) {
    switch (theBiogemeType) {
    case bioPython:
      pPythonDescription->set_label("Python Biogeme") ;
      break ;
    case bioBison:
      pPythonDescription->set_label("Bison Biogeme") ;
      break ;
    case bioBiosim:
      pPythonDescription->set_label("Biosim") ;
      break ;
    case bioMod2Py:
      pPythonDescription->set_label("mod2py") ;
      break ;
    default:
      pPythonDescription->set_label("Biogeme version not properly selected") ;
    }
  }

  if (pDataFileChooser) {
    pDataFileChooser->add_filter(datFilter) ;
  }
  
  if (pModelFileChooser) {
    switch (theBiogemeType) {
    case bioPython:
      pModelFileChooser->set_title("Select a model specification file (extension .py)") ;
      pModelFileChooser->remove_filter(modFilter) ;
      pModelFileChooser->add_filter(pyFilter) ;
      break ;
    case bioBison:
    case bioBiosim:
    case bioMod2Py:
      pModelFileChooser->set_title("Select a model specification file (extension .mod)") ;
      pModelFileChooser->remove_filter(pyFilter) ;
      pModelFileChooser->add_filter(modFilter) ;
      break ;
    }
    
  }
 
}

void bioGtkGui::runBiogeme() {
  if (theLog) {
    theLog->initLog() ;
  }
  if (theFileViewer) {
    theFileViewer->reset() ;
  }
  
  Glib::ustring title ;
  switch (theBiogemeType) {
    case bioPython:
      title = "Running Python biogeme..." ;
      break ;
    case bioBison:
      title = "Running Bison biogeme..." ;
      break ;
  case bioBiosim:
      title = "Running biosim..." ;
      break ;
  case bioMod2Py:
      title = "Running mod2py..." ;
      break ;
  }

  getModelFileName() ;
  getDataFileName() ;
  Glib::ustring dataFileDisplay ;
  if (modelFileName_dir == dataFileName_dir) {
    dataFileDisplay = dataFileName_file ;
  }
  else {
    dataFileDisplay = dataFileName ;
  }
  
  theLog->show(title,
	       modelFileName_dir,
	       modelFileName_file,
	       dataFileDisplay) ;

  if (pSwitchButtonLogWindow) {
    pSwitchButtonLogWindow->set_state(true) ;
  }

  switch (theBiogemeType) {
  case bioPython:
    runPythonBiogeme() ;
    break ;
  case bioBison:
    runBisonBiogeme() ;
    break ;
  case bioBiosim:
    runBiosim() ;
    break ;
  case bioMod2Py:
    runMod2Py() ;
    break ;
  }

  theLog->activateOkButton() ;
  theFileViewer->populateFiles() ;
  theFileViewer->show() ;
  if (pSwitchButtonFileViewer) {
    pSwitchButtonFileViewer->set_state(true) ;
  }
}

void bioGtkGui::getModelFileName() {
  if (pModelFileChooser) {
    modelFileName = pModelFileChooser->get_filename() ;
  }
  modelFileName_file = Glib::path_get_basename(modelFileName) ;
  modelFileName_dir = Glib::path_get_dirname(modelFileName) ;
}

void bioGtkGui::getDataFileName() {
  if (pDataFileChooser) {
    dataFileName = pDataFileChooser->get_filename() ;
  }
  dataFileName_file = Glib::path_get_basename(dataFileName) ;
  dataFileName_dir = Glib::path_get_dirname(dataFileName) ;
}


patBoolean bioGtkGui::readyToRun() {
    switch (theBiogemeType) {
    case bioPython:
    case bioBison:
    case bioBiosim:
      return (modelFileSelected && dataFileSelected && !bisonExecuted) ;
    case bioMod2Py:
      return (modelFileSelected  && !bisonExecuted) ;
    default:
      return patFALSE ;
    }


}

void bioGtkGui::modelFileButtonSelected() {
  modelFileSelected = patTRUE ;
  assistantApplyPage = pAssistant->get_current_page() ;
  if (readyToRun()) {
    if (pAssistant) {
      pAssistant->set_page_complete(*(pAssistant->get_nth_page(assistantApplyPage))) ;
    }
  }
}

void bioGtkGui::dataFileButtonSelected() {
  dataFileSelected = patTRUE ;
  assistantApplyPage = pAssistant->get_current_page() ;
  if (readyToRun()) {
    if (pAssistant) {
      pAssistant->set_page_complete(*(pAssistant->get_nth_page(assistantApplyPage))) ;
    }
  }
}

void bioGtkGui::resetAll() {
  patSingletonFactory::the()->reset() ;
#ifdef PYTHONBIOGEME
  bioPythonSingletonFactory::the()->reset() ;
#endif
}

void bioGtkGui::runPythonBiogeme() {
  cout << "runPythonBiogeme" << endl ;
#ifdef PYTHONBIOGEME
  resetAll() ;
  patDisplay::the().setScreenImportanceLevel(patImportance::patDEBUG) ;

  patEnvPathVariable pythonPath("PYTHONPATH") ;
  pythonPath.readFromSystem() ;
  pythonPath.addPath(".") ;
  pythonPath.addPath(__PYTHONBIOGEME) ;
  pythonPath.addPath(Glib::get_current_dir()) ;
  pythonPath.addPath(thePythonPath) ;
  pythonPath.registerToSystem(err) ;


#ifdef HAVE_CHDIR
  chdir(modelFileName_dir.c_str()) ;
  stringstream ppath ;
#endif

  bioMain theMain ;

  if (modelFileName_dir == dataFileName_dir) {
    
    theMain.run(patString(modelFileName_file),patString(dataFileName_file),err) ;
    if (err != NULL) {
      //      freezeButtons() ;
      Gtk::MessageDialog dialog(*pAssistant,Glib::ustring(err->describe())) ;
      dialog.set_secondary_text("This run of PythonBiogeme cannot be completed") ;
      dialog.run() ;
    }
  }
  else {
    theMain.run(patString(modelFileName_file),patString(dataFileName),err) ;
    if (err != NULL) {
      //      freezeButtons() ;
      Gtk::MessageDialog dialog(*pAssistant,Glib::ustring(err->describe())) ;
      dialog.set_secondary_text("This run of PythonBiogeme cannot be completed");
      dialog.run() ;
    }

  }

  patString htmlFile = theMain.getHtmlFile() ;
  if (err != NULL) {
    //      freezeButtons() ;
      Gtk::MessageDialog dialog(*pAssistant,Glib::ustring(err->describe())) ;
      dialog.set_secondary_text("This run of PythonBiogeme cannot be completed");
      dialog.run() ;
    return ;
  }
#endif
  return ;



}

void bioGtkGui::runBisonBiogeme() {
  cout << "Run bisonbiogeme" << endl ;
#ifdef BISONBIOGEME
  resetAll() ;
  bisonExecuted = patTRUE ;
  patError* err = NULL ;
#ifdef HAVE_CHDIR
  chdir(modelFileName_dir.c_str()) ;
#endif

  patFileNames::the()->setModelName(patString(removeFileExtension(modelFileName_file)));
  patFileNames::the()->emptySamFiles() ;
  if (modelFileName_dir == dataFileName_dir) {
    patFileNames::the()->addSamFile(patString(dataFileName_file)) ;
  }
  else {
    patFileNames::the()->addSamFile(patString(dataFileName)) ;
  }
  patBiogeme biogeme ;

  biogeme.readParameterFile(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  biogeme.loadModelAndSample(err) ;
  if (err != NULL) {
    Gtk::MessageDialog dialog(*pAssistant,Glib::ustring(err->describe())) ;
    dialog.set_secondary_text("This run of BisonBiogeme cannot be completed. Please quit the application") ;
    dialog.run() ;
    freezeButtons() ;
    return ;
  }
  biogeme.estimate(NULL,err) ;
  if (err != NULL) {
      Gtk::MessageDialog dialog(*pAssistant,Glib::ustring(err->describe())) ;
      dialog.set_secondary_text("This run of BisonBiogeme cannot be completed. Please quit the application") ;
      dialog.run() ;
      freezeButtons() ;
      return ;
  }

  freezeButtons() ;
#endif
  return ;

}

void bioGtkGui::runBiosim() {
  cout << "Run biosim"<< endl ;
#ifdef BISONBIOGEME
#ifdef HAVE_CHDIR
  chdir(modelFileName_dir.c_str()) ;
#endif

  patFileNames::the()->setModelName(patString(removeFileExtension(modelFileName_file)));
  patFileNames::the()->emptySamFiles() ;
  if (modelFileName_dir == dataFileName_dir) {
    patFileNames::the()->addSamFile(patString(dataFileName_file)) ;
  }
  else {
    patFileNames::the()->addSamFile(patString(dataFileName)) ;
  }
  patBiogeme biogeme ;

  biogeme.readParameterFile(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }


  biogeme.loadModelAndSample(err) ;
  if (err != NULL) {
      freezeButtons() ;
      Gtk::MessageDialog dialog(*pAssistant,Glib::ustring(err->describe())) ;
      dialog.set_secondary_text("Please quit the application") ;
      dialog.run() ;
    return ;
  }
  biogeme.sampleEnumeration(NULL,0,0,err) ;
  if (err != NULL) {
      freezeButtons() ;
      Gtk::MessageDialog dialog(*pAssistant,Glib::ustring(err->describe())) ;
      dialog.set_secondary_text("Please quit the application") ;
      dialog.run() ;
    return ;
  }

  freezeButtons() ;
#endif
  return ;

}

void bioGtkGui::runMod2Py() {

  cout << "Run mod2py" << endl ;
#ifdef BISONBIOGEME
#ifdef HAVE_CHDIR
  chdir(modelFileName_dir.c_str()) ;
#endif

  patString tmpFileName("__bio__default.dat") ;
  ofstream dataFile(tmpFileName.c_str()) ;
  dataFile << "FakeHeader" << endl ;
  dataFile.close() ;
  patFileNames::the()->setModelName(patString(removeFileExtension(modelFileName_file)));
  patFileNames::the()->emptySamFiles() ;
  patFileNames::the()->addSamFile(tmpFileName) ;
  cout << "Nbr of sample files: " << patFileNames::the()->getNbrSampleFiles() << endl ;
  patBiogeme biogeme ;
  biogeme.readParameterFile(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  // Parameters

  patParameters::the()->setgevGeneratePythonFile(1);
  patParameters::the()->setgevPythonFileWithEstimatedParam (0);

  
  cout << "Load model and sample" << endl ;
  biogeme.loadModelAndSample(err) ;
  if (err != NULL) {
      freezeButtons() ;
      Gtk::MessageDialog dialog(*pAssistant,Glib::ustring(err->describe())) ;
      dialog.set_secondary_text("Please quit the application") ;
      dialog.run() ;
    return ;
  }

  // stringstream str ;
  // str << "The file " << pyFile << " has been created in " << modelFileName_dir ;
  // Gtk::MessageDialog dialog(*this,Glib::ustring(str.str())) ;
  // dialog.set_secondary_text("Make sure to edit it before using it") ;
  // dialog.run() ;

  //Show only the important files
  theFileViewer->hideUsefulFiles() ;
  theFileViewer->hideDebugFiles() ;

  cout << "Done" << endl ;
  freezeButtons() ;
#endif
  return ;
}

void bioGtkGui::manageFileViewer() {
  bool active(false) ;
  if (pSwitchButtonFileViewer) {
    active = pSwitchButtonFileViewer->get_active() ;
    if (active) {
      theFileViewer->show() ;
    }
    else {
      theFileViewer->hide() ;
    }
  }
  return  ;
}

void bioGtkGui::manageLogWindow() {
  bool active(false) ;
  if (pSwitchButtonLogWindow) {
    active = pSwitchButtonLogWindow->get_active() ;
    if (active) {
      theLog->show() ;
    }
    else {
      theLog->hide() ;
    }
  }
  return  ;
}

void bioGtkGui::freezeButtons() {
  if (pAssistant && assistantApplyPage != -1) {

    pAssistant->set_page_complete(*(pAssistant->get_nth_page(assistantApplyPage)),false) ;
  }
  if (pWarningLabel) {
    pWarningLabel->set_label("Bison biogeme and biosim can be run only once. For another run, you have to close the application and start again.") ;
  }
}

Glib::ustring bioGtkGui::copyFile(Glib::ustring fileName, Glib::ustring destDir, patError*& err) {
  Glib::ustring fileName_base ;
  try {
    fileName_base = Glib::path_get_basename(fileName) ;
    Glib::RefPtr<Gio::File> fromFile = 
      Gio::File::create_for_path(fileName); 
    
    Glib::RefPtr<Gio::File> toFile = 
      Gio::File::create_for_path(destDir + fileName_base); 
    
    fromFile->copy(toFile,Gio::FILE_COPY_OVERWRITE); 
    
  } 
  catch(const Glib::Exception& ex) 
    {
      stringstream str ;
      str << "Exception ocurred: " << ex.what() ; 
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return (fileName_base) ;
    }
  return (fileName_base) ;

}

void bioGtkGui::prepareRefBuilder() {


  refBuilder = Gtk::Builder::create();
  patString fileToOpen ;
  bioUiFile uiFile("gtkbiogeme.ui") ;

  if (uiFile.fileFound()) {
    fileToOpen = uiFile.getUiFile(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
  else {
    // Open a dialog to locate gtkguibiogeme.ui
  }

  
  try
  {
    refBuilder->add_from_file(fileToOpen);
  }
  catch(const Glib::FileError& ex)
  {
    stringstream str ;
    str << "FileError: " << ex.what() << endl;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  catch(const Glib::MarkupError& ex)
  {
    stringstream str ;
    str << "MarkupError: " << ex.what() << std::endl;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  catch(const Gtk::BuilderError& ex)
  {
    stringstream str ;
    str << "BuilderError: " << ex.what() << std::endl;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }

}
