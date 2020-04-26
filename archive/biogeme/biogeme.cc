#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patBiogemeScripting.h" 

int main(int argc, char *argv[]) {

  patBiogemeScripting theMain ;
  theMain.estimate(argc,argv) ;
  DEBUG_MESSAGE("Finished. Release memory") ;
}

