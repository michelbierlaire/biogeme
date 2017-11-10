#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patLap.h"
#include "patDisplay.h"

#ifndef patLapCurrent_var
#define patLapCurrent_var
unsigned long patLap::current=0;
#endif

patLap::patLap(){
}

unsigned long patLap::next()
{
	current++;
	//DEBUG_MESSAGE("============================ " << current << " ===============================");
	return current;
}

unsigned long patLap::get(){
	return current;
}
