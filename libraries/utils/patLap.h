#ifndef patLap_h
#define patLap_h

/**************************************
* Temporary static structure to count *
* "laps" while estimating equations   *
**************************************/
//#include <pthread.h>

#define patLapForceCompute (unsigned long) -1

class patLap
{
public:
	static unsigned long next();
	static unsigned long get();
private:
	patLap();
	static unsigned long current;
};

#endif // patLap_h
