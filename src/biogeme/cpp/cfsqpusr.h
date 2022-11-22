/*************************************************************/
/*  CFSQP - Header file to be included in user's main        */
/*          program.                                         */
/*************************************************************/

#include <cstdio>
#include <cmath>
#include <cstdlib>


#define TRUE 1
#define FALSE 0

/* Declare and initialize user-accessible flag indicating    */
/* whether x sent to user functions has been changed within  */
/* CFSQP.				 		     */
int x_is_new=TRUE;

/* Declare and initialize user-accessible stopping criterion */
bioReal objeps=-1.e0;
bioReal objrep=-1.e0;
bioReal gLgeps=-1.e0;
extern int nstop;

/**************************************************************/
/*     Gradients - Finite Difference                          */
/**************************************************************/

#ifdef __STDC__
void    grobfd(int,int,bioReal *,bioReal *,void (*)(int,int,
               bioReal *,bioReal *,void *),void *);
void    grcnfd(int,int,bioReal *,bioReal *,void (*)(int,int,
               bioReal *,bioReal *,void *),void *);
#else
void    grobfd();
void    grcnfd();
#endif

/**************************************************************/
/*     Prototype for CFSQP -   	                              */
/**************************************************************/

#ifdef __STDC__
void    cfsqp(int,int,int,int,int,int,int,int,int,int *,int,int,
              int,int *,bioReal,bioReal,bioReal,bioReal,bioReal *,
              bioReal *,bioReal *,bioReal *,bioReal *,bioReal *,
              void (*)(int,int,bioReal *,bioReal *,void *),
              void (*)(int,int,bioReal *,bioReal *,void *),
              void (*)(int,int,bioReal *,bioReal *,
                   void (*)(int,int,bioReal *,bioReal *,void *),void *),
              void (*)(int,int,bioReal *,bioReal *,
                   void (*)(int,int,bioReal *,bioReal *,void *),void *),
              void *,int*);
#else
void    cfsqp();
#endif
