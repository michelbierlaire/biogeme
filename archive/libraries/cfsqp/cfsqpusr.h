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
patReal objeps=-1.e0;
patReal objrep=-1.e0;
patReal gLgeps=-1.e0;
extern int nstop;

/**************************************************************/
/*     Gradients - Finite Difference                          */
/**************************************************************/

#ifdef __STDC__
void    grobfd(int,int,patReal *,patReal *,void (*)(int,int,
               patReal *,patReal *,void *),void *);
void    grcnfd(int,int,patReal *,patReal *,void (*)(int,int,
               patReal *,patReal *,void *),void *);
#else
void    grobfd();
void    grcnfd();
#endif

/**************************************************************/
/*     Prototype for CFSQP -   	                              */
/**************************************************************/

#ifdef __STDC__
void    cfsqp(int,int,int,int,int,int,int,int,int,int *,int,int,
              int,int *,patReal,patReal,patReal,patReal,patReal *,
              patReal *,patReal *,patReal *,patReal *,patReal *,
              void (*)(int,int,patReal *,patReal *,void *),
              void (*)(int,int,patReal *,patReal *,void *),
              void (*)(int,int,patReal *,patReal *,
                   void (*)(int,int,patReal *,patReal *,void *),void *),
              void (*)(int,int,patReal *,patReal *,
                   void (*)(int,int,patReal *,patReal *,void *),void *),
              void *, patIterationBackup*, patString, int *);
#else
void    cfsqp();
#endif
