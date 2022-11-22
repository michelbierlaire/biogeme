/**
   Some modifications have been made to this file to be incorporated in the
   Biogeme package. Thos modifications are only related to the management
   of the display, and to premature interruption of the iterations.

   Michel Bierlaire, Tue Jun  3 11:18:29 2003
*/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iomanip>
#include <cmath>
#include "bioCfsqp.h"


/*
  THIS SOFTWARE MAY NOT BE COPIED TO MACHINES OUTSIDE THE SITE FOR
  WHICH IT HAD BEEN PROVIDED.  SEE "Conditions for External Use"
  BELOW FOR MORE DETAILS.  INDIVIDUALS INTERESTED IN OBTAINING
  THE SOFTWARE SHOULD CONTACT AEM DESIGN.
*/

/***************************************************************/
/*     CFSQP - Main Header                                     */
/***************************************************************/

#include <stdio.h>
#include <math.h>
#include <cstdlib>

/***************************************************************/
/*     Macros                                                  */
/***************************************************************/

#define DMAX1(a, b) ((a) > (b) ? (a) : (b))
#define DMIN1(a, b) ((a) < (b) ? (a) : (b))
#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif
#define NONE 0
#define OBJECT 1
#define CONSTR 2

/***************************************************************/
/*     Global Variables and Data Structures                    */
/***************************************************************/

struct _objective {
  bioReal val;
  bioReal *grad;
  bioReal mult;
  bioReal mult_L; /* mode A=1 */
  int act_sip;   /* SIP      */
};

struct _constraint {
  bioReal val;
  bioReal *grad;
  bioReal mult;
  int act_sip;   /* SIP      */
  int d1bind;    /* SR constraints  */
};

struct _parameter {
  bioReal *x;
  bioReal *bl;
  bioReal *bu;
  bioReal *mult;
  void *cd;      /* Client data pointer */
};

struct _violation {    /* SIP      */
  int type;
  int index;
};

/*
  char cfsqp_version[] = "CFSQP 2.5d";
*/
bioReal  bgbnd,tolfea;
int  nstop,maxit;

struct {
  int nnineq, M, ncallg, ncallf, mode, modec;
  int tot_actf_sip,tot_actg_sip,nfsip,ncsipl,ncsipn; /* SIP */
} glob_info;

struct {
  int iprint,info,ipd,iter,initvl,iter_mod;
  FILE *io;
} glob_prnt;

struct {
  bioReal epsmac,rteps,udelta,valnom;
} glob_grd;

struct {
  int dlfeas,local,update,first,rhol_is1,d0_is0,get_ne_mult;
} glob_log;

/* User-accessible stopping criterion (see cfsqpusr.h)          */
extern bioReal objeps;
extern bioReal objrep;
extern bioReal gLgeps;
extern int x_is_new;

/* Workspace                                                     */
int     *iw;
bioReal  *w;
int     lenw, leniw;

/***************************************************************/
/*     Memory Utilities                                        */
/***************************************************************/

#ifdef __STDC__
static int      *make_iv(int);
static bioReal   *make_dv(int);
static bioReal   **make_dm(int, int);
static void     free_iv(int *);
static void     free_dv(bioReal *);
static void     free_dm(bioReal **, int);
static bioReal   *convert(bioReal **, int, int);
#else
static int      *make_iv();
static bioReal   *make_dv();
static bioReal   **make_dm();
static void     free_iv();
static void     free_dv();
static void     free_dm();
static bioReal   *convert();
#endif

/***************************************************************/
/*     Utility Subroutines                                     */
/***************************************************************/

#ifdef __STDC__
int
ql0001_(int *,int *,int *,int *,int *,int *,bioReal *,bioReal *,
        bioReal *,bioReal *,bioReal *,bioReal *,bioReal *,bioReal *,
        int *,int *,int *,bioReal *,int *,int *,int *,bioReal *);
static void     diagnl(int, bioReal, bioReal **);
static void     error(char string[],int *);
static void
estlam(int,int,int *,bioReal,bioReal **,bioReal *,bioReal *,bioReal *,
       struct _constraint *,bioReal *,bioReal *,bioReal *,bioReal *);
static bioReal   *colvec(bioReal **,int,int);
static bioReal   scaprd(int,bioReal *,bioReal *);
static bioReal   small(void);
static int      fuscmp(bioReal,bioReal);
static int      indexs(int,int);
static void     matrcp(int,bioReal **,int,bioReal **);
static void     matrvc(int,int,bioReal **,bioReal *,bioReal *);
static void     nullvc(int,bioReal *);
static void
resign(int,int,bioReal *,bioReal *,bioReal *,struct _constraint *,
       bioReal *,int,int);
static void     sbout1(FILE *,int,char *,bioReal,bioReal *,int,int);
static void     sbout2(FILE *,int,int,char *,char *,bioReal *);
static void     shift(int,int,int *);
static bioReal
slope(int,int,int,int,int,struct _objective *,bioReal *,bioReal *,
      bioReal *,bioReal,bioReal,int,bioReal *,int);
static int      element(int *,int,int);
#else
int     ql0001_();      /* QLD Subroutine */
static void     diagnl();
static void     error();
static void     estlam();
static bioReal   *colvec();
static bioReal   scaprd();
static bioReal   small();
static int      fuscmp();
static int      indexs();
static void     matrcp();
static void     matrvc();
static void     nullvc();
static void     resign();
static void     sbout1();
static void     sbout2();
static void     shift();
static bioReal   slope();
static int      element();
#endif

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
/*     Main routines for optimization -                       */
/**************************************************************/

#ifdef __STDC__
static void
cfsqp1(int,int,int,int,int,int,int,int,int,int,int *,int,
       int,int,int,bioReal,bioReal,int *,int *,struct _parameter *,
       struct _constraint *,struct _objective *,bioReal *,
       void (*)(int,int,bioReal *,bioReal *,void *),
       void (*)(int,int,bioReal *,bioReal *,void *),
       void (*)(int,int,bioReal *,bioReal *,
		void (*)(int,int,bioReal *,bioReal *,void *),void *),
       void (*)(int,int,bioReal *,bioReal *,
		void (*)(int,int,bioReal *,bioReal *,void *),void *),int *);
static void
check(int,int,int,int *,int,int,int,int,int,int,int,int *,bioReal,
      bioReal,struct _parameter *);
static void
initpt(int,int,int,int,int,int,int,struct _parameter *,
       struct _constraint *,void (*)(int,int,bioReal *,bioReal *,void *),
       void (*)(int,int,bioReal *,bioReal *,
                void (*)(int,int,bioReal *,bioReal *,void *),void *));
static void
dir(int,int,int,int,int,int,int,int,int,int,int,int,bioReal *,
    bioReal,bioReal,bioReal *,bioReal *,bioReal,bioReal *,bioReal *,int *,
    int *,int *,int *,int *,int *,struct _parameter *,bioReal *,
    bioReal *,struct _constraint *,struct _objective *,bioReal *,
    bioReal *,bioReal *,bioReal *,bioReal *,bioReal *,bioReal **,bioReal *,
    bioReal *,bioReal *,bioReal *,bioReal **,bioReal **,bioReal *,
    bioReal *,struct _violation *,void (*)(int,int,bioReal *,bioReal *,
					  void *),void (*)(int,int,bioReal *,bioReal *,void *));
static void
step1(int,int,int,int,int,int,int,int,int,int,int,int *,int *,int *,
      int *,int *,int *,int *,int *,int,bioReal,struct _objective *,
      bioReal *,bioReal *,bioReal *,bioReal *,bioReal *,bioReal *,bioReal *,
      bioReal *,bioReal *,bioReal *,bioReal *,struct _constraint *,
      bioReal *,bioReal *,struct _violation *viol,
      void (*)(int,int,bioReal *,bioReal *,void *),
      void (*)(int,int,bioReal *,bioReal *,void *),void *);
static void
hessian(int,int,int,int,int,int,int,int,int,int,int,int *,int,
        bioReal *,struct _parameter *,struct _objective *,
        bioReal,bioReal *,bioReal *,bioReal *,bioReal *,bioReal *,
        struct _constraint *,bioReal *,int *,int *,bioReal *,
        bioReal *,bioReal *,bioReal **,bioReal *,bioReal,int *,
        bioReal *,bioReal *,void (*)(int,int,bioReal *,bioReal *,void *),
        void (*)(int,int,bioReal *,bioReal *,void *),
        void (*)(int,int,bioReal *,bioReal *,
                 void (*)(int,int,bioReal *,bioReal *,void *),void *),
        void (*)(int,int,bioReal *,bioReal *,
                 void (*)(int,int,bioReal *,bioReal *,void *),void *),
        bioReal **,bioReal *,bioReal *,struct _violation *);
static void
out(int,int,int,int,int,int,int,int,int,int,int,int *,bioReal *,
    struct _constraint *,struct _objective *,bioReal,
    bioReal,bioReal,bioReal,bioReal,int);
static void
update_omega(int,int,int,int *,int,int,int,int,bioReal,bioReal,
             struct _constraint *,struct _objective *,bioReal *,
             struct _violation *,void (*)(int,int,bioReal *,bioReal *,
					  void *),void (*)(int,int,bioReal *,bioReal *,void *),
             void (*)(int,int,bioReal *,bioReal *,
		      void (*)(int,int,bioReal *,bioReal *,void *),void *),
             void (*)(int,int,bioReal *,bioReal *,
		      void (*)(int,int,bioReal *,bioReal *,void *),void *),
             void *,int);
#else
static void     cfsqp1();
static void     check();
static void     initpt();
static void     dir();
static void     step1();
static void     hessian();
static void     out();
static void     update_omega();
#endif

#ifdef __STDC__
static void
dealloc(int,int,bioReal *,int *,int *,struct _constraint *cs,
	struct _parameter *);
#else
static void dealloc();
#endif

#ifdef __STDC__
void
cfsqp(int nparam,int nf,int nfsr,int nineqn,int nineq,int neqn,
      int neq,int ncsrl,int ncsrn,int *mesh_pts,
      int mode,int iprint,int miter,int *inform,bioReal bigbnd,
      bioReal eps,bioReal epseqn,bioReal udelta,bioReal *bl,bioReal *bu,
      bioReal *x,bioReal *f,bioReal *g,bioReal *lambda,
      void (*obj)(int, int, bioReal *, bioReal *,void *),
      void (*constr)(int,int,bioReal *,bioReal *,void *),
      void (*gradob)(int,int,bioReal *,bioReal *,
                     void (*)(int,int,bioReal *,bioReal *,void *),void *),
      void (*gradcn)(int,int,bioReal *,bioReal *,
                     void (*)(int,int,bioReal *,bioReal *,void *),void *),
      void *cd,int* nIterPtr)
#else
  void
cfsqp(nparam,nf,nfsr,nineqn,nineq,neqn,neq,ncsrl,ncsrn,mesh_pts,
      mode,iprint,miter,inform,bigbnd,eps,epseqn,udelta,bl,bu,x,
      f,g,lambda,obj,constr,gradob,gradcn,cd,nIterPtr)
  int     nparam,nf,nfsr,neqn,nineqn,nineq,neq,ncsrl,ncsrn,mode,
  iprint,miter,*mesh_pts,*inform;
bioReal  bigbnd,eps,epseqn,udelta;
bioReal  *bl,*bu,*x,*f,*g,*lambda;
void    (* obj)(),(* constr)(),(* gradob)(),(* gradcn)();
void    *cd;
int *nIterPtr ;
#endif

/*---------------------------------------------------------------------
 * Brief specification of various arrays and parameters in the calling
 * sequence. See manual for a more detailed description.
 *
 * nparam : number of variables
 * nf     : number of objective functions (count each set of sequentially
 *          related objective functions once)
 * nfsr   : number of sets of sequentially related objectives (possibly
 *          zero)
 * nineqn : number of nonlinear inequality constraints
 * nineq  : total number of inequality constraints
 * neqn   : number of nonlinear equality constraints
 * neq    : total number of equality constraints
 * ncsrl  : number of sets of linear sequentially related inequality
 *          constraints
 * ncsrn  : number of sets of nonlinear sequentially related inequality
 *          constraints
 * mesh_pts : array of integers giving the number of actual objectives/
 *            constraints in each sequentially related objective or
 *            constraint set. The order is as follows:
 *            (i) objective sets, (ii) nonlinear constraint sets,
 *            (iii) linear constraint sets. If one or no sequentially
 *            related constraint or objectives sets are present, the
 *            user may simply pass the address of an integer variable
 *            containing the appropriate number (possibly zero).
 * mode   : mode=CBA specifies job options as described below:
 *          A = 0 : ordinary minimax problems
 *            = 1 : ordinary minimax problems with each individual
 *                  function replaced by its absolute value, ie,
 *                  an L_infty problem
 *          B = 0 : monotone decrease of objective function
 *                  after each iteration
 *            = 1 : monotone decrease of objective function after
 *                  at most four iterations
 *          C = 1 : default operation.
 *            = 2 : requires that constraints always be evaluated
 *                  before objectives during the line search.
 * iprint : print level indicator with the following options-
 *          iprint=0: no normal output, only error information
 *                    (this option is imposed during phase 1)
 *          iprint=1: a final printout at a local solution
 *          iprint=2: a brief printout at the end of each iteration
 *          iprint=3: detailed infomation is printed out at the end
 *                    of each iteration (for debugging purposes)
 *          For iprint=2 or 3, the information may be printed at
 *          iterations that are multiples of 10, instead of every
 *          iteration. This may be done by adding the desired number
 *          of iterations to skip printing to the desired iprint value
 *          as specified above. e.g., sending iprint=23 would give
 *          the iprint=3 information once every 20 iterations.
 * miter  : maximum number of iterations allowed by the user to solve
 *          the problem
 * inform : status report at the end of execution
 *          inform= 0:normal termination
 *          inform= 1:no feasible point found for linear constraints
 *          inform= 2:no feasible point found for nonlinear constraints
 *          inform= 3:no solution has been found in miter iterations
 *          inform= 4:stepsize smaller than machine precision before
 *                    a successful new iterate is found
 *          inform= 5:failure in attempting to construct d0
 *          inform= 6:failure in attempting to construct d1
 *          inform= 7:inconsistent input data
 *          inform= 8:new iterate essentially identical to previous
 *                    iterate, though stopping criterion not satisfied.
 *          inform= 9:penalty parameter too large, unable to satisfy
 *                    nonlinear equality constraint
 * bigbnd : plus infinity
 * eps    : stopping criterion. Execution stopped when the norm of the
 *          Newton direction vector is smaller than eps
 * epseqn : tolerance of the violation of nonlinear equality constraints
 *          allowed by the user at an optimal solution
 * udelta : perturbation size in computing gradients by finite
 *          difference. The actual perturbation is determined by
 *          sign(x_i) X max{udelta, rteps X max{1, |x_i|}} for each
 *          component of x, where rteps is the square root of machine
 *          precision.
 * bl     : array of dimension nparam,containing lower bound of x
 * bu     : array of dimension nparam,containing upper bound of x
 * x      : array of dimension nparam,containing initial guess in input
 *          and final iterate at the end of execution
 * f      : array of dimension sufficient enough to hold the value of
 *          all regular objective functions and the value of all
 *          members of the sequentially related objective sets.
 *          (dimension must be at least 1)
 * g      : array of dimension sufficient enough to hold the value of
 *          all regular constraint functions and the value of all
 *          members of the sequentially related constraint sets.
 *          (dimension must be at least 1)
 * lambda : array of dimension nparam+dim(f)+dim(g), containing
 *          Lagrange multiplier values at x in output. (A concerns the
 *          mode, see above). The first nparam positions contain the
 *          multipliers associated with the simple bounds, the next
 *          dim(g) positions contain the multipliers associated with
 *          the constraints. The final dim(f) positions contain the
 *          multipliers associated with the objective functions. The
 *          multipliers are in the order they were specified in the
 *          user-defined objective and constraint functions.
 * obj    : Pointer to function that returns the value of objective
 *          functions, one upon each call
 * constr : Pointer to function that returns the value of constraints
 *          one upon each call
 * gradob : Pointer to function that computes gradients of f,
 *          alternatively it can be replaced by grobfd to compute
 *          finite difference approximations
 * gradcn : Pointer to function that computes gradients of g,
 *          alternatively it can be replaced by grcnfd to compute
 *          finite difference approximations
 * cd     : Void pointer that may be used by the user for the passing of
 *          "client data" (untouched by CFSQP)
 *
 *----------------------------------------------------------------------
 *
 *
 *                       CFSQP  Version 2.5d
 *
 *                  Craig Lawrence, Jian L. Zhou
 *                         and Andre Tits
 *                  Institute for Systems Research
 *                               and
 *                Electrical Engineering Department
 *                     University of Maryland
 *                     College Park, Md 20742
 *
 *                         February, 1998
 *
 *
 *  The purpose of CFSQP is to solve general nonlinear constrained
 *  minimax optimization problems of the form
 *
 *   (A=0 in mode)     minimize    max_i f_i(x)   for i=1,...,n_f
 *                        or
 *   (A=1 in mode)     minimize    max_j |f_i(x)|   for i=1,...,n_f
 *                       s.t.      bl   <= x <=  bu
 *                                 g_j(x) <= 0,   for j=1,...,nineqn
 *                                 A_1 x - B_1 <= 0
 *
 *                                 h_i(x)  = 0,   for i=1,...,neqn
 *                                 A_2 x - B_2  = 0
 *
 * CFSQP is also able to efficiently handle problems with large sets of
 * sequentially related objectives or constraints, see the manual for
 * details.
 *
 *
 *                  Conditions for External Use
 *                  ===========================
 *
 *   1. The CFSQP routines may not be distributed to third parties.
 *      Interested parties shall contact AEM Design directly.
 *   2. If modifications are performed on the routines, these
 *      modifications shall be communicated to AEM Design.  The
 *      modified routines will remain the sole property of the authors.
 *   3. Due acknowledgment shall be made of the use of the CFSQP
 *      routines in research reports or publications. Whenever
 *      such reports are released for public access, a copy shall
 *      be forwarded to AEM Design.
 *   4. The CFSQP routines may only be used for research and
 *      development, unless it has been agreed otherwise with AEM
 *      Design in writing.
 *
 * Copyright (c) 1993-1998 by Craig T. Lawrence, Jian L. Zhou, and
 *                         Andre L. Tits
 * All Rights Reserved.
 *
 *
 * Enquiries should be directed to:
 *
 *      AEM Design
 *      2691 Smoketree Way
 *      Atlanta, GA 30345
 *      U. S. A.
 *
 *      Phone : 770-934-0174
 *      Fax   : 770-939-8365
 *      E-mail: info@aemdesign.com
 *
 *  References:
 *  [1] E. Panier and A. Tits, `On Combining Feasibility, Descent and
 *      Superlinear Convergence In Inequality Constrained Optimization',
 *      Mathematical Programming, Vol. 59(1993), 261-276.
 *  [2] J. F. Bonnans, E. Panier, A. Tits and J. Zhou, `Avoiding the
 *      Maratos Effect by Means of a Nonmonotone Line search: II.
 *      Inequality Problems - Feasible Iterates', SIAM Journal on
 *      Numerical Analysis, Vol. 29, No. 4, 1992, pp. 1187-1202.
 *  [3] J.L. Zhou and A. Tits, `Nonmonotone Line Search for Minimax
 *      Problems', Journal of Optimization Theory and Applications,
 *      Vol. 76, No. 3, 1993, pp. 455-476.
 *  [4] C.T. Lawrence, J.L. Zhou and A. Tits, `User's Guide for CFSQP
 *      Version 2.5: A C Code for Solving (Large Scale) Constrained
 *      Nonlinear (Minimax) Optimization Problems, Generating Iterates
 *      Satisfying All Inequality Constraints,' Institute for
 *      Systems Research, University of Maryland,Technical Report
 *      TR-94-16r1, College Park, MD 20742, 1997.
 *  [5] C.T. Lawrence and A.L. Tits, `Nonlinear Equality Constraints
 *      in Feasible Sequential Quadratic Programming,' Optimization
 *      Methods and Software, Vol. 6, March, 1996, pp. 265-282.
 *  [6] J.L. Zhou and A.L. Tits, `An SQP Algorithm for Finely
 *      Discretized Continuous Minimax Problems and Other Minimax
 *      Problems With Many Objective Functions,' SIAM Journal on
 *      Optimization, Vol. 6, No. 2, May, 1996, pp. 461--487.
 *  [7] C. T. Lawrence and A. L. Tits, `Feasible Sequential Quadratic
 *      Programming for Finely Discretized Problems from SIP,'
 *      To appear in R. Reemtsen, J.-J. Ruckmann (eds.): Semi-Infinite
 *      Programming, in the series Nonconcex Optimization and its
 *      Applications. Kluwer Academic Publishers, 1998.
 *
 ***********************************************************************
 */
{
  int  i,ipp,j,ncnstr,nclin,nctotl,nob,nobL,modem,nn,
    nppram,nrowa,ncsipl1,ncsipn1,nfsip1;
  int  feasbl,feasb,prnt,Linfty;
  int  *indxob,*indxcn,*mesh_pts1;
  bioReal *signeq;
  bioReal xi,gi,gmax,dummy,epskt;
  struct _constraint *cs;      /* pointer to array of constraints */
  struct _objective  *ob;      /* pointer to array of objectives  */
  struct _parameter  *param;   /* pointer to parameter structure  */
  struct _parameter  _param;

  /*     Make adjustments to parameters for SIP constraints       */
  glob_info.tot_actf_sip = glob_info.tot_actg_sip = 0;
  mesh_pts=mesh_pts-1;
  glob_info.nfsip=nfsr;
  glob_info.ncsipl=ncsrl;
  glob_info.ncsipn=ncsrn;
  nf=nf-nfsr;
  nfsip1=nfsr;
  nfsr=0;
  for (i=1; i<=nfsip1; i++)
    nfsr=nfsr+mesh_pts[i];
  nf=nf+nfsr;
  nineqn=nineqn-ncsrn;
  nineq=nineq-ncsrl-ncsrn;
  ncsipl1=ncsrl;
  ncsipn1=ncsrn;
  ncsrl=0;
  ncsrn=0;
  if (ncsipn1)
    for (i=1; i<=ncsipn1; i++)
      ncsrn=ncsrn+mesh_pts[nfsip1+i];
  if (ncsipl1)
    for (i=1; i<=ncsipl1; i++)
      ncsrl=ncsrl+mesh_pts[nfsip1+ncsipn1+i];
  nineqn=nineqn+ncsrn;
  nineq=nineq+ncsrn+ncsrl;
  /* Create array of constraint structures               */
  cs=(struct _constraint *)calloc(nineq+neq+1,
				  sizeof(struct _constraint));
  for (i=1; i<=nineq+neq; i++) {
    cs[i].grad=make_dv(nparam);
    cs[i].act_sip=FALSE;
    cs[i].d1bind=FALSE;
  }
  /* Create parameter structure                          */
  _param.x=make_dv(nparam+1);
  _param.bl=make_dv(nparam);
  _param.bu=make_dv(nparam);
  _param.mult=make_dv(nparam+1);
  param=&_param;

  /*   Initialize, compute the machine precision, etc.   */
  bl=bl-1; bu=bu-1; x=x-1;
  for (i=1; i<=nparam; i++) {
    param->x[i]=x[i];
    param->bl[i]=bl[i];
    param->bu[i]=bu[i];
  }
  param->cd=cd;    /* Initialize client data */
  dummy=0.e0;
  f=f-1; g=g-1; lambda=lambda-1;
  glob_prnt.iter=0;
  nstop=1;
  nn=nineqn+neqn;
  glob_grd.epsmac=small();
  tolfea=glob_grd.epsmac*1.e2;
  bgbnd=bigbnd;
  glob_grd.rteps=sqrt(glob_grd.epsmac);
  glob_grd.udelta=udelta;
  glob_log.rhol_is1=FALSE;
  glob_log.get_ne_mult=FALSE;
  signeq=make_dv(neqn);

  nob=0;
  gmax=-bgbnd;
  glob_prnt.info=0;
  glob_prnt.iprint=iprint%10;
  ipp=iprint;
  glob_prnt.iter_mod=DMAX1(iprint-iprint%10,1);
  glob_prnt.io=stdout;
  ncnstr=nineq+neq;
  glob_info.nnineq=nineq;
  if (glob_prnt.iprint>0) {
    GENERAL_MESSAGE("\n\n  CFSQP Version 2.5d (Released February 1998)");
    GENERAL_MESSAGE("          Copyright (c) 1993 --- 1998");
    GENERAL_MESSAGE("           C.T. Lawrence, J.L. Zhou");
    GENERAL_MESSAGE("                and A.L. Tits");
    GENERAL_MESSAGE("             All Rights Reserved\n");
  }
  /*-----------------------------------------------------*/
  /*   Check the input data                              */
  /*-----------------------------------------------------*/
  check(nparam,nf,nfsr,&Linfty,nineq,nineqn,neq,neqn,
	ncsrl,ncsrn,mode,&modem,eps,bgbnd,param);
  if (glob_prnt.info==7) {
    *inform=glob_prnt.info;
    return;
  }

  maxit=DMAX1(DMAX1(miter,10*DMAX1(nparam,ncnstr)),1000);
  feasb=TRUE;
  feasbl=TRUE;
  prnt=FALSE;
  nppram=nparam+1;

  /*-----------------------------------------------------*/
  /*   Check whether x is within bounds                  */
  /*-----------------------------------------------------*/
  for (i=1; i<=nparam; i++) {
    xi=param->x[i];
    if (param->bl[i]<=xi && param->bu[i]>=xi) continue;
    feasbl=FALSE;
    break;
  }
  nclin=ncnstr-nn;
  /*-----------------------------------------------------*/
  /*   Check whether linear constraints are feasbile     */
  /*-----------------------------------------------------*/
  if (nclin!=0) {
    for (i=1; i<=nclin; i++) {
      j=i+nineqn;
      if (j<=nineq) {
	constr(nparam,j,(param->x)+1,&gi,param->cd);
	if (gi>glob_grd.epsmac) feasbl=FALSE;
      } else {
	constr(nparam,j+neqn,(param->x)+1,&gi,param->cd);
	if (fabs(gi)>glob_grd.epsmac) feasbl=FALSE;
      }
      cs[j].val=gi;
    }
  }
  /*-------------------------------------------------------*/
  /*   Generate a new point if infeasible                  */
  /*-------------------------------------------------------*/
  if (!feasbl) {
    if (glob_prnt.iprint>0) {
      DETAILED_MESSAGE(" The given initial point is infeasible for inequality")
      DETAILED_MESSAGE(" constraints and linear equality constraints:")
      sbout1(glob_prnt.io,nparam,"                    ",dummy,
	     param->x,2,1);
      prnt=TRUE;
    }
    nctotl=nparam+nclin;
    lenw=2*nparam*nparam+10*nparam+2*nctotl+1;
    leniw=DMAX1(2*nparam+2*nctotl+3,2*nclin+2*nparam+6);
    /*-----------------------------------------------------*/
    /*   Attempt to generate a point satisfying all linear */
    /*   constraints.                                      */
    /*-----------------------------------------------------*/
    nrowa=DMAX1(nclin,1);
    iw=make_iv(leniw);
    w=make_dv(lenw);
    initpt(nparam,nineqn,neq,neqn,nclin,nctotl,nrowa,param,
	   &cs[nineqn],constr,gradcn);
    free_iv(iw);
    free_dv(w);
    if (glob_prnt.info!=0) {
      *inform=glob_prnt.info;
      return;
    }
  }
  indxob=make_iv(DMAX1(nineq+neq,nf));
  indxcn=make_iv(nineq+neq);
 L510:
  if (glob_prnt.info!=-1) {
    for (i=1; i<=nineqn; i++) {
      constr(nparam,i,(param->x)+1,&(cs[i].val),param->cd);
      if (cs[i].val>0.e0) feasb=FALSE;
    }
    glob_info.ncallg=nineqn;
    if (!feasb) {
      /* Create array of objective structures for Phase 1  */
      ob=(struct _objective *)calloc(nineqn+1,
				     sizeof(struct _objective));
      for (i=1; i<=nineqn; i++) {
	ob[i].grad=make_dv(nparam);
	ob[i].act_sip=FALSE;
      }
      for (i=1; i<=nineqn; i++) {
	nob++;
	indxob[nob]=i;
	ob[nob].val=cs[i].val;
	gmax=DMAX1(gmax,ob[nob].val);
      }
      for (i=1; i<=nineq-nineqn; i++)
	indxcn[i]=nineqn+i;
      for (i=1; i<=neq-neqn; i++)
	indxcn[i+nineq-nineqn]=nineq+neqn+i;
      goto L605;
    }
  }

  /* Create array of objective structures for Phase 2 and      */
  /* initialize.                                               */
  ob=(struct _objective *)calloc(nf+1,sizeof(struct _objective));
  for (i=1; i<=nf; i++) {
    ob[i].grad=make_dv(nparam);
    ob[i].act_sip=FALSE;
  }
  for (i=1; i<=nineqn; i++) {
    indxcn[i]=i;
  }
  for (i=1; i<=neq-neqn; i++)
    cs[i+nineq+neqn].val=cs[i+nineq].val;
  for (i=1; i<=neqn; i++) {
    j=i+nineq;
    constr(nparam,j,(param->x)+1,&(cs[j].val),param->cd);
    indxcn[nineqn+i]=j;
  }
  for (i=1; i<=nineq-nineqn; i++)
    indxcn[i+nn]=nineqn+i;
  for (i=1; i<=neq-neqn; i++)
    indxcn[i+nineq+neqn]=nineq+neqn+i;
  glob_info.ncallg+=neqn;

 L605:
  if (glob_prnt.iprint>0 && feasb && !prnt) {
    DETAILED_MESSAGE("The given initial point is feasible for inequality")
    DETAILED_MESSAGE("         constraints and linear equality constraints:")
    sbout1(glob_prnt.io,nparam,"                    ",dummy,
	   param->x,2,1);
    prnt=TRUE;
  }
  if (nob==0) {
    if (glob_prnt.iprint>0) {
      if (glob_prnt.info!=0) {
	DETAILED_MESSAGE("To generate a feasible point for nonlinear inequality")
	DETAILED_MESSAGE("constraints and linear equality constraints, ")
	DETAILED_MESSAGE("ncallg = " << std::setw(10) << glob_info.ncallg)
	if (ipp==0)
	  DETAILED_MESSAGE(" iteration           " << std::setw(26) << glob_prnt.iter)
	if (ipp>0)
	  DETAILED_MESSAGE(" iteration           " << std::setw(26) << glob_prnt.iter-1)
	if (ipp==0) glob_prnt.iter++;
      }
      if (feasb && !feasbl) {
	DETAILED_MESSAGE("Starting from the generated point feasible for")
	DETAILED_MESSAGE("inequality constraints and linear equality constraints:")
	sbout1(glob_prnt.io,nparam,"                    ",
	       dummy,param->x,2,1);
	   
      }
      if (glob_prnt.info!=0 || !prnt || !feasb) {
	DETAILED_MESSAGE("Starting from the generated point feasible for")
	DETAILED_MESSAGE("inequality constraints and linear equality constraints:")
	sbout1(glob_prnt.io,nparam,"                    ",
	       dummy,param->x,2,1);
      }
    }
    feasb=TRUE;
    feasbl=TRUE;
  }
  if (ipp>0 && !feasb && !prnt) {
    DETAILED_MESSAGE(" The given initial point is infeasible for inequality")
    DETAILED_MESSAGE(" constraints and linear equality constraints:")
    sbout1(glob_prnt.io,nparam,"                    ",dummy,
	   param->x,2,1);
    prnt=TRUE;
  }
  if (nob==0) nob=1;
  if (feasb) {
    nob=nf;
    glob_prnt.info=0;
    glob_prnt.iprint=iprint%10;
    ipp=iprint;
    glob_prnt.iter_mod=DMAX1(iprint-iprint%10,1);
    glob_info.mode=modem;
    epskt=eps;
    if (Linfty) nobL=2*nob;
    else nobL=nob;
    if (nob!=0 || neqn!=0) goto L910;
    DETAILED_MESSAGE("current feasible iterate with no objective specified")
    *inform=glob_prnt.info;
    for (i=1; i<=nineq+neq; i++)
      g[i]=cs[i].val;
    dealloc(nineq,neq,signeq,indxcn,indxob,cs,param);
    free((char *) ob);
    return;
  }
  ipp=0;
  glob_info.mode=0;
  nobL=nob;
  glob_prnt.info=-1;
  epskt=1.e-10;
 L910:
  nctotl=nppram+ncnstr+DMAX1(nobL,1);
  leniw=2*(ncnstr+DMAX1(nobL,1))+2*nppram+6;
  lenw=2*nppram*nppram+10*nppram+6*(ncnstr+DMAX1(nobL,1)+1);
  glob_info.M=4;
  if (modem==1 && nn==0) glob_info.M=3;

  param->x[nparam+1]=gmax;
  if (feasb) {
    for (i=1; i<=neqn; i++) {
      if (cs[i+nineq].val>0.e0) signeq[i]=-1.e0;
      else signeq[i]=1.e0;
    }
  }
  if (!feasb) {
    ncsipl1=ncsrl;
    ncsipn1=0;
    nfsip1=ncsrn;
    mesh_pts1=&mesh_pts[glob_info.nfsip];
  } else {
    ncsipl1=ncsrl;
    ncsipn1=ncsrn;
    nfsip1=nfsr;
    mesh_pts1=mesh_pts;
  }
  /*---------------------------------------------------------------*/
  /*    either attempt to generate a point satisfying all          */
  /*    constraints or try to solve the original problem           */
  /*---------------------------------------------------------------*/
  nrowa=DMAX1(ncnstr+DMAX1(nobL,1),1);
  w=make_dv(lenw);
  iw=make_iv(leniw);

  cfsqp1(miter,nparam,nob,nobL,nfsip1,nineqn,neq,neqn,ncsipl1,ncsipn1,
	 mesh_pts1,ncnstr,nctotl,nrowa,feasb,epskt,epseqn,indxob,
	 indxcn,param,cs,ob,signeq,obj,constr,gradob,gradcn,nIterPtr);

  free_iv(iw);
  free_dv(w);
  if (glob_prnt.info==-1) { /* Successful phase 1 termination  */
    for (i=1; i<=nob; i++)
      cs[i].val=ob[i].val;
    nob=0;
    for (i=1; i<=nineqn; i++)
      free_dv(ob[i].grad);
    free((char *) ob);
    goto L510;
  }
  if (glob_prnt.info!=0) {
    if (feasb) {
      for (i=1; i<=nparam; i++)
	x[i]=param->x[i];
      for (i=1; i<=nineq+neq; i++)
	g[i]=cs[i].val;
      *inform=glob_prnt.info;
      dealloc(nineq,neq,signeq,indxcn,indxob,cs,param);
      for (i=1; i<=nf; i++) {
	f[i]=ob[i].val;
	free_dv(ob[i].grad);
      }
      free((char *) ob);
      return;
    }
    glob_prnt.info=2;
    DETAILED_MESSAGE("Error: No feasible point is found for nonlinear inequality")
    DETAILED_MESSAGE("constraints and linear equality constraints")
    *inform=glob_prnt.info;
    dealloc(nineq,neq,signeq,indxcn,indxob,cs,param);
    for (i=1; i<=nineqn; i++)
      free_dv(ob[i].grad);
    free((char *) ob);
    return;
  }
  /* Successful phase 2 termination                            */
  *inform=glob_prnt.info;
  for (i=1; i<=nparam; i++) {
    x[i]=param->x[i];
    lambda[i]=param->mult[i];
  }
  for (i=1; i<=nineq+neq; i++) {
    g[i]=cs[i].val;
    lambda[i+nparam]=cs[i].mult;
  }
  for (i=1; i<=nf; i++) {
    f[i]=ob[i].val;
    lambda[i+nparam+nineq+neq]=ob[i].mult;
    free_dv(ob[i].grad);
  }
  /* If just one objective, set multiplier=1 */
  if (nf==1) lambda[1+nparam+nineq+neq]=1.e0;
  free((char *) ob);
  dealloc(nineq,neq,signeq,indxcn,indxob,cs,param);
  return;
}

/***************************************************************/
/*     Free allocated memory                                   */
/***************************************************************/

#ifdef __STDC__
static void
dealloc(int nineq,int neq,bioReal *signeq,int *indxob,
        int *indxcn,struct _constraint *cs,struct _parameter *param)
#else
  static void
dealloc(nineq,neq,signeq,indxob,indxcn,cs,param)
  int nineq,neq;
bioReal *signeq;
int    *indxob,*indxcn;
struct _constraint *cs;
struct _parameter  *param;
#endif
{
  int i;

  free_dv(param->x);
  free_dv(param->bl);
  free_dv(param->bu);
  free_dv(param->mult);
  free_dv(signeq);
  free_iv(indxob);
  free_iv(indxcn);
  for (i=1; i<=nineq+neq; i++)
    free_dv(cs[i].grad);
  free((char *) cs);
}

/************************************************************/
/*   CFSQP : Main routine                                   */
/************************************************************/

#ifdef __STDC__
static void
dealloc1(int,int,bioReal **,bioReal **,bioReal **,bioReal *,bioReal *,
         bioReal *,bioReal *,bioReal *,bioReal *,bioReal *,bioReal *,
         bioReal *,bioReal *,bioReal *,bioReal *,int *,int *,int *);
#else
static void dealloc1();
#endif

#ifdef __STDC__
static void
cfsqp1(int miter,int nparam,int nob,int nobL,int nfsip,int nineqn,
       int neq,int neqn,int ncsipl,int ncsipn,int *mesh_pts,int ncnstr,
       int nctotl,int nrowa,int feasb,bioReal epskt,bioReal epseqn,
       int *indxob,int *indxcn,struct _parameter *param,
       struct _constraint *cs, struct _objective *ob,
       bioReal *signeq,void (*obj)(int,int,bioReal *,bioReal *,void *),
       void (*constr)(int,int,bioReal *,bioReal *,void *),
       void (*gradob)(int,int,bioReal *,bioReal *,
		      void (*)(int,int,bioReal *,bioReal *,void *),void *),
       void (*gradcn)(int,int,bioReal *,bioReal *,
		      void (*)(int,int,bioReal *,bioReal *,void *),void *),int* nIterPtr)
#else
  static void
cfsqp1(miter,nparam,nob,nobL,nfsip,nineqn,neq,neqn,ncsipl,ncsipn,
       mesh_pts,ncnstr,nctotl,nrowa,feasb,epskt,epseqn,indxob,
       indxcn,param,cs,ob,signeq,obj,constr,gradob,gradcn,nIterPtr)
  int     miter,nparam,nob,nobL,nfsip,nineqn,neq,neqn,ncnstr,
  nctotl,nrowa,feasb,ncsipl,ncsipn,*mesh_pts;
int     *indxob,*indxcn;
bioReal  epskt,epseqn;
bioReal  *signeq;
struct _constraint *cs;
struct _objective  *ob;
struct _parameter  *param;
int *nIterPtr ;
void   (* obj)(),(* constr)(),(* gradob)(),(* gradcn)();
#endif
{
  int   i,iskp,nfs,ncf,ncg,nn,nstart,nrst,ncnst1;
  int   *iact,*iskip,*istore;
  bioReal Cbar,Ck,dbar,fmax,fM,fMp,steps,d0nm,dummy,
    sktnom,scvneq,grdftd,psf;
  bioReal *di,*d,*gm,*grdpsf,*penp,*bl,*bu,*clamda,
    *cvec,*psmu,*span,*backup;
  bioReal **hess,**hess1,**a;
  bioReal *tempv;
  struct _violation *viol;
  struct _violation _viol;

  /*   Allocate memory                              */

  hess=make_dm(nparam,nparam);
  hess1=make_dm(nparam+1,nparam+1);
  a=make_dm(nrowa,nparam+2);
  di=make_dv(nparam+1);
  d=make_dv(nparam+1);
  gm=make_dv(4*neqn);
  grdpsf=make_dv(nparam);
  penp=make_dv(neqn);
  bl=make_dv(nctotl);
  bu=make_dv(nctotl);
  clamda=make_dv(nctotl+nparam+1);
  cvec=make_dv(nparam+1);
  psmu=make_dv(neqn);
  span=make_dv(4);
  backup=make_dv(nob+ncnstr);
  iact=make_iv(nob+nineqn+neqn);
  iskip=make_iv(glob_info.nnineq+1);
  istore=make_iv(nineqn+nob);

  viol=&_viol;
  viol->index=0;
  viol->type=NONE;

  glob_prnt.initvl=1;
  glob_log.first=TRUE;
  nrst=glob_prnt.ipd=0;
  dummy=0.e0;
  scvneq=0.e0;
  steps=0.e0;
  sktnom=0.e0;
  d0nm=0.e0;
  if (glob_prnt.iter==0) diagnl(nparam,1.e0,hess);
  if (feasb) {
    glob_log.first=TRUE;
    if (glob_prnt.iter>0) glob_prnt.iter--;
    if (glob_prnt.iter!=0) diagnl(nparam,1.e0,hess);
  }
  Ck=Cbar=1.e-2;
  dbar=5.e0;
  nstart=1;
  glob_info.ncallf=0;
  nstop=1;
  nfs=0;
  if (glob_info.mode!=0)
    nfs=glob_info.M;
  if (feasb) {
    nn=nineqn+neqn;
    ncnst1=ncnstr;
  } else {
    nn=0;
    ncnst1=ncnstr-nineqn-neqn;
  }
  scvneq=0.e0;
  for (i=1; i<=ncnst1; i++) {
    glob_grd.valnom=cs[indxcn[i]].val;
    backup[i]=glob_grd.valnom;
    if (feasb && i>nineqn && i<=nn) {
      gm[i-nineqn]=glob_grd.valnom*signeq[i-nineqn];
      scvneq=scvneq+fabs(glob_grd.valnom);
    }
    if (feasb && i<=nn) {
      iact[i]=indxcn[i];
      if (i<=nineqn) istore[i]=0;
      if (i>nineqn) penp[i-nineqn]=2.e0;
    }
    gradcn(nparam,indxcn[i],(param->x)+1,(cs[indxcn[i]].grad)+1,
	   constr,param->cd);
  }
  nullvc(nparam,grdpsf);
  psf=0.e0;
  if (feasb && neqn!=0)
    resign(nparam,neqn,&psf,grdpsf,penp,cs,signeq,12,12);
  fmax=-bgbnd;
  for (i=1; i<=nob; i++) {
    if (!feasb) {
      glob_grd.valnom=ob[i].val;
      iact[i]=i;
      istore[i]=0;
      gradcn(nparam,indxob[i],(param->x)+1,(ob[i].grad)+1,constr,
	     param->cd);
    } else {
      iact[nn+i]=i;
      istore[nineqn+i]=0;
      obj(nparam,i,(param->x)+1,&(ob[i].val),param->cd);
      glob_grd.valnom=ob[i].val;
      backup[i+ncnst1]=glob_grd.valnom;
      gradob(nparam,i,(param->x)+1,(ob[i].grad)+1,obj,param->cd);
      glob_info.ncallf++;
      if (nobL!=nob) fmax=DMAX1(fmax,-ob[i].val);
    }
    fmax=DMAX1(fmax,ob[i].val);
  }
  if (feasb && nob==0) fmax=0.e0;
  fM=fmax;
  fMp=fmax-psf;
  span[1]=fM;

  if (glob_prnt.iprint>=3 && glob_log.first) {
    for (i=1; i<=nob; i++) {
      if (feasb) {
	if (nob>1) {
	  tempv=ob[i].grad;
	  sbout2(glob_prnt.io,nparam,i,"gradf(j,",")",tempv);
	}
	if (nob==1) {
	  tempv=ob[1].grad;
	  sbout1(glob_prnt.io,nparam,"gradf(j)            ",
		 dummy,tempv,2,2);
	}
	continue;
      }
      tempv=ob[i].grad;
      sbout2(glob_prnt.io,nparam,indxob[i],"gradg(j,",")",tempv);
    }
    if (ncnstr!=0) {
      for (i=1; i<=ncnst1; i++) {
	tempv=cs[indxcn[i]].grad;
	sbout2(glob_prnt.io,nparam,indxcn[i],"gradg(j,",")",tempv);
      }
      if (neqn!=0) {
	sbout1(glob_prnt.io,nparam,"grdpsf(j)           ",dummy,
	       grdpsf,2,2);
	sbout1(glob_prnt.io,neqn,"P                   ",dummy,
	       penp,2,2);
      }
    }
    for (i=1; i<=nparam; i++) {
      tempv=colvec(hess,i,nparam);
      sbout2(glob_prnt.io,nparam,i,"hess (j,",")",tempv);
      free_dv(tempv);
    }
  }

  /*----------------------------------------------------------*
   *              Main loop of the algorithm                  *
   *----------------------------------------------------------*/

  nstop=1;
  *nIterPtr = 0 ;
  for (;;) {
     
    ++(*nIterPtr) ;
    out(miter,nparam,nob,nobL,nfsip,nineqn,nn,nineqn,ncnst1,
	ncsipl,ncsipn,mesh_pts,param->x,cs,ob,fM,fmax,steps,
	sktnom,d0nm,feasb);
    if (nstop==0) {
      if (!feasb) {
	dealloc1(nparam,nrowa,a,hess,hess1,di,d,gm,
                 grdpsf,penp,bl,bu,clamda,cvec,psmu,span,backup,
                 iact,iskip,istore);
	return;
      }
      for (i=1; i<=ncnst1; i++) cs[i].val=backup[i];
      for (i=1; i<=nob; i++) ob[i].val=backup[i+ncnst1];
      for (i=1; i<=neqn; i++)
	cs[glob_info.nnineq+i].mult = signeq[i]*psmu[i];
      dealloc1(nparam,nrowa,a,hess,hess1,di,d,gm,
	       grdpsf,penp,bl,bu,clamda,cvec,psmu,span,backup,
	       iact,iskip,istore);
      return;
    }
    if (!feasb && glob_prnt.iprint==0) glob_prnt.iter++;
    /*   Update the SIP constraint set Omega_k  */
    if ((ncsipl+ncsipn)!=0 || nfsip)
      update_omega(nparam,ncsipl,ncsipn,mesh_pts,nineqn,nob,nobL,
		   nfsip,steps,fmax,cs,ob,param->x,viol,
		   constr,obj,gradob,gradcn,param->cd,feasb);
    /*   Compute search direction               */
    dir(nparam,nob,nobL,nfsip,nineqn,neq,neqn,nn,ncsipl,ncsipn,
	ncnst1,feasb,&steps,epskt,epseqn,&sktnom,&scvneq,Ck,&d0nm,
	&grdftd,indxob,indxcn,iact,&iskp,iskip,istore,param,di,d,
	cs,ob,&fM,&fMp,&fmax,&psf,grdpsf,penp,a,bl,bu,clamda,cvec,
	hess,hess1,backup,signeq,viol,obj,constr);
    if (nstop==0 && !glob_log.get_ne_mult) continue;
    glob_log.first=FALSE;
    if (!glob_log.update && !glob_log.d0_is0) {
      /*   Determine step length                                */
      step1(nparam,nob,nobL,nfsip,nineqn,neq,neqn,nn,ncsipl,ncsipn,
	    ncnst1,&ncg,&ncf,indxob,indxcn,iact,&iskp,iskip,istore,
	    feasb,grdftd,ob,&fM,&fMp,&fmax,&psf,penp,&steps,&scvneq,
	    bu,param->x,di,d,cs,backup,signeq,viol,obj,constr,
	    param->cd);
      if (nstop==0) continue;
    }
    /*   Update the Hessian                                      */
    hessian(nparam,nob,nfsip,nobL,nineqn,neq,neqn,nn,ncsipn,ncnst1,
	    nfs,&nstart,feasb,bu,param,ob,fmax,&fM,&fMp,&psf,grdpsf,
	    penp,cs,gm,indxob,indxcn,bl,clamda,di,hess,d,steps,&nrst,
	    signeq,span,obj,constr,gradob,gradcn,hess1,cvec,psmu,viol);
    if (nstop==0 || glob_info.mode==0) continue;
    if (d0nm>dbar) Ck=DMAX1(0.5e0*Ck,Cbar);
    if (d0nm<=dbar && glob_log.dlfeas) Ck=Ck;
    if (d0nm<=dbar && !glob_log.dlfeas &&
	!glob_log.rhol_is1) Ck=10.e0*Ck;
  }
}

/*******************************************************************/
/*    Free up memory used by CFSQP1                                */
/*******************************************************************/

#ifdef __STDC__
static void
dealloc1(int nparam,int nrowa,bioReal **a,bioReal **hess,bioReal **hess1,
         bioReal *di,bioReal *d,bioReal *gm,bioReal *grdpsf,bioReal *penp,
         bioReal *bl,bioReal *bu,bioReal *clamda,bioReal *cvec,bioReal *psmu,
         bioReal *span,bioReal *backup,int *iact,int *iskip,int *istore)
#else
  static void
dealloc1(nparam,nrowa,a,hess,hess1,di,d,gm,grdpsf,penp,bl,bu,clamda,
         cvec,psmu,span,backup,iact,iskip,istore)
  int     nparam,nrowa;
bioReal  **a,**hess,**hess1;
bioReal  *di,*d,*gm,*grdpsf,*penp,*bl,*bu,*clamda,*cvec,*psmu,*span,
  *backup;
int     *iact,*iskip,*istore;
#endif
{
  free_dm(a,nrowa);
  free_dm(hess,nparam);
  free_dm(hess1,nparam+1);
  free_dv(di);
  free_dv(d);
  free_dv(gm);
  free_dv(grdpsf);
  free_dv(penp);
  free_dv(bl);
  free_dv(bu);
  free_dv(clamda);
  free_dv(cvec);
  free_dv(psmu);
  free_dv(span);
  free_dv(backup);
  free_iv(iact);
  free_iv(iskip);
  free_iv(istore);
}
/************************************************************/
/*   CFSQP - Check the input data                           */
/************************************************************/

#ifdef __STDC__
static void
check(int nparam,int nf,int nfsip,int *Linfty,int nineq,
      int nnl,int neq,int neqn,int ncsipl,int ncsipn,int mode,
      int *modem,bioReal eps,bioReal bigbnd,struct _parameter *param)
#else
  static void
check(nparam,nf,nfsip,Linfty,nineq,nnl,neq,neqn,ncsipl,ncsipn,
      mode,modem,eps,bigbnd,param)
  int     nparam,nf,nfsip,nineq,nnl,neq,neqn,ncsipl,ncsipn,mode,*modem,
  *Linfty;
bioReal  bigbnd,eps;
struct  _parameter *param;
#endif
{
  int i;
  bioReal bli,bui;

  if (nparam<=0)
    error("nparam should be positive!                ",
	  &glob_prnt.info);
  if (nf<0)
    error("nf should not be negative!                ",
	  &glob_prnt.info);
  if (nineq<0)
    error("nineq should not be negative!             ",
	  &glob_prnt.info);
  if (nineq>=0 && nnl>=0 && nineq<nnl)
    error("nineq should be no smaller than nnl!     ",
	  &glob_prnt.info);
  if (neqn<0)
    error("neqn should not be negative!              ",
	  &glob_prnt.info);
  if (neq<neqn)
    error("neq should not be smaller than neqn      ",
	  &glob_prnt.info);
  if (nf<nfsip)
    error("nf should not be smaller than nfsip      ",
	  &glob_prnt.info);
  if (nineq<ncsipn+ncsipl)
    error("ncsrl+ncsrn should not be larger than nineq",
	  &glob_prnt.info);
  if (nparam<=neq-neqn)
    error("Must have nparam > number of linear equalities",
	  &glob_prnt.info);
  if (glob_prnt.iprint<0 || glob_prnt.iprint>3)
    error("iprint mod 10 should be 0,1,2 or 3!       ",
	  &glob_prnt.info);
  if (eps<=glob_grd.epsmac) {
    error("eps should be bigger than epsmac!         ",
	  &glob_prnt.info);
    DETAILED_MESSAGE("epsmac = " << std::setw(22) << std::setprecision(14) 
		     << std::setiosflags(std::ios::scientific|std::ios::showpos) 
		     << glob_grd.epsmac << " which is machine dependent") 

  }
  if (!(mode==100 || mode==101 || mode==110 || mode==111
	|| mode==200 || mode==201 || mode==210 || mode==211))
    error("mode is not properly specified!           ",
	  &glob_prnt.info);
  if (glob_prnt.info!=0) {
    DETAILED_MESSAGE("Error: Input parameters are not consistent.")
    return;
  }
  for (i=1; i<=nparam; i++) {
    bli=param->bl[i];
    bui=param->bu[i];
    if (bli>bui) {
      DETAILED_MESSAGE("lower bounds should be smaller than upper bounds")
      glob_prnt.info=7;
    }
    if (glob_prnt.info!=0) return;
    if (bli<(-bigbnd)) param->bl[i]=-bigbnd;
    if (bui>bigbnd) param->bu[i]=bigbnd;
  }
  if (mode >= 200) {
    i=mode-200;
    glob_info.modec=2;
  } else {
    i=mode-100;
    glob_info.modec=1;
  }
  if (i<10) *modem=0;
  else {
    *modem=1;
    i-=10;
  }
  if (!i) *Linfty=FALSE;
  else *Linfty=TRUE;
}
/****************************************************************/
/*    CFSQP : Generate a feasible point satisfying simple       */
/*            bounds and linear constraints.                    */
/****************************************************************/

#ifdef __STDC__
static void
initpt(int nparam,int nnl,int neq,int neqn,int nclin,int nctotl,
       int nrowa,struct _parameter *param,struct _constraint *cs,
       void (*constr)(int,int,bioReal *,bioReal *,void *),
       void (*gradcn)(int,int,bioReal *,bioReal *,
		      void (*)(int,int,bioReal *,bioReal *,void *),void *))
#else
  static void
initpt(nparam,nnl,neq,neqn,nclin,nctotl,nrowa,param,cs,
       constr,gradcn)
  int     nparam,nnl,neq,neqn,nclin,nctotl,nrowa;
struct _constraint *cs;
struct _parameter  *param;
void    (* constr)(),(* gradcn)();
#endif
{
  int i,j,infoql,mnn,temp1,iout,zero;
  bioReal x0i,*atemp,*htemp;
  bioReal *x,*bl,*bu,*cvec,*clamda,*bj;
  bioReal **a,**hess;

  hess=make_dm(nparam,nparam);
  a=make_dm(nrowa,nparam);
  x=make_dv(nparam);
  bl=make_dv(nctotl);
  bu=make_dv(nctotl);
  cvec=make_dv(nparam);
  clamda=make_dv(nctotl+nparam+1);
  bj=make_dv(nclin);

  glob_prnt.info=1;
  for (i=1; i<=nclin; i++) {
    glob_grd.valnom=cs[i].val;
    j=i+nnl;
    if (j<=glob_info.nnineq)
      gradcn(nparam,j,(param->x)+1,cs[i].grad+1,constr,param->cd);
    else gradcn(nparam,j+neqn,(param->x)+1,cs[i].grad+1,constr,
		param->cd);
  }
  for (i=1; i<=nparam; i++) {
    x0i=param->x[i];
    bl[i]=param->bl[i]-x0i;
    bu[i]=param->bu[i]-x0i;
    cvec[i]=0.e0;
  }
  for (i=nclin; i>=1; i--)
    bj[nclin-i+1]=-cs[i].val;
  for (i=nclin; i>=1; i--)
    for (j=1; j<=nparam; j++)
      a[nclin-i+1][j]=-cs[i].grad[j];
  diagnl(nparam,1.e0,hess);
  nullvc(nparam,x);

  iout=6;
  zero=0;
  mnn=nrowa+2*nparam;
  iw[1]=1;
  temp1=neq-neqn;
  htemp=convert(hess,nparam,nparam);
  atemp=convert(a,nrowa,nparam);

  ql0001_(&nclin,&temp1,&nrowa,&nparam,&nparam,&mnn,(htemp+1),
	  (cvec+1),(atemp+1),(bj+1),(bl+1),(bu+1),(x+1),(clamda+1),
	  &iout,&infoql,&zero,(w+1),&lenw,(iw+1),&leniw,
	  &glob_grd.epsmac);

  free_dv(htemp);
  free_dv(atemp);
  if (infoql==0) {
    for (i=1; i<=nparam; i++)
      param->x[i]=param->x[i]+x[i];
    x_is_new=TRUE;
    for (i=1; i<=nclin; i++) {
      j=i+nnl;
      if (j<=glob_info.nnineq) constr(nparam,j,(param->x)+1,
				      &(cs[i].val),param->cd);
      else constr(nparam,j+neqn,(param->x)+1,&(cs[i].val),param->cd);
    }
    glob_prnt.info=0;
  }
  if (glob_prnt.info==1 && glob_prnt.iprint!=0) {
    DETAILED_MESSAGE("\n Error: No feasible point is found for the\n linear constraints.")
  }
  free_dm(a,nrowa);
  free_dm(hess,nparam);
  free_dv(x);
  free_dv(bl);
  free_dv(bu);
  free_dv(cvec);
  free_dv(clamda);
  free_dv(bj);
  return;
}
/****************************************************************/
/*   CFSQP : Update the SIP "active" objective and constraint   */
/*           sets Omega_k and Xi_k.                             */
/****************************************************************/

#ifdef __STDC__
static void
update_omega(int nparam,int ncsipl,int ncsipn,int *mesh_pts,
	     int nineqn,int nob,int nobL,int nfsip,bioReal steps,
	     bioReal fmax,struct _constraint *cs,struct _objective *ob,
	     bioReal *x,struct _violation *viol,
	     void (*constr)(int,int,bioReal *,bioReal *,void *),
	     void (*obj)(int,int,bioReal *,bioReal *,void *),
	     void (*gradob)(int,int,bioReal *,bioReal *,
			    void (*)(int,int,bioReal *,bioReal *,void *),void *),
	     void (*gradcn)(int,int,bioReal *,bioReal *,
			    void (*)(int,int,bioReal *,bioReal *,void *),void *),
	     void *cd,int feasb)
#else
  static void
update_omega(nparam,ncsipl,ncsipn,mesh_pts,nineqn,nob,nobL,nfsip,
	     steps,fmax,cs,ob,x,viol,constr,obj,gradob,gradcn,cd,feasb)
  int     nparam,ncsipl,ncsipn,*mesh_pts,nineqn,nobL,nob,nfsip,feasb;
bioReal  *x,steps,fmax;
struct _constraint *cs;
struct _objective *ob;
struct _violation *viol;
void    (* constr)();
void    (* obj)();
void    (* gradob)();
void    (* gradcn)();
void    *cd;
#endif
{
  int i,j,i_max,index,offset,nineq,display;
  bioReal epsilon,g_max,fprev,fnow,fnext,fmult;

  epsilon=1.e0;
  glob_info.tot_actf_sip=glob_info.tot_actg_sip=0;
  nineq=glob_info.nnineq;
  if (glob_prnt.iter%glob_prnt.iter_mod) display=FALSE;
  else display=TRUE;
  /* Clear previous constraint sets                   */
  for (i=1; i<=ncsipl; i++)
    cs[nineq-ncsipl+i].act_sip=FALSE;
  for (i=1; i<=ncsipn; i++)
    cs[nineqn-ncsipn+i].act_sip=FALSE;
  /* Clear previous objective sets                    */
  for (i=nob-nfsip+1; i<=nob; i++)
    ob[i].act_sip=FALSE;

  /*--------------------------------------------------*/
  /* Update Constraint Sets Omega_k                   */
  /*--------------------------------------------------*/

  if (ncsipn != 0) {
    offset=nineqn-ncsipn;
    for (i=1; i<=glob_info.ncsipn; i++) {
      for (j=1; j<=mesh_pts[glob_info.nfsip+i]; j++) {
	offset++;
	if (j==1) {
	  if (cs[offset].val >= -epsilon &&
	      cs[offset].val>=cs[offset+1].val) {
	    cs[offset].act_sip=TRUE;
	    glob_info.tot_actg_sip++;
	    if (cs[offset].mult==0.e0 && !glob_log.first) {
	      glob_grd.valnom=cs[offset].val;
	      gradcn(nparam,offset,x+1,cs[offset].grad+1,constr,
		     cd);
	    }
	    continue;
	  }
	} else if (j==mesh_pts[glob_info.nfsip+i]) {
	  if (cs[offset].val >= -epsilon &&
	      cs[offset].val>cs[offset-1].val) {
	    cs[offset].act_sip=TRUE;
	    glob_info.tot_actg_sip++;
	    if (cs[offset].mult==0.e0 && !glob_log.first) {
	      glob_grd.valnom=cs[offset].val;
	      gradcn(nparam,offset,x+1,cs[offset].grad+1,constr,
		     cd);
	    }
	    continue;
	  }
	} else {
	  if (cs[offset].val >= -epsilon && cs[offset].val >
	      cs[offset-1].val && cs[offset].val >=
	      cs[offset+1].val) {
	    cs[offset].act_sip=TRUE;
	    glob_info.tot_actg_sip++;
	    if (cs[offset].mult==0.e0 && !glob_log.first) {
	      glob_grd.valnom=cs[offset].val;
	      gradcn(nparam,offset,x+1,cs[offset].grad+1,constr,
		     cd);
	    }
	    continue;
	  }
	}
	if (cs[offset].val >= -glob_grd.epsmac) {
	  cs[offset].act_sip=TRUE;
	  glob_info.tot_actg_sip++;
	  if (cs[offset].mult==0.e0 && !glob_log.first) {
	    glob_grd.valnom=cs[offset].val;
	    gradcn(nparam,offset,x+1,cs[offset].grad+1,constr,cd);
	  }
	  continue;
	}
	if (cs[offset].mult>0.e0) {
	  cs[offset].act_sip=TRUE;
	  glob_info.tot_actg_sip++;
	}
	/* Add if binding for d1  */
	if (cs[offset].d1bind) {
	  cs[offset].act_sip=TRUE;
	  glob_info.tot_actg_sip++;
	  if (cs[offset].mult==0.e0 && !glob_log.first) {
	    glob_grd.valnom=cs[offset].val;
	    gradcn(nparam,offset,x+1,cs[offset].grad+1,constr,cd);
	  }
	}

      }
    }
  }
  if (ncsipl != 0) {
    /* Don't need to get gradients */
    offset=nineq-ncsipl;
    for (i=1; i<=glob_info.ncsipl; i++) {
      if (feasb) index=glob_info.nfsip+glob_info.ncsipn+i;
      else index=glob_info.ncsipn+i;
      for (j=1; j<=mesh_pts[index]; j++) {
	offset++;
	if (j==1) {
	  if (cs[offset].val >= -epsilon &&
	      cs[offset].val>=cs[offset+1].val) {
	    cs[offset].act_sip=TRUE;
	    glob_info.tot_actg_sip++;
	    continue;
	  }
	} else
	  if (j==mesh_pts[index]) {
	    if (cs[offset].val >= -epsilon &&
		cs[offset].val>cs[offset-1].val) {
	      cs[offset].act_sip=TRUE;
	      glob_info.tot_actg_sip++;
	      continue;
	    }
	  } else {
	    if (cs[offset].val >= -epsilon && cs[offset].val >
		cs[offset-1].val && cs[offset].val >=
		cs[offset+1].val) {
	      cs[offset].act_sip=TRUE;
	      glob_info.tot_actg_sip++;
	      continue;
	    }
	  }
	if (cs[offset].val >= -glob_grd.epsmac ||
	    cs[offset].mult>0.e0 || cs[offset].d1bind) {
	  cs[offset].act_sip=TRUE;
	  glob_info.tot_actg_sip++;
	}
      }
    }
  }
  /* Include some extra points during 1st iteration        */
  /* (gradients are already evaluated for first iteration) */
  /* Current heuristics: maximizers and end-points.        */
  if (glob_log.first) {
    if (feasb) {
      offset=nineqn-ncsipn;
      for (i=1; i<=glob_info.ncsipn; i++) {
	i_max= ++offset;
	g_max=cs[i_max].val;
	if (!cs[i_max].act_sip) { /* add first point       */
	  cs[i_max].act_sip=TRUE;
	  glob_info.tot_actg_sip++;
	}
	for (j=2;j<=mesh_pts[glob_info.nfsip+i];j++) {
	  offset++;
	  if (cs[offset].val>g_max) {
	    i_max=offset;
	    g_max=cs[i_max].val;
	  }
	}
	if (!cs[i_max].act_sip) {
	  cs[i_max].act_sip=TRUE;
	  glob_info.tot_actg_sip++;
	}
	if (!cs[offset].act_sip) { /* add last point          */
	  cs[offset].act_sip=TRUE;
	  glob_info.tot_actg_sip++;
	}
      }
    }
    offset=nineq-ncsipl;
    for (i=1; i<=glob_info.ncsipl; i++) {
      i_max= ++offset;
      g_max=cs[i_max].val;
      if (!cs[i_max].act_sip) { /* add first point       */
	cs[i_max].act_sip=TRUE;
	glob_info.tot_actg_sip++;
      }
      if (feasb) index=glob_info.nfsip+glob_info.ncsipn+i;
      else index=glob_info.ncsipn+i;
      for (j=2;j<=mesh_pts[index]; j++) {
	offset++;
	if (cs[offset].val>g_max) {
	  i_max=offset;
	  g_max=cs[i_max].val;
	}
      }
      if (!cs[i_max].act_sip) {
	cs[i_max].act_sip=TRUE;
	glob_info.tot_actg_sip++;
      }
      if (!cs[offset].act_sip) { /* add last point          */
	cs[offset].act_sip=TRUE;
	glob_info.tot_actg_sip++;
      }
    }
  }

  /* If necessary, append xi_bar                              */
  if (steps<1.e0 && viol->type==CONSTR) {
    i=viol->index;
    if (!cs[i].act_sip) {
      cs[i].act_sip=TRUE;
      glob_info.tot_actg_sip++;
    }
  }
  if (glob_prnt.iprint>=2 && display)
    DETAILED_MESSAGE(" |Xi_k| for g       " << std::setw(22) 
		     << std::setprecision(14) 
		     << std::setiosflags(std::ios::scientific|std::ios::showpos) 
		     << glob_info.tot_actg_sip)

  for (i=1; i<=ncsipl; i++)
    cs[nineq-ncsipl+i].d1bind=FALSE;
  for (i=1; i<=ncsipn; i++)
    cs[nineqn-ncsipn+i].d1bind=FALSE;

  /*---------------------------------------------------------*/
  /* Update Objective Set Omega_k                                    */
  /*---------------------------------------------------------*/

  if (nfsip) {
    offset=nob-nfsip;
    if (feasb) index=glob_info.nfsip;
    else index=glob_info.ncsipn;
    for (i=1; i<=index; i++) {
      for (j=1; j<=mesh_pts[i]; j++) {
	offset++;
	if (nobL>nob) {
	  fnow=fabs(ob[offset].val);
	  fmult=DMAX1(fabs(ob[offset].mult),
		      fabs(ob[offset].mult_L));
	} else {
	  fnow=ob[offset].val;
	  fmult=ob[offset].mult;
	}
	if (j==1) {
	  if (nobL>nob) fnext=fabs(ob[offset+1].val);
	  else fnext=ob[offset+1].val;
	  if ((fnow>=fmax-epsilon)&& fnow>=fnext) {
	    ob[offset].act_sip=TRUE;
	    glob_info.tot_actf_sip++;
	    if (fmult==0.e0 && !glob_log.first) {
	      glob_grd.valnom=ob[offset].val;
	      if (feasb) gradob(nparam,offset,x+1,
				ob[offset].grad+1,obj,cd);
	      else gradcn(nparam,offset,x+1,ob[offset].grad+1,
			  constr,cd);
	    }
	    continue;
	  }
	} else if (j==mesh_pts[i]) {
	  if (nobL>nob) fprev=fabs(ob[offset-1].val);
	  else fprev=ob[offset-1].val;
	  if ((fnow>=fmax-epsilon)&& fnow>fprev) {
	    ob[offset].act_sip=TRUE;
	    glob_info.tot_actf_sip++;
	    if (fmult==0.e0 && !glob_log.first) {
	      glob_grd.valnom=ob[offset].val;
	      if (feasb) gradob(nparam,offset,x+1,
				ob[offset].grad+1,obj,cd);
	      else gradcn(nparam,offset,x+1,ob[offset].grad+1,
			  constr,cd);
	    }
	    continue;
	  }
	} else {
	  if (nobL>nob) {
	    fprev=fabs(ob[offset-1].val);
	    fnext=fabs(ob[offset+1].val);
	  } else {
	    fprev=ob[offset-1].val;
	    fnext=ob[offset+1].val;
	  }
	  if ((fnow>=fmax-epsilon)&& fnow>fprev &&
	      fnow>=fnext) {
	    ob[offset].act_sip=TRUE;
	    glob_info.tot_actf_sip++;
	    if (fmult==0.e0 && !glob_log.first) {
	      glob_grd.valnom=ob[offset].val;
	      if (feasb) gradob(nparam,offset,x+1,
				ob[offset].grad+1,obj,cd);
	      else gradcn(nparam,offset,x+1,ob[offset].grad+1,
			  constr,cd);
	    }
	    continue;
	  }
	}
	if (fnow>= fmax-glob_grd.epsmac && !ob[offset].act_sip) {
	  ob[offset].act_sip=TRUE;
	  glob_info.tot_actf_sip++;
	  if (fmult==0.e0 && !glob_log.first) {
	    glob_grd.valnom=ob[offset].val;
	    if (feasb) gradob(nparam,offset,x+1,
			      ob[offset].grad+1,obj,cd);
	    else gradcn(nparam,offset,x+1,ob[offset].grad+1,
			constr,cd);
	  }
	  continue;
	}
	if (fmult!=0.e0 && !ob[offset].act_sip) {
	  ob[offset].act_sip=TRUE;
	  glob_info.tot_actf_sip++;
	  continue;
	}
      }
    }
    /* Addition of objectives for first iteration.          */
    /* Current heuristics: maximizers and end-points        */
    if (glob_log.first) {
      offset=nob-nfsip;
      if (feasb) index=glob_info.nfsip;
      else index=glob_info.ncsipn;
      for (i=1; i<=index; i++) {
	i_max= ++offset;
	if (nobL==nob) g_max=ob[i_max].val;
	else g_max=fabs(ob[i_max].val);
	if (!ob[i_max].act_sip) { /* add first point       */
	  ob[i_max].act_sip=TRUE;
	  glob_info.tot_actf_sip++;
	}
	for (j=2;j<=mesh_pts[i];j++) {
	  offset++;
	  if (nobL==nob) fnow=ob[offset].val;
	  else fnow=fabs(ob[offset].val);
	  if (fnow>g_max) {
	    i_max=offset;
	    g_max=fnow;
	  }
	}
	if (!ob[i_max].act_sip) {
	  ob[i_max].act_sip=TRUE;
	  glob_info.tot_actf_sip++;
	}
	if (!ob[offset].act_sip) { /* add last point          */
	  ob[offset].act_sip=TRUE;
	  glob_info.tot_actf_sip++;
	}
      }
    }

    /* If necessary, append omega_bar                          */
    if (steps<1.e0 && viol->type==OBJECT) {
      i=viol->index;
      if (!ob[i].act_sip) {
	ob[i].act_sip=TRUE;
	glob_info.tot_actf_sip++;
      }
    }
    if (glob_prnt.iprint>=2 && display)
      DETAILED_MESSAGE(" |Omega_k| for f    " << std::setw(26) <<
		       glob_info.tot_actf_sip);
  }
  viol->type=NONE;
  viol->index=0;
  return;
}
/*******************************************************************/
/*   CFSQP : Computation of the search direction                   */
/*******************************************************************/

#ifdef __STDC__
static void
dqp(int,int,int,int,int,int,int,int,int,int,int,int,int,
    int,int,int *,struct _parameter *,bioReal *,int,
    struct _objective *,bioReal,bioReal *,struct _constraint *,
    bioReal **,bioReal *,bioReal *,bioReal *,bioReal *,
    bioReal **,bioReal **,bioReal *,bioReal,int);
static void
di1(int,int,int,int,int,int,int,int,int,int,int,int,int *,
    int,struct _parameter *,bioReal *,struct _objective *,
    bioReal,bioReal *,struct _constraint *,bioReal *,
    bioReal *,bioReal *,bioReal *,bioReal **,bioReal *,bioReal);
#else
static void dqp();
static void di1();
#endif

#ifdef __STDC__
static void
dir(int nparam,int nob,int nobL,int nfsip,int nineqn,int neq,int neqn,
    int nn,int ncsipl,int ncsipn,int ncnstr,
    int feasb,bioReal *steps,bioReal epskt,bioReal epseqn,
    bioReal *sktnom,bioReal *scvneq,bioReal Ck,bioReal *d0nm,
    bioReal *grdftd,int *indxob,int *indxcn,int *iact,int *iskp,
    int *iskip,int *istore,struct _parameter *param,bioReal *di,
    bioReal *d,struct _constraint *cs,struct _objective *ob,
    bioReal *fM,bioReal *fMp,bioReal *fmax,bioReal *psf,bioReal *grdpsf,
    bioReal *penp,bioReal **a,bioReal *bl,bioReal *bu,bioReal *clamda,
    bioReal *cvec,bioReal **hess,bioReal **hess1,
    bioReal *backup,bioReal *signeq,struct _violation *viol,
    void (*obj)(int,int,bioReal *,bioReal *,void *),
    void (*constr)(int,int,bioReal *,bioReal *,void *))
#else
  static void
dir(nparam,nob,nobL,nfsip,nineqn,neq,neqn,nn,ncsipl,ncsipn,ncnstr,
    feasb,steps,epskt,epseqn,sktnom,scvneq,Ck,d0nm,
    grdftd,indxob,indxcn,iact,iskp,iskip,istore,param,di,d,cs,ob,
    fM,fMp,fmax,psf,grdpsf,penp,a,bl,bu,clamda,cvec,hess,hess1,
    backup,signeq,viol,obj,constr)
  int     nparam,nob,nobL,nfsip,nineqn,neq,neqn,nn,ncsipl,ncsipn,ncnstr,
  *iskp,feasb;
int     *indxob,*indxcn,*iact,*iskip,*istore;
bioReal  *steps,epskt,epseqn,*sktnom,Ck,*d0nm,*grdftd,*fM,*fMp,
  *fmax,*psf,*scvneq;
bioReal  *di,*d,*grdpsf,*penp,**a,*bl,*bu,*clamda,*cvec,**hess,
  **hess1,*backup,*signeq;
struct  _constraint *cs;
struct  _objective  *ob;
struct  _parameter  *param;
struct  _violation  *viol;
void    (* obj)(),(* constr)();
#endif
{
  int  i,j,k,kk,ncg,ncf,nqprm0,nclin0,nctot0,infoqp,nqprm1,ncl,
    nclin1,ncc,nff,nrowa0,nrowa1,ninq,nobb,nobbL,
    nncn,ltem1,ltem2,display,need_d1;
  bioReal fmxl,vv,dx,dmx,dnm1,dnm,v0,v1,vk,temp1,temp2,theta,
    rhol,rhog,rho,grdfd0,grdfd1,dummy,grdgd0,grdgd1,thrshd,
    sign,*adummy,dnmtil,*tempv;

  ncg=ncf=*iskp=0;
  ncl=glob_info.nnineq-nineqn;
  glob_log.local=glob_log.update=FALSE;
  glob_log.rhol_is1=FALSE;
  thrshd=tolfea;
  adummy=make_dv(1);
  adummy[1]=0.e0;
  dummy=0.e0;
  temp1=temp2=0.e0;
  if (glob_prnt.iter%glob_prnt.iter_mod) display=FALSE;
  else display=TRUE;
  need_d1=TRUE;

  if (nobL<=1) {
    nqprm0=nparam;
    nclin0=ncnstr;
  } else {
    nqprm0=nparam+1;
    nclin0=ncnstr+nobL;
  }
  nctot0=nqprm0+nclin0;
  vv=0.e0;
  nrowa0=DMAX1(nclin0,1);
  for (i=1; i<=ncnstr; i++) {
    if (feasb) {
      if (i>nineqn && i<=glob_info.nnineq)
	iskip[glob_info.nnineq+2-i]=i;
      iw[i]=i;
    } else {
      if (i<=ncl) iskip[ncl+2-i]=nineqn+i;
      if (i<=ncl) iw[i]=nineqn+i;
      if (i>ncl) iw[i]=nineqn+neqn+i;
    }
  }
  for (i=1; i<=nob; i++)
    iw[ncnstr+i]=i;
  nullvc(nparam, cvec);
  glob_log.d0_is0=FALSE;
  dqp(nparam,nqprm0,nob,nobL,nfsip,nineqn,neq,neqn,nn,ncsipl,ncsipn,
      ncnstr,nctot0,nrowa0,nineqn,&infoqp,param,di,feasb,ob,
      *fmax,grdpsf,cs,a,cvec,bl,bu,clamda,hess,hess1,di,vv,0);
  if (infoqp!=0) {
    glob_prnt.info=5;
    if (!feasb) glob_prnt.info=2;
    nstop=0;
    free_dv(adummy);
    return;
  }
  /*-------------------------------------------------------------*/
  /*    Reorder indexes of constraints & objectives              */
  /*-------------------------------------------------------------*/
  if (nn>1) {
    j=1;
    k=nn;
    for (i=nn; i>=1; i--) {
      if (fuscmp(cs[indxcn[i]].mult,thrshd)) {
	iact[j]=indxcn[i];
	j++;
      } else {
	iact[k]=indxcn[i];
	k--;
      }
    }
  }
  if (nobL>1) {
    j=nn+1;
    k=nn+nob;
    for (i=nob; i>=1; i--) {
      kk=nqprm0+ncnstr;
      ltem1=fuscmp(ob[i].mult,thrshd);
      ltem2=(nobL!=nob) && (fuscmp(ob[i].mult_L,thrshd));
      if (ltem1 || ltem2) {
	iact[j]=i;
	j++;
      } else {
	iact[k]=i;
	k--;
      }
    }
  }
  if (nob>0) vv=ob[iact[nn+1]].val;
  *d0nm=sqrt(scaprd(nparam,di,di));
  if (glob_log.first && nclin0==0) {
    dx=sqrt(scaprd(nparam,param->x,param->x));
    dmx=DMAX1(dx,1.e0);
    if (*d0nm>dmx) {
      for (i=1; i<=nparam; i++)
	di[i]=di[i]*dmx/(*d0nm);
      *d0nm=dmx;
    }
  }
  matrvc(nparam,nparam,hess,di,w);
  if (nn==0) *grdftd = -scaprd(nparam,w,di);
  *sktnom=sqrt(scaprd(nparam,w,w));
  if (((*d0nm<=epskt)||((gLgeps>0.e0)&&(*sktnom<=gLgeps)))&&
      (neqn==0 || *scvneq<=epseqn)) {
    /* We are finished! */
    nstop=0;
    if (feasb && glob_log.first && neqn!=0) {
      /*  Finished, but still need to estimate nonlinear equality
	  constraint multipliers   */
      glob_log.get_ne_mult = TRUE;
      glob_log.d0_is0 = TRUE;
    }
    if (!feasb) glob_prnt.info=2;
    free_dv(adummy);
    if (glob_prnt.iprint<3 || !display) return;
    if (nobL<=1) nff=1;
    if (nobL>1) nff=2;
    sbout1(glob_prnt.io,nparam,"multipliers for x   ",dummy,
	   param->mult,2,2);
    if (ncnstr!=0) {
      DETAILED_MESSAGE("\t\t\t            for g    \t "<< std::setw(22) << std::setprecision(14) 
		       << std::setiosflags(std::ios::scientific|std::ios::showpos) 
		       << cs[1].mult);
      for (j=2; j<=ncnstr; j++)
	DETAILED_MESSAGE(" \t\t\t\t\t\t " << std::setw(22) 
			 << std::setprecision(14) 
			 << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			 << cs[j].mult);
    }
    if (nobL>1) {
      DETAILED_MESSAGE("\t\t\t            for f    \t " 
		       << std::setw(22) 
		       << std::setprecision(14) 
		       << std::setiosflags(std::ios::scientific|std::ios::showpos) 
		       << ob[1].mult);
      for (j=2; j<=nob; j++)
	DETAILED_MESSAGE(" \t\t\t\t\t\t " << std::setw(22) 
			 << std::setprecision(14) 
			 << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			 << ob[j].mult);
    }
    return;
  }
  if (glob_prnt.iprint>=3 && display) {
    sbout1(glob_prnt.io,nparam,"d0                  ",dummy,di,2,2);
    sbout1(glob_prnt.io,0,"d0norm              ",*d0nm,adummy,1,2);
    sbout1(glob_prnt.io,0,"ktnorm              ",*sktnom,adummy,1,2);
  }
  if (neqn!=0 && *d0nm<=DMIN1(0.5e0*epskt,(0.1e-1)*glob_grd.rteps)
      && *scvneq>epseqn) {
    /* d0 is "zero", but equality constraints not satisfied  */
    glob_log.d0_is0 = TRUE;
    return;
  }
  /*--------------------------------------------------------------*/
  /*     Single objective without nonlinear constraints requires  */
  /*     no d1 and dtilde; multi-objectives without nonlinear     */
  /*     constraints requires no d1.                              */
  /*--------------------------------------------------------------*/
  if (nn!=0) *grdftd=slope(nob,nobL,neqn,nparam,feasb,ob,grdpsf,
                           di,d,*fmax,dummy,0,adummy,0);

  if (nn==0 && nobL<=1) {
    for (i=1; i<=nparam; i++) d[i]=0.e0;
    dnmtil=0.e0;
    free_dv(adummy);
    return;
  }
  if (nn==0) {
    dnm=*d0nm;
    rho=0.e0;
    rhog=0.e0;
    goto L310;
  }
  /*-------------------------------------------------------------*/
  /*     compute modified first order direction d1               */
  /*-------------------------------------------------------------*/

  /* First check that it is necessary */
  if (glob_info.mode==1) {
    vk=DMIN1(Ck*(*d0nm)*(*d0nm),*d0nm);
    need_d1=FALSE;
    for (i=1; i<=nn; i++) {
      tempv=cs[indxcn[i]].grad;
      grdgd0=scaprd(nparam,tempv,di);
      temp1=vk+cs[indxcn[i]].val+grdgd0;
      if (temp1>0.e0) {
	need_d1=TRUE;
	break;
      }
    }
  }
  if (need_d1) {
    nqprm1=nparam+1;
    if (glob_info.mode==0) nclin1=ncnstr+DMAX1(nobL,1);
    if (glob_info.mode==1) nclin1=ncnstr;
    nrowa1=DMAX1(nclin1,1);
    ninq=glob_info.nnineq;
    di1(nparam,nqprm1,nob,nobL,nfsip,nineqn,neq,neqn,ncnstr,
	ncsipl,ncsipn,nrowa1,&infoqp,glob_info.mode,
	param,di,ob,*fmax,grdpsf,cs,cvec,bl,bu,clamda,
	hess1,d,*steps);
    if (infoqp!=0) {
      glob_prnt.info=6;
      if (!feasb) glob_prnt.info=2;
      nstop=0;
      free_dv(adummy);
      return;
    }
    dnm1=sqrt(scaprd(nparam,d,d));
    if (glob_prnt.iprint>=3 && display) {
      sbout1(glob_prnt.io,nparam,"d1                  ",dummy,d,2,2);
      sbout1(glob_prnt.io,0,"d1norm              ",dnm1,adummy,1,2);
    }
  } else {
    dnm1=0.e0;
    for (i=1; i<=nparam; i++) d[i]=0.e0;
  }
  if (glob_info.mode!=1) {
    v0=pow(*d0nm,2.1);
    v1=DMAX1(0.5e0,pow(dnm1,2.5));
    rho=v0/(v0+v1);
    rhog=rho;
  } else {
    rhol=0.e0;
    if (need_d1) {
      for (i=1; i<=nn; i++) {
	tempv=cs[indxcn[i]].grad;
	grdgd0=scaprd(nparam,tempv,di);
	grdgd1=scaprd(nparam,tempv,d);
	temp1=vk+cs[indxcn[i]].val+grdgd0;
	temp2=grdgd1-grdgd0;
	if (temp1<=0.e0) continue;
	if (fabs(temp2)<glob_grd.epsmac) {
	  rhol=1.e0;
	  glob_log.rhol_is1=TRUE;
	  break;
	}
	rhol=DMAX1(rhol,-temp1/temp2);
	if (temp2<0.e0 && rhol<1.e0) continue;
	rhol=1.e0;
	glob_log.rhol_is1=TRUE;
	break;
      }
    }
    theta=0.2e0;
    if (rhol==0.e0) {
      rhog=rho=0.e0;
      dnm=*d0nm;
      goto L310;
    }
    if (nobL>1) {
      rhog=slope(nob,nobL,neqn,nparam,feasb,ob,grdpsf,
		 di,d,*fmax,theta,glob_info.mode,adummy,0);
      rhog=DMIN1(rhol,rhog);
    } else {
      grdfd0= *grdftd;
      if (nob==1) grdfd1=scaprd(nparam,ob[1].grad,d);
      else grdfd1=0.e0;
      grdfd1=grdfd1-scaprd(nparam,grdpsf,d);
      temp1=grdfd1-grdfd0;
      temp2=(theta-1.e0)*grdfd0/temp1;
      if(temp1<=0.e0) rhog=rhol;
      else rhog=DMIN1(rhol,temp2);
    }
    rho=rhog;
    if (*steps==1.e0 && rhol<0.5e0) rho=rhol;
  }
  for (i=1; i<=nparam; i++) {
    if (rho!=rhog) cvec[i]=di[i];
    di[i]=(1.e0-rho)*di[i]+rho*d[i];
  }
  dnm=sqrt(scaprd(nparam,di,di));
  if (!(glob_prnt.iprint<3 || glob_info.mode==1 || nn==0)&&display) {
    sbout1(glob_prnt.io,0,"rho                 ",rho,adummy,1,2);
    sbout1(glob_prnt.io,nparam,"d                   ",dummy,di,2,2);
    sbout1(glob_prnt.io,0,"dnorm               ",dnm,adummy,1,2);
  }
 L310:
  for (i=1; i<=nob; i++) bl[i]=ob[i].val;
  if (rho!=1.e0) {
    if (!(glob_prnt.iprint!=3 || glob_info.mode==0 || nn==0)
	&& display) {
      sbout1(glob_prnt.io,0,"Ck                  ",Ck,adummy,1,2);
      sbout1(glob_prnt.io,0,"rhol                ",rho,adummy,1,2);
      sbout1(glob_prnt.io,nparam,"dl                  ",dummy,di,2,2);
      sbout1(glob_prnt.io,0,"dlnorm              ",dnm,adummy,1,2);
    }
    if (glob_info.mode!=0) {
      glob_log.local=TRUE;
      step1(nparam,nob,nobL,nfsip,nineqn,neq,neqn,nn,ncsipl,ncsipn,
	    ncnstr,&ncg,&ncf,indxob,indxcn,iact,iskp,iskip,istore,
	    feasb,*grdftd,ob,fM,fMp,fmax,psf,penp,steps,scvneq,bu,
	    param->x,di,d,cs,backup,signeq,viol,obj,constr,param->cd);
      if (!glob_log.update) nstop=1;
      else {
	free_dv(adummy);
	return;
      }
      glob_log.local=FALSE;
      if (rho!=rhog && nn!=0)
	for (i=1; i<=nparam; i++)
	  di[i]=(1-rhog)*cvec[i]+rhog*d[i];
      dnm=sqrt(scaprd(nparam,di,di));
    }
  }
  if (!(glob_prnt.iprint<3 || glob_info.mode==0 || nn==0) &&
      display) {
    sbout1(glob_prnt.io,0,"rhog                ",rhog,adummy,1,2);
    sbout1(glob_prnt.io,nparam,"dg                  ",dummy,di,2,2);
    sbout1(glob_prnt.io,0,"dgnorm              ",dnm,adummy,1,2);
  }
  if (rho !=0.e0) *grdftd=slope(nob,nobL,neqn,nparam,feasb,ob,
                                grdpsf,di,d,*fmax,theta,0,bl,1);
  if (glob_info.mode!=1 || rho!=rhog)
    for (i=1; i<=nparam; i++)
      bu[i]=param->x[i]+di[i];
  x_is_new=TRUE;
  if (rho!=rhog) ncg=0;
  ncc=ncg+1;
  fmxl=-bgbnd;
  ninq=nncn=ncg;
  j=0;
  /*--------------------------------------------------------------*/
  /*   iskip[1]-iskip[iskp] store the indexes of linear inequality*/
  /*   constraints that are not to be used to compute d~          */
  /*   iskip[nnineq-nineqn+1]-iskip[nnineq-ncn+1-iskp] store      */
  /*   those that are to be used to compute d~                    */
  /*--------------------------------------------------------------*/
  for (i=ncc; i<=ncnstr; i++) {
    if (i<=nn) kk=iact[i];
    else kk=indxcn[i];
    if (kk>nineqn && kk<=glob_info.nnineq) {
      iskip[ncl+1-j]=kk;
      j++;
    }
    if (kk<=glob_info.nnineq) {
      tempv=cs[kk].grad;
      temp1=dnm*sqrt(scaprd(nparam,tempv,tempv));
      temp2=cs[kk].mult;
    }
    if (temp2!=0.e0 || cs[kk].val >= (-0.2e0*temp1) ||
	kk>glob_info.nnineq) {
      ninq++;
      iw[ninq]=kk;
      if (feasb && kk<=nineqn) istore[kk]=1;
      constr(nparam,kk,bu+1,&(cs[kk].val),param->cd);
      if (!feasb || (feasb && (kk>glob_info.nnineq+neqn))) continue;
      if (kk<=nineqn) nncn=ninq;
      fmxl=DMAX1(fmxl,cs[kk].val);
      if (feasb &&(kk<=nineqn || (kk>glob_info.nnineq
				  && kk<=(glob_info.nnineq+neqn)))) glob_info.ncallg++;
      if (fabs(fmxl)>bgbnd) {
	for (i=1; i<=nparam; i++) d[i]=0.e0;
	dnmtil=0.e0;
	nstop=1;
	free_dv(adummy);
	return;
      }
      continue;
    }
    if (kk<=nineqn) continue;
    (*iskp)++;
    iskip[*iskp]=kk;
    j--;
  }
  if ((neqn!=0)&&(feasb))
    resign(nparam,neqn,psf,grdpsf,penp,cs,signeq,10,20);
  ninq-=neq;
  /*  if (!feasb) ninq+=neqn;   BUG???   */
  if (ncg!=0) for (i=1; i<=ncg; i++) {
    iw[i]=iact[i];
    if (iact[i]<=nineqn) istore[iact[i]]=1;
    fmxl=DMAX1(fmxl,cs[iact[i]].val);
    if (fabs(fmxl)>bgbnd) {
      for (i=1; i<=nparam; i++) d[i]=0.e0;
      dnmtil=0.e0;
      nstop=1;
      free_dv(adummy);
      return;
    }
  }
  if (nobL<=1) {
    iw[1+ninq+neq]=1;
    nobb=nob;
    goto L1110;
  }
  if (rho!=rhog) ncf=0;
  nff=ncf+1;
  nobb=ncf;
  sign=1.e0;
  fmxl=-bgbnd;
  if (ob[iact[nn+1]].mult<0.e0) sign=-1.e0;
  for (i=nff; i<=nob; i++) {
    kk=iact[nn+i];
    if (!feasb) kk=iact[i];
    if (feasb) k=nn+1;
    if (!feasb) k=1;
    for (j=1; j<=nparam; j++)
      w[nparam+j]=sign*ob[iact[k]].grad[j]-ob[kk].grad[j];
    temp1=fabs(ob[kk].val-sign*vv);
    temp2=dnm*sqrt(scaprd(nparam,&w[nparam],&w[nparam]));
    if (temp1!=0.e0 && temp2!=0.e0) {
      temp1=temp1/temp2;
      temp2=ob[kk].mult;
      if (temp2==0.e0 && temp1>0.2e0) continue;
    }
    nobb++;
    iw[nobb+ninq+neq]=kk;
    if (feasb) istore[nineqn+kk]=1;
    else istore[kk]=1;
    if (!feasb) {
      constr(nparam,indxob[kk],bu+1,&(ob[kk].val),param->cd);
      glob_info.ncallg++;
    } else {
      obj(nparam,kk,bu+1,&(ob[kk].val),param->cd);
      glob_info.ncallf++;
      if (nobL!=nob) fmxl=DMAX1(fmxl, -ob[kk].val);
    }
    fmxl=DMAX1(fmxl,ob[kk].val);
    if (fabs(fmxl)>bgbnd) {
      for (i=1; i<=nparam; i++) d[i]=0.e0;
      dnmtil=0.e0;
      nstop=1;
      free_dv(adummy);
      return;
    }
  }
  if (ncf != 0) {
    for (i=1; i<=ncf; i++) {
      iw[ninq+neq+i]=iact[i+nn];
      istore[nineqn+iact[i+nn]]=1;
      fmxl=DMAX1(fmxl,ob[iact[i+nn]].val);
      if (nobL != nob) fmxl=DMAX1(fmxl,-ob[iact[i+nn]].val);
      if (fabs(fmxl)>bgbnd) {
	for (i=1; i<=nparam; i++) d[i]=0.e0;
	dnmtil=0.e0;
	nstop=1;
	free_dv(adummy);
	return;
      }
    }
  }
 L1110:
  matrvc(nparam,nparam,hess,di,cvec);
  vv=-DMIN1(0.01e0*dnm,pow(dnm,2.5));
  /*--------------------------------------------------------------*/
  /*    compute a correction dtilde to d=(1-rho)d0+rho*d1         */
  /*--------------------------------------------------------------*/
  if (nobL!=nob) nobbL=2*nobb;
  if (nobL==nob) nobbL=nobb;
  if (nobbL<=1) {
    nqprm0=nparam;
    nclin0=ninq+neq;
  } else {
    nqprm0=nparam+1;
    nclin0=ninq+neq+nobbL;
  }
  nctot0=nqprm0+nclin0;
  nrowa0=DMAX1(nclin0,1);
  i=ninq+neq;
  nstop=1;
  dqp(nparam,nqprm0,nobb,nobbL,nfsip,nncn,neq,neqn,nn,ncsipl,ncsipn,i,
      nctot0,nrowa0,nineqn,&infoqp,param,di,feasb,ob,fmxl,
      grdpsf,cs,a,cvec,bl,bu,clamda,hess,hess1,d,vv,1);
  dnmtil=sqrt(scaprd(nparam,d,d));
  if (infoqp!=0 || dnmtil>dnm) {
    for (i=1; i<=nparam; i++) d[i]=0.e0;
    dnmtil=0.e0;
    nstop=1;
    free_dv(adummy);
    return;
  }
  if (dnmtil!=0.e0)
    for (i=1; i<=nineqn+nob; i++)
      istore[i]=0;
  if (glob_prnt.iprint<3 || !display) {
    free_dv(adummy);
    return;
  }
  sbout1(glob_prnt.io,nparam,"dtilde              ",dummy,d,2,2);
  sbout1(glob_prnt.io,0,"dtnorm              ",dnmtil,adummy,1,2);
  free_dv(adummy);
  return;
}

/*******************************************************************/
/*     job=0:     compute d0                                       */
/*     job=1:     compute d~                                       */
/*******************************************************************/
#ifdef __STDC__
static void
dqp(int nparam,int nqpram,int nob,int nobL,int nfsip,int nineqn,
    int neq,int neqn,int nn,int ncsipl,int ncsipn,int ncnstr,
    int nctotl,int nrowa,int nineqn_tot,int *infoqp,
    struct _parameter *param,bioReal *di,int feasb,struct _objective *ob,
    bioReal fmax,bioReal *grdpsf,struct _constraint *cs,
    bioReal **a,bioReal *cvec,bioReal *bl,bioReal *bu,bioReal *clamda,
    bioReal **hess,bioReal **hess1,bioReal *x,
    bioReal vv,int job)
#else
  static void
dqp(nparam,nqpram,nob,nobL,nfsip,nineqn,neq,neqn,nn,ncsipl,ncsipn,
    ncnstr,nctotl,nrowa,nineqn_tot,infoqp,param,di,feasb,ob,
    fmax,grdpsf,cs,a,cvec,bl,bu,clamda,hess,hess1,x,vv,job)
  int     nparam,nqpram,nob,nobL,nfsip,nineqn,neq,neqn,nn,ncsipl,ncsipn,
  ncnstr,nctotl,nrowa,nineqn_tot,*infoqp,job,feasb;
bioReal  fmax,vv;
bioReal  *di,*grdpsf,**a,*cvec,*bl,*bu,*clamda,**hess,**hess1,*x;
struct  _constraint *cs;
struct  _objective  *ob;
struct  _parameter  *param;
#endif
{
  int i,ii,j,jj,ij,k,iout,mnn,nqnp,zero,temp1,temp2,ncnstr_used,
    numf_used;
  int *iw_hold;
  bioReal x0i,xdi,*bj,*htemp,*atemp;

  iout=6;
  bj=make_dv(nrowa);
  iw_hold=make_iv(nrowa);
  for (i=1; i<=nparam; i++) {
    x0i=param->x[i];
    if (job==1) xdi=di[i];
    if (job==0) xdi=0.e0;
    bl[i]=param->bl[i]-x0i-xdi;
    bu[i]=param->bu[i]-x0i-xdi;
    cvec[i]=cvec[i]-grdpsf[i];
  }
  if (nobL>1) {
    bl[nqpram]=-bgbnd;
    bu[nqpram]=bgbnd;
  }
  ii=ncnstr-nn;
  /*---------------------------------------------------------------*/
  /*     constraints are assigned to a in reverse order            */
  /*---------------------------------------------------------------*/
  k=0;
  for (i=1; i<=ncnstr; i++) {
    jj=iw[ncnstr+1-i];
    if ((jj>glob_info.nnineq)||(jj<=nineqn_tot-ncsipn)||
	((jj>nineqn_tot)&&(jj<=glob_info.nnineq-ncsipl))||
	cs[jj].act_sip) {
      k++;
      x0i=vv;
      if (i<=(neq-neqn) || (i>neq && i<=(ncnstr-nineqn)))
	x0i=0.e0;
      if (!feasb) x0i=0.e0;
      bj[k]=x0i-cs[jj].val;
      for (j=1; j<=nparam; j++)
	a[k][j]=-cs[jj].grad[j];
      if (nobL>1) a[k][nqpram]=0.e0;
      iw_hold[k]=jj;
    }
  }
  ncnstr_used=k;
  /*---------------------------------------------------------------*/
  /* Assign objectives for QP                                      */
  /*---------------------------------------------------------------*/
  if (nobL==1) {
    for (i=1; i<=nparam; i++)
      cvec[i]=cvec[i]+ob[1].grad[i];
  } else if (nobL>1) {
    numf_used=nob-nfsip+glob_info.tot_actf_sip;
    if (job&&nfsip) {     /* compute # objectives used for dtilde */
      numf_used=0;
      for (i=1; i<=nob; i++)
	if (ob[iw[ncnstr+i]].act_sip) numf_used++;
    }
    for (i=1; i<=nob; i++) {
      ij=ncnstr+i;
      if ((i<=nob-nfsip) || ob[iw[ij]].act_sip) {
	k++;
	iw_hold[k]=iw[ij];  /* record which are used */
	bj[k]=fmax-ob[iw[ij]].val;
	if (nobL>nob) bj[k+numf_used]=fmax+ob[iw[ij]].val;
	for (j=1; j<=nparam; j++) {
	  a[k][j]=-ob[iw[ij]].grad[j];
	  if (nobL>nob) a[k+numf_used][j]=ob[iw[ij]].grad[j];
	}
	a[k][nqpram]=1.e0;
	if (nobL>nob) a[k+numf_used][nqpram]=1.e0;
      }
    }
    cvec[nqpram]=1.e0;
    if (nobL>nob) k=k+numf_used;  /* k=# rows for a         */
  }                                /*  =# constraints for QP */
  matrcp(nparam,hess,nparam+1,hess1);
  nullvc(nqpram,x);

  iw[1]=1;
  zero=0;
  temp1=neq-neqn;
  temp2=nparam+1;
  mnn=k+2*nqpram;
  htemp=convert(hess1,nparam+1,nparam+1);
  atemp=convert(a,nrowa,nqpram);

  ql0001_(&k,&temp1,&nrowa,&nqpram,&temp2,&mnn,(htemp+1),
	  (cvec+1),(atemp+1),(bj+1),(bl+1),(bu+1),(x+1),
	  (clamda+1),&iout,infoqp,&zero,(w+1),&lenw,(iw+1),&leniw,
	  &glob_grd.epsmac);

  free_dv(htemp);
  free_dv(atemp);
  if (*infoqp!=0 || job==1) {
    free_iv(iw_hold);
    free_dv(bj);
    return;
  }

  /*---------------------------------------------------------------*/
  /*  Save multipliers from the computation of d0                  */
  /*---------------------------------------------------------------*/
  nullvc(nqpram,param->mult);
  if (ncsipl+ncsipn)
    for (i=1; i<=ncnstr; i++)
      cs[i].mult=0.e0;
  if (nfsip)
    for (i=1; i<=nob; i++) {
      ob[i].mult=0.e0;
      ob[i].mult_L=0.e0;
    }
  for (i=1; i<=nqpram; i++) {
    ii=k+i;
    if (clamda[ii]==0.e0 && clamda[ii+nqpram]==0.e0) continue;
    else if (clamda[ii]!=0.e0) clamda[ii]=-clamda[ii];
    else clamda[ii]=clamda[ii+nqpram];
  }
  nqnp=nqpram+ncnstr;
  for (i=1; i<=nqpram; i++)                  /* Simple bounds */
    param->mult[i]=clamda[k+i];
  if (nctotl>nqnp) {                         /* Objectives    */
    for (i=1; i<=numf_used; i++) {
      ij=ncnstr_used+i;
      if (nobL!=nob) {
	ii=k-2*numf_used+i;
	ob[iw_hold[ij]].mult=clamda[ii]-clamda[ii+numf_used];
	ob[iw_hold[ij]].mult_L=clamda[ii+numf_used];
      } else {
	ii=k-numf_used+i;
	ob[iw_hold[ij]].mult=clamda[ii];
      }
    }
  }
  for (i=1; i<=ncnstr_used; i++)             /* Constraints   */
    cs[iw_hold[i]].mult=clamda[i];
  free_iv(iw_hold);
  free_dv(bj);
  return;
}

/****************************************************************/
/*    Computation of first order direction d1                   */
/****************************************************************/
#ifdef __STDC__
static void
di1(int nparam,int nqpram,int nob,int nobL,int nfsip,int nineqn,
    int neq,int neqn,int ncnstr,int ncsipl,int ncsipn,
    int nrowa,int *infoqp,int mode,struct _parameter *param,
    bioReal *d0,struct _objective *ob,bioReal fmax,bioReal
    *grdpsf,struct _constraint *cs,bioReal *cvec,bioReal *bl,bioReal *bu,
    bioReal *clamda,bioReal **hess1,bioReal *x,bioReal steps)
#else
  static void
di1(nparam,nqpram,nob,nobL,nfsip,nineqn,neq,neqn,ncnstr,ncsipl,
    ncsipn,nrowa,infoqp,mode,param,d0,ob,fmax,grdpsf,cs,
    cvec,bl,bu,clamda,hess1,x,steps)
  int     nparam,nqpram,nob,nobL,nfsip,nineqn,neq,neqn,ncnstr,
  nrowa,*infoqp,mode,ncsipl,ncsipn;
bioReal  fmax,steps;
bioReal  *d0,*grdpsf,*cvec,*bl,*bu,*clamda,**hess1,*x;
struct _constraint *cs;
struct _objective  *ob;
struct _parameter  *param;
#endif
{
  int i,k,ii,jj,iout,j,mnn,zero,temp1,temp3,ncnstr_used,numf_used;
  int *iw_hold;
  bioReal x0i,eta,*atemp,*htemp,**a,*bj;

  if ((ncsipl+ncsipn)!=0)
    nrowa=nrowa-(ncsipl+ncsipn)+glob_info.tot_actg_sip;
  if (nfsip) {
    if (nobL>nob) nrowa=nrowa-2*nfsip+2*glob_info.tot_actf_sip;
    else nrowa=nrowa-nfsip+glob_info.tot_actf_sip;
  }
  nrowa=DMAX1(nrowa,1);
  a=make_dm(nrowa,nqpram);
  bj=make_dv(nrowa);
  iw_hold=make_iv(nrowa);
  iout=6;
  if (mode==0) eta=0.1e0;
  if (mode==1) eta=3.e0;
  for (i=1; i<=nparam; i++) {
    x0i=param->x[i];
    bl[i]=param->bl[i]-x0i;
    bu[i]=param->bu[i]-x0i;
    if (mode==0) cvec[i]=-eta*d0[i];
    if (mode==1) cvec[i]=0.e0;
  }
  bl[nqpram]=-bgbnd;
  bu[nqpram]=bgbnd;
  cvec[nqpram]=1.e0;
  ii=ncnstr-nineqn;
  k=0;
  for (i=1; i<=ncnstr; i++) {
    jj=ncnstr+1-i;
    if ((jj>glob_info.nnineq)||(jj<=nineqn-ncsipn)||
	((jj>nineqn)&&(jj<=glob_info.nnineq-ncsipl))||
	cs[jj].act_sip) {
      k++;
      bj[k]=-cs[jj].val;
      for (j=1; j<=nparam; j++)
	a[k][j]=-cs[jj].grad[j];
      a[k][nqpram]=0.e0;
      if ((i>(neq-neqn) && i<=neq) || i>ii) a[k][nqpram]=1.e0;
      iw_hold[k]=jj;
    }
  }
  ncnstr_used=k;

  if (mode!=1) {
    numf_used=nob-nfsip+glob_info.tot_actf_sip;
    for (i=1; i<=nob; i++) {
      if ((i<=nob-nfsip) || ob[i].act_sip) {
	k++;
	bj[k]=fmax-ob[i].val;
	for (j=1; j<=nparam; j++) {
	  a[k][j]=-ob[i].grad[j]+grdpsf[j];
	  if (nobL>nob) a[k+numf_used][j]=ob[i].grad[j]+grdpsf[j];
	}
	a[k][nqpram]=1.e0;
	if (nobL>nob) a[k+numf_used][nqpram]=1.e0;
      }
    }
    if (nob==0) {
      k++;
      bj[k]=fmax;
      for (j=1; j<=nparam; j++)
	a[k][j]=grdpsf[j];
      a[k][nqpram]=1.e0;
    }
  }
  diagnl(nqpram,eta,hess1);
  nullvc(nqpram,x);
  hess1[nqpram][nqpram]=0.e0;

  iw[1]=1;
  zero=0;
  temp1=neq-neqn;
  if (nobL>nob) temp3=k+numf_used;
  else temp3=k;
  mnn=temp3+2*nqpram;
  htemp=convert(hess1,nparam+1,nparam+1);
  atemp=convert(a,nrowa,nqpram);

  ql0001_(&temp3,&temp1,&nrowa,&nqpram,&nqpram,&mnn,(htemp+1),
	  (cvec+1),(atemp+1),(bj+1),(bl+1),(bu+1),(x+1),
	  (clamda+1),&iout,infoqp,&zero,(w+1),&lenw,(iw+1),&leniw,
	  &glob_grd.epsmac);

  free_dv(htemp);
  free_dv(atemp);
  free_dm(a,nrowa);
  free_dv(bj);
  /* Determine binding constraints */
  if (ncsipl+ncsipn) {
    for (i=1; i<=ncnstr_used; i++)
      if (clamda[i]>0.e0) cs[iw_hold[i]].d1bind=TRUE;
  }
  free_iv(iw_hold);
  return;
}
/*****************************************************************/
/*     CFSQP : Armijo or nonmonotone line search, with some      */
/*             ad hoc strategies to decrease the number of       */
/*             function evaluations as much as possible.         */
/*****************************************************************/

#ifdef __STDC__
static void
step1(int nparam,int nob,int nobL,int nfsip,int nineqn,int neq,int neqn,
      int nn,int ncsipl,int ncsipn,int ncnstr,int *ncg,int *ncf,
      int *indxob,int *indxcn,int *iact,int *iskp,int *iskip,
      int *istore,int feasb,bioReal grdftd,struct _objective *ob,
      bioReal *fM,bioReal *fMp,bioReal *fmax,bioReal *psf,bioReal *penp,
      bioReal *steps,bioReal *scvneq,bioReal *xnew,bioReal *x,bioReal *di,
      bioReal *d,struct _constraint *cs,bioReal *backup,bioReal *signeq,
      struct _violation *sip_viol,
      void (*obj)(int,int,bioReal *,bioReal *,void *),
      void (*constr)(int,int,bioReal *,bioReal *,void *),void *cd)
#else
  static void
step1(nparam,nob,nobL,nfsip,nineqn,neq,neqn,nn,ncsipl,ncsipn,ncnstr,
      ncg,ncf,indxob,indxcn,iact,iskp,iskip,istore,feasb,grdftd,ob,
      fM,fMp,fmax,psf,penp,steps,scvneq,xnew,x,di,d,cs,backup,
      signeq,sip_viol,obj,constr,cd)
  int     nparam,nob,nobL,nfsip,nineqn,neq,neqn,nn,ncsipl,ncsipn,ncnstr,
  *ncg,*ncf,feasb,*iskp;
int     *indxob,*indxcn,*iact,*iskip,*istore;
bioReal  grdftd,*fM,*fMp,*fmax,*steps,*scvneq,*psf;
bioReal  *xnew,*x,*di,*d,*penp,*backup,*signeq;
struct  _constraint *cs;
struct  _objective  *ob;
struct  _violation  *sip_viol;
void    (* obj)(),(* constr)();
void    *cd;
#endif
{
  int  i,ii,ij,jj,itry,ikeep,j,job,nlin,mnm,ltem1,ltem2,reform,
    fbind,cdone,fdone,eqdone,display,sipldone;
  bioReal prod1,prod,dummy,fmax1,tolfe,ostep,temp,**adummy,fii;

  nlin=glob_info.nnineq-nineqn;
  itry=ii=jj=1;
  ostep=*steps=1.e0;
  fbind=cdone=fdone=eqdone=FALSE;
  dummy=0.e0;
  sipldone=(ncsipl==0);
  if (glob_log.local) glob_log.dlfeas=FALSE;
  ikeep=nlin-*iskp;
  prod1=(0.1e0)*grdftd;        /* alpha = 0.1e0  */
  tolfe=0.e0;                  /* feasibility tolerance */
  adummy=make_dm(1,1);
  adummy[1][1]=0.e0;
  if (glob_prnt.iter%glob_prnt.iter_mod) display=FALSE;
  else display=TRUE;
  if (glob_prnt.iprint >= 3 && display)
    sbout1(glob_prnt.io,0,"directional deriv   ",grdftd,*(adummy+1),
	   1,2);
  w[1]=*fM;
  for (;;) {
    reform=TRUE;
    if (glob_prnt.iprint >= 3 && display)
      DETAILED_MESSAGE("\t\t\t trial number            " << std::setw(22) << itry);
    prod=prod1*(*steps);
    if (!feasb || (nobL > 1)) prod=prod+tolfe;
    for (i=1; i<=nparam; i++) {
      if (glob_log.local) xnew[i]=x[i]+(*steps)*di[i];
      else xnew[i]=x[i]+(*steps)*di[i]+d[i]*(*steps)*(*steps);
    }
    x_is_new=TRUE;
    if (glob_prnt.iprint >= 3 && display) {
      sbout1(glob_prnt.io,0,"trial step          ",*steps,
	     *(adummy+1),1,2);
      sbout1(glob_prnt.io,nparam,"trial point         ",
	     dummy,xnew,2,2);
    }

    /* Generate an upper bound step size using the linear constraints
       not used in the computation of dtilde */
    if (*iskp != 0) {
      ostep=*steps;
      for (i=ii; i<=*iskp; i++) {
	ij=iskip[i];
	constr(nparam,ij,xnew+1,&(cs[ij].val),cd);
	if (glob_prnt.iprint >= 3 && display) {
	  if (i==1) DETAILED_MESSAGE("\t\t\t trial constraints  "
				     <<ij
				     << " \t " 
				     << std::setw(22) 
				     << std::setprecision(14) 
				     << std::setiosflags(std::ios::scientific|std::ios::showpos) 
				     << cs[ij].val);
	  if (i!=1) DETAILED_MESSAGE("\t\t\t\t\t "
				     <<ij
				     <<" \t "
				     << std::setw(22) 
				     << std::setprecision(14) 
				     << std::setiosflags(std::ios::scientific|std::ios::showpos) 
				     << cs[ij].val);
	}
	if (cs[ij].val<=tolfe) continue;
	ii=i;
	if (ncsipl && ii>glob_info.nnineq-ncsipl) {
	  sip_viol->type = CONSTR;
	  sip_viol->index = ij;
	} else {
	  sip_viol->type = NONE; /* non-SIP constraint violated */
	  sip_viol->index = 0;
	}
	goto L1120;
      }
      *iskp=0;
    }

    /* Refine the upper bound using the linear SI constraints not
       in Omega_k */
    if (!sipldone) {
      for (i=jj; i<=ncsipl; i++) {
	ij=glob_info.nnineq-ncsipl+i;
	if (cs[ij].act_sip||element(iskip,nlin-ikeep,ij))
	  continue;
	constr(nparam,ij,xnew+1,&(cs[ij].val),cd);
	if (glob_prnt.iprint >= 3 && display) {
	  if (i==1) DETAILED_MESSAGE("\t\t\t trial constraints  "
				     <<ij
				     <<" \t " 
				     << std::setw(22) 
				     << std::setprecision(14) 
				     << std::setiosflags(std::ios::scientific|std::ios::showpos) 
				     << cs[ij].val);
	  if (i!=1)
	    DETAILED_MESSAGE("\t\t\t\t\t "
			     << ij
			     <<" \t " 
			     << std::setw(22) 
			     << std::setprecision(14) 
			     << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			     << cs[ij].val);
	}
	if (cs[ij].val<=tolfe) continue;
	jj=i;
	sip_viol->type=CONSTR;
	sip_viol->index=ij;
	goto L1120;
      }
      sipldone=TRUE;
    }
    if (nn==0) goto L310;

    /* Check nonlinear constraints                            */
    if (!glob_log.local && fbind) goto L315;
    do {
      for (i=1; i<=nn; i++) {
	*ncg=i;
	ii=iact[i];
	ij=glob_info.nnineq+neqn;
	if (!((ii<=nineqn && istore[ii]==1)||
	      (ii>glob_info.nnineq && ii<=ij && eqdone))) {
	  temp=1.e0;
	  if(ii>glob_info.nnineq && ii<=ij)
	    temp=signeq[ii-glob_info.nnineq];
	  constr(nparam,ii,xnew+1,&(cs[ii].val),cd);
	  cs[ii].val *= temp;
	  glob_info.ncallg++;
	}
	if (glob_prnt.iprint>=3 && display) {
	  if (i==1 && ikeep==nlin) 
	    DETAILED_MESSAGE("\t\t\t trial constraints  "
			     << ii
			     <<" \t " 
			     << std::setw(22) 
			     << std::setprecision(14) 
			     << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			     << cs[ii].val);
	  if (i!=1 || ikeep!=nlin) 
	    DETAILED_MESSAGE("\t\t\t\t\t "<< ii << " \t " 
			     << std::setw(22) 
			     << std::setprecision(14) 
			     << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			     << cs[ii].val);
	}
	if (!(glob_log.local || cs[ii].val<=tolfe)) {
	  shift(nn,ii,iact);
	  if (ncsipn && ii>nineqn-ncsipn) {
	    sip_viol->type=CONSTR;
	    sip_viol->index=ii;
	  } else {
	    sip_viol->type=NONE; /* non-SIP constraint violated */
	    sip_viol->index=0;
	  }
	  goto L1110;
	}
	if (glob_log.local && cs[ii].val>tolfe) {
	  if (ncsipn && ii>nineqn-ncsipn) {
	    sip_viol->type=CONSTR;
	    sip_viol->index=ii;
	  } else {
	    sip_viol->type=NONE; /* non-SIP constraint violated */
	    sip_viol->index=0;
	  }
	  goto L1500;
	}
      }
    L310:    cdone=eqdone=TRUE;
      if (glob_log.local) glob_log.dlfeas=TRUE; /* dl is feasible */
    L315:    if (fdone) break;
      if (nob>0) fmax1=-bgbnd;
      else fmax1=0.e0;
      for (i=0; i<=nob; i++) {
	if (nob!=0 && i==0) continue;
	*ncf=i;
	ii=iact[nn+i];
	if (feasb) {
	  if (!(eqdone || neqn==0)) {
	    for (j=1; j<=neqn; j++)
	      constr(nparam,glob_info.nnineq+j,xnew+1,
		     &(cs[glob_info.nnineq+j].val),cd);
	    glob_info.ncallg+=neqn;
	  }
	  if (neqn != 0) {
	    if (eqdone)    job=20;
	    if (!eqdone)   job=10;
	    resign(nparam,neqn,psf,*(adummy+1),penp,cs,signeq,
		   job,10);
	  }
	  if (istore[nineqn+ii]!=1 && i!=0) {
	    obj(nparam,ii,xnew+1,&(ob[ii].val),cd);
	    glob_info.ncallf++;
	  }
	  if (i==0) fii=0.e0;
	  else fii=ob[ii].val;
	  if (i==0 && glob_prnt.iprint>=3 && display)
	    DETAILED_MESSAGE("\t\t\t trial penalty term \t " 
			     << std::setw(22) 
			     << std::setprecision(14) 
			     << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			     << -*psf);
	  if (i==1 && glob_prnt.iprint>=3 && display)
	    DETAILED_MESSAGE("\t\t\t trial objectives   "
			     << ii 
			     << " \t " 
			     << std::setw(22) 
			     << std::setprecision(14) 
			     << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			     << fii-*psf);
	  if (i>1 && glob_prnt.iprint>=3 && display)
	    DETAILED_MESSAGE("\t\t\t\t\t "
			     << ii
			     <<" \t " 
			     << std::setw(22) 
			     << std::setprecision(14) 
			     << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			     << fii-*psf);
	} else {
	  if (istore[ii]!=1) {
	    constr(nparam,indxob[ii],xnew+1,&(ob[ii].val),cd);
	    glob_info.ncallg++;
	  }
	  if (ob[ii].val>tolfe) reform=FALSE;
	  if (i==1 && glob_prnt.iprint>2 && display)
	    DETAILED_MESSAGE("\t\t\t trial objectives   " 
			     << indxob[ii] 
			     << " \t " 
			     << std::setw(22) << std::setprecision(14) 
			     << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			     << ob[ii].val);
	  if (i!=1 && glob_prnt.iprint>2 && display)
	    DETAILED_MESSAGE("\t\t\t\t\t "
			     << indxob[ii]
			     << " \t " 
			     << std::setw(22) 
			     << std::setprecision(14) 
			     << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			     << ob[ii].val);
	  fii=ob[ii].val;
	}
	fmax1=DMAX1(fmax1,fii);
	if (nobL!=nob) fmax1=DMAX1(fmax1,-fii);
	if (!feasb && reform) continue;
	if (!glob_log.local) {
	  if ((fii-*psf)>(*fMp+prod)) {
	    fbind=TRUE;
	    shift(nob,ii,&iact[nn]);
	    if (nfsip && ii>nob-nfsip) {
	      sip_viol->type=OBJECT;
	      sip_viol->index=ii;
	    } else {
	      sip_viol->type=NONE;
	      sip_viol->index=0;
	    }
	    goto L1110;
	  }
	  if (nobL==nob || (-fii-*psf)<=(*fMp+prod))
	    continue;
	  fbind=TRUE;
	  shift(nob,ii,&iact[nn]);
	  if (nfsip && ii>nob-nfsip) {
	    sip_viol->type=OBJECT;
	    sip_viol->index=ii;
	  } else {
	    sip_viol->type=NONE;
	    sip_viol->index=0;
	  }
	  goto L1110;
	}
	ltem1=(fii-*psf)>(*fMp+prod);
	ltem2=(nobL!=nob)&&((-fii-*psf)>(*fMp+prod));
	if (ltem1 || ltem2) goto L1500;
      }
      fbind=FALSE;
      fdone=eqdone=TRUE;
    } while (!cdone);
    if (ostep==*steps) mnm=ikeep+neq-neqn;
    if (ostep!=*steps) mnm=ncnstr-nn;
    for (i=1; i<=mnm; i++) {
      ii=indxcn[i+nn];
      if (ikeep!=nlin && ostep==*steps) {
	if (i<= ikeep) ii=iskip[nlin+2-i];
	else ii=indxcn[nn+i-ikeep+nlin];
      }
      constr(nparam,ii,xnew+1,&(cs[ii].val),cd);
    }
    *scvneq=0.e0;
    for (i=1; i<=ncnstr; i++) {
      if (i>glob_info.nnineq && i<=(glob_info.nnineq+neqn))
	*scvneq=*scvneq-cs[i].val;
      backup[i]=cs[i].val;
    }
    for (i=1; i<=nob; i++)
      backup[i+ncnstr]=ob[i].val;
    if (!feasb && reform) {
      for (i=1; i<=nparam; i++)
	x[i]=xnew[i];
      nstop=0;
      goto L1500;
    }
    if (glob_log.local) *ncg=ncnstr;
    if (glob_log.local) glob_log.update=TRUE;
    *fM=fmax1;
    *fMp=fmax1-*psf;
    *fmax=fmax1;
    for (i=1; i<=nn; i++)
      iact[i]=indxcn[i];
    for (i=1; i<=nob; i++)
      iact[nn+i]=i;
    goto L1500;
  L1110:
    cdone=fdone=eqdone=reform=FALSE;
  L1120:
    itry++;
    if (glob_info.modec==2) fbind=FALSE;
    if (*steps >= 1.e0)
      for (i=1; i<=nob+nineqn; i++)
	istore[i]=0;
    *steps=*steps*0.5e0;
    if (*steps<glob_grd.epsmac) break;
  }
  glob_prnt.info=4;
  nstop=0;
 L1500:
  free_dm(adummy,1);
  if (*steps<1.e0) return;
  for (i=1; i<=nob+nineqn; i++)
    istore[i]=0;
  return;
}
/******************************************************************/
/*   CFSQP : Update the Hessian matrix using BFGS formula with    */
/*           Powell's modification.                               */
/******************************************************************/

#ifdef __STDC__
static void
hessian(int nparam,int nob,int nfsip,int nobL,int nineqn,int neq,
	int neqn,int nn,int ncsipn,int ncnstr,int nfs,int *nstart,
	int feasb,bioReal *xnew,struct _parameter *param,
	struct _objective *ob,bioReal fmax,bioReal *fM,bioReal *fMp,
	bioReal *psf,bioReal *grdpsf,bioReal *penp,struct _constraint *cs,
	bioReal *gm,int *indxob,int *indxcn,bioReal *delta,bioReal *eta,
	bioReal *gamma,bioReal **hess,bioReal *hd,bioReal steps,int *nrst,
	bioReal *signeq,bioReal *span,
	void (*obj)(int,int,bioReal *,bioReal *,void *),
	void (*constr)(int,int,bioReal *,bioReal *,void *),
	void (*gradob)(int,int,bioReal *,bioReal *,
		       void (*)(int,int,bioReal *,bioReal *,void *),void *),
	void (*gradcn)(int,int,bioReal *,bioReal *,
		       void (*)(int,int,bioReal *,bioReal *,void *),void *),
	bioReal **phess,bioReal *psb,bioReal *psmu,
	struct _violation *sip_viol)
#else
  static void
hessian(nparam,nob,nfsip,nobL,nineqn,neq,neqn,nn,ncsipn,ncnstr,
        nfs,nstart,feasb,xnew,param,ob,fmax,fM,fMp,psf,grdpsf,penp,
        cs,gm,indxob,indxcn,delta,eta,gamma,hess,hd,steps,nrst,signeq,
        span,obj,constr,gradob,gradcn,phess,psb,psmu,sip_viol)
  int     nparam,nob,nobL,nineqn,neq,neqn,nn,nfsip,ncsipn,ncnstr,
  nfs,*nstart,feasb,*nrst;
int     *indxob,*indxcn;
bioReal  steps,*psf,fmax,*fM,*fMp;
bioReal  *xnew,*grdpsf,*penp,*gm,*delta,*eta,*gamma,
  **hess,*hd,*signeq,*span,**phess,*psb,*psmu;
struct  _constraint *cs;
struct  _objective  *ob;
struct  _parameter  *param;
struct  _violation  *sip_viol;
void    (* obj)(), (* constr)(),(* gradob)(), (* gradcn)();
#endif
{
  int    i,j,k,ifail,np,mnm,done,display;
  bioReal dhd,gammd,etad,dummy,theta,signgj,psfnew,delta_s;
  bioReal *tempv;

  /* Check to see whether user-accessible stopping criterion
     is satisfied. The check of gLgeps is made just after
     computing d0 */

  if (!glob_log.get_ne_mult) {
    if (feasb && nstop && !neqn)
      if ((fabs(w[1]-fmax) <= objeps) ||
	  (fabs(w[1]-fmax) <= objrep*fabs(w[1]))) nstop=0;
    if (!nstop) {
      for (i=1; i<= nparam; i++)
	param->x[i]=xnew[i];
      x_is_new=TRUE;
      return;
    }
  }

  delta_s=glob_grd.rteps;  /* SIP */
  if (glob_prnt.iter%glob_prnt.iter_mod) display=FALSE;
  else display=TRUE;
  psfnew=0.e0;
  glob_prnt.ipd=0;
  done=FALSE;
  dummy=0.e0;
  nullvc(nparam,delta);
  nullvc(nparam,eta);
  for (j=1; j<=2; j++) {
    nullvc(nparam, gamma);
    if (nobL>1) {
      for (i=1; i<=nparam; i++) {
	hd[i]=0.e0;
	for (k=1; k<=nob; k++)
	  hd[i]=hd[i]+ob[k].grad[i]*ob[k].mult;
      }
    }
    if (feasb) {
      if (nineqn != 0) {
	for (i=1; i<=nparam; i++) {
	  gamma[i]=0.e0;
	  for (k=1; k<=nineqn; k++)
	    gamma[i]=gamma[i]+cs[k].grad[i]*cs[k].mult;
	}
      }
      if (neqn != 0) {
	for (i=1; i<=nparam; i++) {
	  eta[i]=0.e0;
	  for (k=1; k<=neqn; k++)
	    eta[i]=eta[i]+cs[glob_info.nnineq+k].grad[i]*
	      cs[glob_info.nnineq+k].mult;
	}
      }
    }
    for (i=1; i<=nparam; i++) {
      if (nobL>1) {
	if (done) psb[i]=hd[i]+param->mult[i]+gamma[i];
	gamma[i]=gamma[i]+hd[i]-grdpsf[i]+eta[i];
      } else if (nobL==1) {
	if (done) psb[i]=ob[1].grad[i]+param->mult[i]+gamma[i];
	gamma[i]=gamma[i]+ob[1].grad[i]-grdpsf[i]+eta[i];
      } else if (nobL==0) {
	if (done) psb[i]=param->mult[i]+gamma[i];
	gamma[i]=gamma[i]-grdpsf[i]+eta[i];
      }
      if (!done) delta[i]=gamma[i];
    }
    if (!done && !glob_log.d0_is0) {
      if (nn != 0) {
	for (i=1; i<=nn; i++) {
	  if ((feasb) && (i>nineqn)) signgj=signeq[i-nineqn];
	  if ((!feasb) || (i<=nineqn)) signgj=1.e0;
	  if ((feasb) && (ncsipn) && (i>nineqn-ncsipn) &&
	      (cs[indxcn[i]].mult==0.e0)) continue;
	  glob_grd.valnom=cs[indxcn[i]].val*signgj;
	  gradcn(nparam,indxcn[i],xnew+1,cs[indxcn[i]].grad+1,
		 constr,param->cd);
	}
	resign(nparam,neqn,psf,grdpsf,penp,cs,signeq,11,11);
      }
      for (i=1; i<=nob; i++) {
	glob_grd.valnom=ob[i].val;
	if ((i<=nob-nfsip)||(i>nob-nfsip &&
			     ((ob[i].mult !=0.e0)||(ob[i].mult_L !=0.e0)))) {
	  if (feasb)
	    gradob(nparam,i,xnew+1,ob[i].grad+1,obj,param->cd);
	  else gradcn(nparam,indxob[i],xnew+1,ob[i].grad+1,
		      constr,param->cd);
	}
      }
      done=TRUE;
    }
    if (glob_log.d0_is0) done=TRUE;
  }
  if (!glob_log.d0_is0) {
    if (!(feasb && steps<delta_s && ((sip_viol->type==OBJECT &&
				      !ob[sip_viol->index].act_sip)||(sip_viol->type==CONSTR &&
								      !cs[sip_viol->index].act_sip)))) {
      if (*nrst<(5*nparam) || steps>0.1e0) {
	(*nrst)++;
	for (i=1; i<=nparam; i++) {
	  gamma[i]=gamma[i]-delta[i];
	  delta[i]=xnew[i]-param->x[i];
	}
	matrvc(nparam,nparam,hess,delta,hd);
	dhd=scaprd(nparam,delta,hd);
	if (sqrt(scaprd(nparam,delta,delta)) <= glob_grd.epsmac) {
	  /* xnew too close to x!! */
	  nstop=0;
	  glob_prnt.info=8;
	  return;
	}
	gammd=scaprd(nparam,delta,gamma);
	if (gammd >= (0.2e0*dhd)) theta=1.e0;
	else theta=0.8e0*dhd/(dhd-gammd);
	for (i=1; i<=nparam; i++)
	  eta[i]=hd[i]*(1.e0-theta)+theta*gamma[i];
	etad=theta*gammd+(1.e0-theta)*dhd;
	for (i=1; i<=nparam; i++) {
	  for (j=i; j<=nparam; j++) {
	    hess[i][j]=hess[i][j] - hd[i]*hd[j]/dhd +
	      eta[i]*eta[j]/etad;
	    hess[j][i]=hess[i][j];
	  }
	}
      } else {
	*nrst=0;
	diagnl(nparam,1.e0,hess);
      }
    }
    for (i=1; i<=nparam; i++)
      param->x[i]=xnew[i];
    x_is_new=TRUE;
  }
  if (neqn!=0 && (feasb)) {
    i=glob_info.nnineq-nineqn;
    if (i!=0) {
      for (j=1; j<=nparam; j++) {
	gamma[j]=0.e0;
	for (k=1; k<=i; k++)
	  gamma[j]=gamma[j]+cs[k+nineqn].grad[j]*
	    cs[nineqn+k].mult;
      }
      for (i=1; i<=nparam; i++)
	psb[i]=psb[i]+gamma[i];
    }
    i=neq-neqn;
    if (i!=0) {
      for (j=1; j<=nparam; j++) {
	gamma[j]=0.e0;
	for (k=1; k<=i; k++)
	  gamma[j]=gamma[j]+cs[k+neqn+glob_info.nnineq].grad[j]*
	    cs[glob_info.nnineq+neqn+k].mult;
      }
      for (i=1; i<=nparam; i++)
	psb[i]=psb[i]+gamma[i];
    }
    /* Update penalty parameters for nonlinear equality constraints */
    estlam(nparam,neqn,&ifail,bgbnd,phess,delta,eta,
	   gamma,cs,psb,hd,xnew,psmu);
    if (glob_log.get_ne_mult) return;
    for (i=1; i<=neqn; i++) {
      if (ifail != 0 || glob_log.d0_is0) penp[i]=2.e0*penp[i];
      else {
	etad=psmu[i]+penp[i];
	if (etad < 1.e0)
	  penp[i]=DMAX1((1.e0-psmu[i]),(2.e0*penp[i]));
      }
      if (penp[i] > bgbnd) {
	nstop=0;
	glob_prnt.info=9;
	return;
      }
    }
    resign(nparam,neqn,psf,grdpsf,penp,cs,signeq,20,12);
    *fMp=*fM-*psf;
  }
  if (nfs != 0) {
    (*nstart)++;
    np=indexs(*nstart,nfs);
    span[np]=fmax;
    for (i=1; i<=neqn; i++)
      gm[(np-1)*neqn+i]=cs[glob_info.nnineq+i].val;
    if (neqn != 0) {
      psfnew=0.e0;
      for (i=1; i<=neqn; i++)
	psfnew=psfnew+gm[i]*penp[i];
    }
    *fM=span[1];
    *fMp=span[1]-psfnew;
    mnm=DMIN1(*nstart,nfs);
    for (i=2; i<=mnm; i++) {
      if (neqn != 0) {
	psfnew=0.e0;
	for (j=1; j<=neqn; j++)
	  psfnew=psfnew+gm[(i-1)*neqn +j]*penp[j];
      }
      *fM=DMAX1(*fM, span[i]);
      *fMp=DMAX1(*fMp,span[i]-psfnew);
    }
  }
  if (glob_prnt.iprint < 3 || !display) return;
  for (i=1; i<=nob; i++) {
    if (!feasb) {
      sbout2(glob_prnt.io,nparam,indxob[i],"gradg(j,",
	     ")",ob[i].grad);
      continue;
    }
    if (nob>1) sbout2(glob_prnt.io,nparam,i,"gradf(j,",")",
		      ob[i].grad);
    if (nob==1) sbout1(glob_prnt.io,nparam,"gradf(j)            ",
		       dummy,ob[1].grad,2,2);
  }
  if (ncnstr != 0) {
    for (i=1; i<=ncnstr; i++) {
      tempv=cs[indxcn[i]].grad;
      sbout2(glob_prnt.io,nparam,indxcn[i],"gradg(j,",")",tempv);
    }
    if (neqn != 0) {
      sbout1(glob_prnt.io,nparam,"grdpsf(j)           ",
	     dummy,grdpsf,2,2);
      sbout1(glob_prnt.io,neqn,"P                   ",dummy,
	     penp,2,2);
      sbout1(glob_prnt.io,neqn,"psmu                ",dummy,
	     psmu,2,2);
    }
  }
  sbout1(glob_prnt.io,nparam,"multipliers for x   ",dummy,
	 param->mult,2,2);
  if (ncnstr != 0) {
    DETAILED_MESSAGE("\t\t\t             for g   \t "
		     << std::setw(22) 
		     << std::setprecision(14) 
		     << std::setiosflags(std::ios::scientific|std::ios::showpos) 
		     << cs[1].mult);
    for (j=2; j<=ncnstr; j++)
      DETAILED_MESSAGE(" \t\t\t\t\t\t " << std::setw(22) << std::setprecision(14) 
		       << std::setiosflags(std::ios::scientific|std::ios::showpos) 
		       << cs[j].mult);
  }
  if (nobL > 1) {
    DETAILED_MESSAGE("\t\t\t             for f   \t " 
		     << std::setw(22) << std::setprecision(14) 
		     << std::setiosflags(std::ios::scientific|std::ios::showpos) 
		     << ob[1].mult);
    for (j=2; j<=nob; j++)
      DETAILED_MESSAGE(" \t\t\t\t\t\t " << std::setw(22) << std::setprecision(14) 
		       << std::setiosflags(std::ios::scientific|std::ios::showpos) 
		       << ob[j].mult);
  }
  for (i=1; i<=nparam; i++) {
    tempv=colvec(hess,i,nparam);
    sbout2(glob_prnt.io,nparam,i,"hess (j,",")",tempv);
    free_dv(tempv);
  }
  return;
}
/**************************************************************/
/*   CFSQP : Output                                           */
/**************************************************************/

#ifdef __STDC__
static void
out(int miter,int nparam,int nob,int nobL,int nfsip,int ncn,
    int nn,int nineqn,int ncnstr,int ncsipl,int ncsipn,
    int *mesh_pts,bioReal *x,struct _constraint *cs,
    struct _objective *ob,bioReal fM,bioReal fmax,
    bioReal steps,bioReal sktnom,bioReal d0norm,int feasb)
#else
  static void
out(miter,nparam,nob,nobL,nfsip,ncn,nn,nineqn,ncnstr,ncsipl,ncsipn,
    mesh_pts,x,cs,ob,fM,fmax,steps,sktnom,d0norm,feasb)
  int     miter,nparam,nob,nobL,nfsip,ncn,nn,ncnstr,feasb,
  ncsipl,ncsipn,nineqn,*mesh_pts;
bioReal  fM,fmax,steps,sktnom,d0norm;
bioReal  *x;
struct  _constraint *cs;
struct  _objective  *ob;
#endif
{
  int i,j,index,display,offset;
  bioReal SNECV,dummy,*adummy,gmax;

  adummy=make_dv(1);
  adummy[1]=0.e0;
  dummy=0.e0;

  // patModelSpec::the()->saveBackup() ;
  // if (patFileExists()(patParameters::the()->getgevStopFileName())) {
  //   WARNING("Iterations interrupted by the user with the file " 
  //   	    << patParameters::the()->getgevStopFileName()) ;
  //   glob_prnt.info=3;
  //   nstop=0;
  //   if (glob_prnt.iprint==0) goto L9000;
  // }

  if (glob_prnt.iter>=miter && nstop != 0) {
    glob_prnt.info=3;
    nstop=0;
    if (glob_prnt.iprint==0) goto L9000;
  }
  if (glob_prnt.iprint==0 && glob_prnt.iter<miter) {
    glob_prnt.iter++;
    goto L9000;
  }
  if ((glob_prnt.info>0 && glob_prnt.info<3) || glob_prnt.info==7)
    goto L120;
  if (glob_prnt.iprint==1 && nstop!=0) {
    glob_prnt.iter++;
    if (glob_prnt.initvl==0) goto L9000;
    if (feasb && nob>0) {
      DETAILED_MESSAGE(" objectives");
      for (i=1; i<=nob-nfsip; i++) {
	if (nob==nobL) {
	  GENERAL_MESSAGE(" \t\t\t " << std::setw(22) << std::setprecision(14) 
			  << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			  << ob[i].val);
	}
	else {
	  GENERAL_MESSAGE(" \t\t\t " << std::setw(22) << std::setprecision(14) 
			  << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			  << fabs(ob[i].val));
	}
      }
      if (nfsip) {
	offset=nob-nfsip;
	for (i=1; i<=glob_info.nfsip; i++) {
	  if (nob==nobL) gmax=ob[++offset].val;
	  else gmax=fabs(ob[++offset].val);
	  for (j=2; j<=mesh_pts[i]; j++) {
	    offset++;
	    if (nob==nobL && ob[offset].val>gmax)
	      gmax=ob[offset].val;
	    else if (nob!=nobL && fabs(ob[offset].val)>gmax)
	      gmax=fabs(ob[offset].val);
	  }
	  DETAILED_MESSAGE(" \t\t\t " << std::setw(22) << std::setprecision(14) 
			   << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			   << gmax);
	}
      }
    }
    if (glob_info.mode==1 && glob_prnt.iter>1 && feasb)
      sbout1(glob_prnt.io,0,"objective max4      ",fM,adummy,1,1);
    if (nob>1)
      sbout1(glob_prnt.io,0,"objmax              ",fmax,adummy,1,1);
    if (ncnstr==0) {
      DETAILED_MESSAGE("");
    }
    else {
      DETAILED_MESSAGE(" constraints");
      for (i=1; i<=nineqn-ncsipn; i++)
	DETAILED_MESSAGE(" \t\t\t " << std::setw(22) << std::setprecision(14) 
			 << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			 << cs[i].val);
      if (ncsipn) {
	offset=nineqn-ncsipn;
	for (i=1; i<=glob_info.ncsipn; i++) {
	  gmax=cs[++offset].val;
	  for (j=2; j<=mesh_pts[glob_info.nfsip+i]; j++) {
	    offset++;
	    if (cs[offset].val>gmax) gmax=cs[offset].val;
	  }
	  DETAILED_MESSAGE(" \t\t\t " << std::setw(22) << std::setprecision(14) 
			   << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			   << gmax);
	}
      }
      for (i=nineqn+1; i<=glob_info.nnineq-ncsipl; i++)
	DETAILED_MESSAGE(" \t\t\t " << std::setw(22) << std::setprecision(14) 
			 << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			 << cs[i].val);
      if (ncsipl) {
	offset=glob_info.nnineq-ncsipl;
	for (i=1; i<=glob_info.ncsipl; i++) {
	  gmax=cs[++offset].val;
	  if (feasb) index=glob_info.nfsip+glob_info.ncsipn+i;
	  else index=glob_info.ncsipn+i;
	  for (j=2; j<=mesh_pts[index]; j++) {
	    offset++;
	    if (cs[offset].val>gmax) gmax=cs[offset].val;
	  }
	  DETAILED_MESSAGE(" \t\t\t " << std::setw(22) << std::setw(22) << std::setprecision(14) 
			   << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			   << std::setprecision(14) 
			   << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			   << gmax);
	}
      }
      for (i=glob_info.nnineq+1; i<=ncnstr; i++)
	DETAILED_MESSAGE(" \t\t\t " << cs[i].val);
    }
    if (ncnstr!=0) DETAILED_MESSAGE("");
    goto L9000;
  }
  if (glob_prnt.iprint==1 && nstop==0)
    DETAILED_MESSAGE(" iteration           " << std::setw(26) << glob_prnt.iter)
  if (glob_prnt.iprint<=2 && nstop==0)
    DETAILED_MESSAGE(" inform              " << std::setw(26) << glob_prnt.info)
  if (glob_prnt.iprint==1 && nstop==0 && (ncsipl+ncsipn)!=0)
    DETAILED_MESSAGE(" |Xi_k|              " << std::setw(26) << glob_info.tot_actg_sip)
  if (glob_prnt.iprint==1 && nstop==0 && nfsip!=0)
    DETAILED_MESSAGE(" |Omega_k|           " << std::setw(26) <<
		     glob_info.tot_actf_sip)
  glob_prnt.iter++;
  if (!((glob_prnt.iter)%glob_prnt.iter_mod)) display=TRUE;
  else display=(nstop==0);
  if (glob_prnt.iter_mod!=1 && display)
    DETAILED_MESSAGE("\n iteration           " << std::setw(26) <<
		     glob_prnt.iter-1)
  if (glob_prnt.initvl==0 && display)
    sbout1(glob_prnt.io,nparam,"x                   ",dummy,x,2,1) ;
  if (display) {
    if (nob>0) {
      DETAILED_MESSAGE(" objectives");
      for (i=1; i<=nob-nfsip; i++) {
	if (nob==nobL) {
	  GENERAL_MESSAGE(" \t\t\t " << std::setw(22) << std::setprecision(14) 
			  << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			  << ob[i].val);
	}
	else {
	  GENERAL_MESSAGE(" \t\t\t " << std::setw(22) << std::setprecision(14) 
			  << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			  << fabs(ob[i].val));
	}
      }
    }
    if (nfsip) {
      offset=nob-nfsip;
      if (feasb) index=glob_info.nfsip;
      else index=glob_info.ncsipn;
      for (i=1; i<=index; i++) {
	if (nob==nobL) gmax=ob[++offset].val;
	else gmax=fabs(ob[++offset].val);
	for (j=2; j<=mesh_pts[i]; j++) {
	  offset++;
	  if (nob==nobL && ob[offset].val>gmax)
	    gmax=ob[offset].val;
	  else if (nob!=nobL && fabs(ob[offset].val)>gmax)
	    gmax=fabs(ob[offset].val);
	}
	DETAILED_MESSAGE(" \t\t\t " << std::setw(22) << std::setprecision(14) 
			 << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			 << gmax);
      }
    }
  }
  if (glob_info.mode==1 && glob_prnt.iter>1 && display)
    sbout1(glob_prnt.io,0,"objective max4      ",fM,adummy,1,1);
  if (nob>1 && display)
    sbout1(glob_prnt.io,0,"objmax              ",fmax,adummy,1,1);
  if (ncnstr != 0 && display ) {
    DETAILED_MESSAGE(" constraints");
    for (i=1; i<=nineqn-ncsipn; i++)
      DETAILED_MESSAGE(" \t\t\t " << std::setw(22) << std::setprecision(14) 
		       << std::setiosflags(std::ios::scientific|std::ios::showpos) 
		       << cs[i].val);
    if (ncsipn) {
      offset=nineqn-ncsipn;
      for (i=1; i<=glob_info.ncsipn; i++) {
	gmax=cs[++offset].val;
	for (j=2; j<=mesh_pts[glob_info.nfsip+i]; j++) {
	  offset++;
	  if (cs[offset].val>gmax) gmax=cs[offset].val;
	}
	DETAILED_MESSAGE(" \t\t\t " << std::setw(22) << std::setprecision(14) 
			 << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			 << gmax);
      }
    }
    for (i=nineqn+1; i<=glob_info.nnineq-ncsipl; i++)
      DETAILED_MESSAGE(" \t\t\t " << std::setw(22) << std::setprecision(14) 
		       << std::setiosflags(std::ios::scientific|std::ios::showpos) 
		       << cs[i].val);
    if (ncsipl) {
      offset=glob_info.nnineq-ncsipl;
      for (i=1; i<=glob_info.ncsipl; i++) {
	gmax=cs[++offset].val;
	if (feasb) index=glob_info.nfsip+glob_info.ncsipn+i;
	else index=glob_info.ncsipn+i;
	for (j=2; j<=mesh_pts[index];
	     j++) {
	  offset++;
	  if (cs[offset].val>gmax) gmax=cs[offset].val;
	}
	DETAILED_MESSAGE(" \t\t\t " << std::setw(22) << std::setprecision(14) 
			 << std::setiosflags(std::ios::scientific|std::ios::showpos) 
			 << gmax);
      }
    }
    for (i=glob_info.nnineq+1; i<=ncnstr; i++)
      DETAILED_MESSAGE(" \t\t\t " << std::setw(22) << std::setprecision(14) 
		       << std::setiosflags(std::ios::scientific|std::ios::showpos) 
		       << cs[i].val);
    if (feasb) {
      SNECV=0.e0;
      for (i=glob_info.nnineq+1; i<=glob_info.nnineq+nn-nineqn; i++)
	SNECV=SNECV+fabs(cs[i].val);
      if (glob_prnt.initvl==0 && (nn-nineqn)!=0)
	sbout1(glob_prnt.io,0,"SNECV               ",
	       SNECV,adummy,1,1);
    }
  }
  if (glob_prnt.iter<=1 && display) {
    DETAILED_MESSAGE(" ");
    DETAILED_MESSAGE(" iteration           " << std::setw(26) << glob_prnt.iter);
    goto L9000;
  }
  if (glob_prnt.iprint>=2 && glob_prnt.initvl==0 && display)
    sbout1(glob_prnt.io,0,"step                ",steps,adummy,1,1);
  if (glob_prnt.initvl==0 && display &&
      (nstop==0 || glob_prnt.info!=0 || glob_prnt.iprint==2)) {
    sbout1(glob_prnt.io,0,"d0norm              ",d0norm,adummy,1,1);
    sbout1(glob_prnt.io,0,"ktnorm              ",sktnom,adummy,1,1);
  }
  if (glob_prnt.initvl==0 && feasb && display)
    DETAILED_MESSAGE(" ncallf              " << std::setw(26) <<
		     glob_info.ncallf)
  if (glob_prnt.initvl==0 && (nn!=0 || !feasb) && display)
    DETAILED_MESSAGE(" ncallg              " << std::setw(26) <<
		     glob_info.ncallg)
  if (glob_prnt.iprint>=3 && glob_prnt.iter_mod!=1 && nstop!=0
      && !(glob_prnt.iter%glob_prnt.iter_mod))
    DETAILED_MESSAGE("\n The following was calculated during iteration "<<std::setw(5) << glob_prnt.iter <<":") 
  if (nstop != 0 && (glob_prnt.iter_mod==1))
    DETAILED_MESSAGE("\n iteration           " << std::setw(26) <<
		     glob_prnt.iter)
 L120:
  if (nstop!=0 || glob_prnt.iprint==0) goto L9000;
  DETAILED_MESSAGE("")
  if (glob_prnt.iprint>=3)
    DETAILED_MESSAGE(" inform              "<< std::setw(26) <<
		     glob_prnt.info)
  if (glob_prnt.info==0) GENERAL_MESSAGE("\nNormal termination: You have obtained a solution !!");
  if (glob_prnt.info==0 && sktnom>0.1e0) WARNING("Warning: Norm of Kuhn-Tucker vector is large !!");
  if (glob_prnt.info==3) {
    WARNING("\nWarning: Maximum iterations have been reached before\nobtaining a solution !!\n\n");
  }
  if (glob_prnt.info==4) {
    WARNING("\nError : Step size has been smaller than the computed\nmachine precision !!\n");
  }
  if (glob_prnt.info==5) WARNING("\nError: Failure in constructing d0 !!\n");
  if (glob_prnt.info==6) WARNING("\nError: Failure in constructing d1 !!\n");
  if (glob_prnt.info==8) {
    WARNING("\nError: The new iterate is numerically equivalent to the\nprevious iterate, though the stopping criterion is not \nsatisfied");
  }
  if (glob_prnt.info==9) {
    WARNING("\nError: Could not satisfy nonlinear equality constraints -\n       Penalty parameter too large\n");
  }
 L9000:
  free_dv(adummy);
  glob_prnt.initvl=0;
  return;
}
/*************************************************************/
/*   CFSQP : Computation of gradients of objective           */
/*           functions by forward finite differences         */
/*************************************************************/

#ifdef __STDC__
void grobfd(int nparam,int j,bioReal *x,bioReal *gradf,
            void (*obj)(int,int,bioReal *,bioReal *,void *),void *cd)
#else
  void grobfd(nparam,j,x,gradf,obj,cd)
  int nparam,j;
bioReal *x,*gradf;
void   (*obj)();
void   *cd;
#endif
{
  int i;
  bioReal xi,delta;

  for (i=0; i<=nparam-1; i++) {
    xi=x[i];
    delta=DMAX1(glob_grd.udelta,
		glob_grd.rteps*DMAX1(1.e0,fabs(xi)));
    if (xi<0.e0) delta=-delta;
    if (!(glob_prnt.ipd==1 || j != 1 || glob_prnt.iprint<3)) {
      /*  formats are not set yet...  */
      if (i==0) DETAILED_MESSAGE("\tdelta(i)\t " << std::setw(22) << std::setprecision(14) 
				 << std::setiosflags(std::ios::scientific|std::ios::showpos) 
				 << delta);
      if (i!=0) DETAILED_MESSAGE("\t\t\t " << std::setw(22) << std::setprecision(14) 
				 << std::setiosflags(std::ios::scientific|std::ios::showpos) 
				 << delta);
    }
    x[i]=xi+delta;
    x_is_new=TRUE;
    (*obj)(nparam,j,x,&gradf[i],cd);
    gradf[i]=(gradf[i]-glob_grd.valnom)/delta;
    x[i]=xi;
    x_is_new=TRUE;
  }
  return;
}

/***********************************************************/
/*   CFSQP : Computation of gradients of constraint        */
/*           functions by forward finite differences       */
/***********************************************************/

#ifdef __STDC__
void grcnfd(int nparam,int j,bioReal *x,bioReal *gradg,
            void (*constr)(int,int,bioReal *,bioReal *,void *),void *cd)
#else
  void grcnfd(nparam,j,x,gradg,constr,cd)
  int nparam,j;
bioReal *x,*gradg;
void   (*constr)();
void   *cd;
#endif
{
  int i;
  bioReal xi,delta;

  for (i=0; i<=nparam-1; i++) {
    xi=x[i];
    delta=DMAX1(glob_grd.udelta,
		glob_grd.rteps*DMAX1(1.e0,fabs(xi)));
    if (xi<0.e0) delta=-delta;
    if (!(j != 1 || glob_prnt.iprint<3)) {
      /*  formats are not set yet...  */
      if (i==0) DETAILED_MESSAGE("\tdelta(i)\t " << std::setw(22) << std::setprecision(14) 
				 << std::setiosflags(std::ios::scientific|std::ios::showpos) 
				 << delta);
      if (i!=0) DETAILED_MESSAGE("\t\t\t " << std::setw(22) << std::setprecision(14) 
				 << std::setiosflags(std::ios::scientific|std::ios::showpos) 
				 << delta);
      glob_prnt.ipd=1;
    }
    x[i]=xi+delta;
    x_is_new=TRUE;
    (*constr)(nparam,j,x,&gradg[i],cd);
    gradg[i]=(gradg[i]-glob_grd.valnom)/delta;
    x[i]=xi;
    x_is_new=TRUE;
  }
  return;
}
/************************************************************/
/*    Utility functions used by CFSQP -                     */
/*    Available functions:                                  */
/*      diagnl        error         estlam                  */
/*      colvec        scaprd        small                   */
/*      fool          matrvc        matrcp                  */
/*      nullvc        resign        sbout1                  */
/*      sbout2        shift         slope                   */
/*      fuscmp        indexs        element                 */
/************************************************************/

#ifdef __STDC__
static void fool(bioReal, bioReal, bioReal *);
#else
static void fool();
#endif

/************************************************************/
/*    Set a=diag*I, a diagonal matrix                       */
/************************************************************/

#ifdef __STDC__
static void diagnl(int nrowa,bioReal diag,bioReal **a)
#else
  static void diagnl(nrowa,diag,a)
  int nrowa;
bioReal **a,diag;
#endif
{
  int i,j;

  for (i=1; i<=nrowa; i++) {
    for (j=i; j<=nrowa; j++) {
      a[i][j]=0.e0;
      a[j][i]=0.e0;
    }
    a[i][i]=diag;
  }
  return;
}

/***********************************************************/
/*    Display error messages                               */
/***********************************************************/

#ifdef __STDC__
static void error(char string[],int *inform)
#else
  static void error(string,inform)
  char string[];
int *inform;
#endif
{
  if (glob_prnt.iprint>0)
    WARNING(string);
  *inform=7;
  return;
}

/***********************************************************/
/*    Compute an estimate of multipliers for updating      */
/*    penalty parameter (nonlinear equality constraints)   */
/***********************************************************/

#ifdef __STDC__
static void
estlam(int nparam,int neqn,int *ifail,bioReal bigbnd,bioReal **hess,
       bioReal *cvec,bioReal *a,bioReal *b,struct _constraint *cs,
       bioReal *psb,bioReal *bl,bioReal *bu,bioReal *x)
#else
  static void
estlam(nparam,neqn,ifail,bigbnd,hess,cvec,a,b,cs,psb,bl,bu,x)
  int nparam,neqn,*ifail;
bioReal bigbnd,**hess,*cvec,*a,*b,*psb,*bl,*bu,*x;
struct _constraint *cs;
#endif
{
  int i,j,zero,one,lwar2,mnn,iout;
  bioReal *ctemp;

  for (i=1; i<=neqn; i++) {
    bl[i]= (-bigbnd);
    bu[i]= bigbnd;
    cvec[i]=scaprd(nparam,cs[i+glob_info.nnineq].grad,psb);
    x[i]= 0.e0;
    for (j=i; j<=neqn; j++) {
      hess[i][j]=scaprd(nparam,cs[i+glob_info.nnineq].grad,
			cs[j+glob_info.nnineq].grad);
      hess[j][i]=hess[i][j];
    }
  }
  zero=0;
  one=1;
  iw[1]=1;
  mnn=2*neqn;
  ctemp=convert(hess,neqn,neqn);
  lwar2=lenw-1;
  iout=6;

  ql0001_(&zero,&zero,&one,&neqn,&neqn,&mnn,(ctemp+1),(cvec+1),
	  (a+1),(b+1),(bl+1),(bu+1),(x+1),(w+1),&iout,ifail,
	  &zero,(w+3),&lwar2,(iw+1),&leniw,&glob_grd.epsmac);

  free_dv(ctemp);
  return;
}

/**************************************************************/
/*   Extract a column vector from a matrix                    */
/**************************************************************/

#ifdef __STDC__
static bioReal *colvec(bioReal **a,int col,int nrows)
#else
  static bioReal *colvec(a,col,nrows)
  bioReal **a;
int col,nrows;
#endif
{
  bioReal *temp;
  int i;

  temp=make_dv(nrows);
  for (i=1;i<=nrows;i++)
    temp[i]=a[i][col];
  return temp;
}

/************************************************************/
/*    Compute the scalar product z=x'y                      */
/************************************************************/

#ifdef __STDC__
static bioReal scaprd(int n,bioReal *x,bioReal *y)
#else
  static bioReal scaprd(n,x,y)
  bioReal *x, *y;
int n;
#endif
{
  int i;
  bioReal z;

  z=0.e0;
  for (i=1;i<=n;i++)
    z=z+x[i]*y[i];
  return z;
}

/***********************************************************/
/*    Used by small()                                      */
/***********************************************************/

#ifdef __STDC__
static void fool(bioReal x,bioReal y,bioReal *z)
#else
  static void fool(x,y,z)
  bioReal x,y,*z;
#endif
{
  *z=x*y+y;
  return;
}

/**********************************************************/
/*    Computes the machine precision                      */
/**********************************************************/

static bioReal small()
{
  bioReal one,two,z,tsmall;

  one=1.e0;
  two=2.e0;
  tsmall=one;
  do {
    tsmall=tsmall/two;
    fool(tsmall,one,&z);
  } while (z>1.e0);
  return tsmall*two*two;
}

/**********************************************************/
/*     Compares value with threshold to see if exceeds    */
/**********************************************************/

#ifdef __STDC__
static int fuscmp(bioReal val,bioReal thrshd)
#else
  static int fuscmp(val,thrshd)
  bioReal val,thrshd;
#endif
{
  int temp;

  if (fabs(val)<=thrshd) temp=FALSE;
  else temp=TRUE;
  return temp;
}

/**********************************************************/
/*     Find the residue of i with respect to nfs          */
/**********************************************************/

#ifdef __STDC__
static int indexs(int i,int nfs)
#else
  static int indexs(i,nfs)
  int i,nfs;
#endif
{
  int mm=i;

  while (mm>nfs) mm -= nfs;
  return mm;
}

/*********************************************************/
/*     Copies matrix a to matrix b                       */
/*********************************************************/

#ifdef __STDC__
static void matrcp(int ndima,bioReal **a,int ndimb,bioReal **b)
#else
  static void matrcp(ndima,a,ndimb,b)
  bioReal **a,**b;
int ndima,ndimb;
#endif
{
  int i,j;

  for (i=1; i<=ndima; i++)
    for (j=1; j<=ndima; j++)
      b[i][j]=a[i][j];
  if (ndimb<=ndima) return;
  for (i=1; i<=ndimb; i++) {
    b[ndimb][i]=0.e0;
    b[i][ndimb]=0.e0;
  }
  return;
}

/*******************************************************/
/*     Computes y=ax                                   */
/*******************************************************/

#ifdef __STDC__
static void matrvc(int la,int na,bioReal **a,bioReal *x,bioReal *y)
#else
  static void matrvc(la,na,a,x,y)
  bioReal **a,*x,*y;
int la,na;
#endif
{
  int i,j;
  bioReal yi;

  for (i=1; i<=la; i++) {
    yi=0.e0;
    for (j=1; j<=na; j++)
      yi=yi + a[i][j]*x[j];
    y[i]=yi;
  }
  return;
}

/******************************************************/
/*      Set x=0                                       */
/******************************************************/

#ifdef __STDC__
static void nullvc(int nparam,bioReal *x)
#else
  static void nullvc(nparam,x)
  int nparam;
bioReal *x;
#endif
{
  int i;

  for (i=1; i<=nparam; i++)
    x[i]=0.e0;
  return;
}

/*********************************************************/
/*   job1=10: g*signeq,   job1=11: gradg*signeq,         */
/*                        job1=12: job1=10&11            */
/*   job1=20: do not change sign                         */
/*   job2=10: psf,        job2=11: grdpsf,               */
/*                        job2=12: job2=10&11            */
/*   job2=20: do not change sign                         */
/*********************************************************/

#ifdef __STDC__
static void
resign(int n,int neqn,bioReal *psf,bioReal *grdpsf,bioReal *penp,
       struct _constraint *cs,bioReal *signeq,int job1,int job2)
#else
  static void
resign(n,neqn,psf,grdpsf,penp,cs,signeq,job1,job2)
  int job1,job2,n,neqn;
bioReal *psf,*grdpsf,*penp,*signeq;
struct _constraint *cs;
#endif
{
  int i,j,nineq;

  nineq=glob_info.nnineq;
  if (job2==10 || job2==12) *psf=0.e0;
  for (i=1; i<=neqn; i++) {
    if (job1==10 || job1==12) cs[i+nineq].val=
				signeq[i]*cs[i+nineq].val;
    if (job2==10 || job2==12) *psf=*psf+cs[i+nineq].val*penp[i];
    if (job1==10 || job1==20) continue;
    for (j=1; j<=n; j++)
      cs[i+nineq].grad[j]=cs[i+nineq].grad[j]*signeq[i];
  }
  if (job2==10 || job2==20) return;
  nullvc(n,grdpsf);
  for (i=1; i<=n; i++)
    for (j=1; j<=neqn; j++)
      grdpsf[i]=grdpsf[i]+cs[j+nineq].grad[i]*penp[j];
  return;
}

/**********************************************************/
/*      Write output to file                              */
/**********************************************************/

#ifdef __STDC__
static void
sbout1(FILE *io,int n,char *s1,bioReal z,bioReal *z1,int job,int level)
#else
  static void sbout1(io,n,s1,z,z1,job,level)
  FILE *io;
int n,job,level;
bioReal z,*z1;
char *s1;
#endif
{
  int j;

  if (job != 2) {
    if (level == 1) {
      DETAILED_MESSAGE(" " << s1 << "    " << std::setw(22) << std::setprecision(14) 
		       << std::setiosflags(std::ios::scientific|std::ios::showpos) 
		       << z) ;
    }
    if (level == 2) {
      DETAILED_MESSAGE("\t\t\t" << s1 << std::setw(22) << std::setprecision(14) 
		       << std::setiosflags(std::ios::scientific|std::ios::showpos)<< z) ;
    }
    return;
  }
  if (n==0) return;
  if (level == 1) {
    DETAILED_MESSAGE(" "<<s1<<"\t " << std::setw(22) << std::setprecision(14) 
		     << std::setiosflags(std::ios::scientific|std::ios::showpos)<< z1[1]) ;
  }
  if (level == 2) {
    DETAILED_MESSAGE("\t\t\t "<<s1<< std::setw(22) << std::setprecision(14) 
		     << std::setiosflags(std::ios::scientific|std::ios::showpos)<< z1[1]) ;
  }
  for (j=2; j<=n; j++) {
    if (level == 1) {
      DETAILED_MESSAGE(" \t\t\t " << std::setw(22) << std::setprecision(14) 
		       << std::setiosflags(std::ios::scientific|std::ios::showpos)<< z1[j]) ;
    }
    if (level == 2) {
      DETAILED_MESSAGE("\t\t\t\t\t\t " << std::setw(22) << std::setprecision(14) 
		       << std::setiosflags(std::ios::scientific|std::ios::showpos)<< z1[j]) ;
    }
  }
  return;
}

/*********************************************************/
/*      Write output to file                             */
/*********************************************************/

#ifdef __STDC__
static void
sbout2(FILE *io,int n,int i,char *s1,char *s2,bioReal *z)
#else
  static void sbout2(io,n,i,s1,s2,z)
  FILE *io;
int n,i;
bioReal *z;
char *s1,*s2;
#endif
{
  int j;


  DETAILED_MESSAGE("\t\t\t" << std::setw(8) << s1 << ' ' 
		   << std::setw(5) << i <<  ' ' 
		   << std::setw(1) << s2 << ' ' 
		   << std::setw(22) << std::setprecision(14) 
		   << std::setiosflags(std::ios::scientific|std::ios::showpos) << z[1]) ;
  for (j=2; j<=n; j++)
    DETAILED_MESSAGE("\t\t\t\t\t\t"
		     << std::setw(22) << std::setprecision(14) 
		     << std::setiosflags(std::ios::scientific|std::ios::showpos) << z[j]) ;
  return;
}

/*********************************************************/
/*      Extract ii from iact and push in front           */
/*********************************************************/

#ifdef __STDC__
static void shift(int n,int ii,int *iact)
#else
  static void shift(n,ii,iact)
  int n,ii,*iact;
#endif
{
  int j,k;

  if (ii == iact[1]) return;
  for (j=1; j<=n; j++) {
    if (ii != iact[j]) continue;
    for (k=j; k>=2; k--)
      iact[k]=iact[k-1];
    break;
  }
  if (n!=0) iact[1]=ii;
  return;
}

/****************************************************************/
/*      job=0 : Compute the generalized gradient of the minimax */
/*      job=1 : Compute rhog in mode = 1                        */
/****************************************************************/

#ifdef __STDC__
static bioReal
slope(int nob,int nobL,int neqn,int nparam,int feasb,
      struct _objective *ob,bioReal *grdpsf,bioReal *x,bioReal *y,
      bioReal fmax,bioReal theta,int job,bioReal *prev,int old)
#else
  static bioReal
slope(nob,nobL,neqn,nparam,feasb,ob,grdpsf,x,y,fmax,theta,job,
      prev,old)
  int nob,nobL,neqn,nparam,job,feasb,old;
bioReal fmax, theta;
bioReal *grdpsf,*x,*y,* prev;
struct _objective *ob;
#endif
{
  int i;
  bioReal slope1,rhs,rhog,grdftx,grdfty,diff,grpstx,grpsty;
  bioReal tslope;

  tslope=-bgbnd;
  if (feasb && nob==0) tslope=0.e0;
  if (neqn==0 || !feasb) {
    grpstx=0.e0;
    grpsty=0.e0;
  } else {
    grpstx=scaprd(nparam,grdpsf,x);
    grpsty=scaprd(nparam,grdpsf,y);
  }
  for (i=1; i<=nob; i++) {
    if (old) slope1=prev[i]+scaprd(nparam,ob[i].grad,x);
    else slope1=ob[i].val+scaprd(nparam,ob[i].grad,x);
    tslope=DMAX1(tslope,slope1);
    if (nobL != nob) tslope=DMAX1(tslope,-slope1);
  }
  tslope=tslope-fmax-grpstx;
  if (job == 0) return tslope;
  rhs=theta*tslope+fmax;
  rhog=1.e0;
  for (i=1; i<=nob; i++) {
    grdftx=scaprd(nparam,ob[i].grad,x)-grpstx;
    grdfty=scaprd(nparam,ob[i].grad,y)-grpsty;
    diff=grdfty-grdftx;
    if (diff <= 0.e0) continue;
    rhog=DMIN1(rhog,(rhs-ob[i].val-grdftx)/diff);
    if (nobL != nob) rhog=DMIN1(rhog,-(rhs+ob[i].val+grdftx)/diff);
  }
  tslope=rhog;
  return tslope;
}

/************************************************************/
/*  Determine whether index is in set                       */
/************************************************************/

#ifdef __STDC__
static int element(int *set,int length,int index)
#else
  static int element(set, length, index)
  int *set;
int length,index;
#endif
{
  int i,temp;

  temp=FALSE;
  for (i=1; i<=length; i++) {
    if (set[i]==0) break;
    if (set[i]==index) {
      temp=TRUE;
      return temp;
    }
  }
  return temp;
}
/*************************************************************/
/*     Memory allocation utilities for CFSQP                 */
/*                                                           */
/*     All vectors and matrices are intended to              */
/*     be subscripted from 1 to n, NOT 0 to n-1.             */
/*     The addreses returned assume this convention.         */
/*************************************************************/

/*************************************************************/
/*     Create bioReal precision vector                        */
/*************************************************************/

#ifdef __STDC__
static bioReal *
make_dv(int len)
#else
  static bioReal *
make_dv(len)
  int len;
#endif
{
  bioReal *v;

  if (!len) len=1;
  v=(bioReal *)calloc(len,sizeof(bioReal));
  if (!v) {
    FATAL("Run-time error in make_dv");
    exit(1);
  }
  return --v;
}

/*************************************************************/
/*     Create integer vector                                 */
/*************************************************************/

#ifdef __STDC__
static int *
make_iv(int len)
#else
  static int *
make_iv(len)
  int len;
#endif
{
  int *v;

  if (!len) len=1;
  v=(int *)calloc(len,sizeof(int));
  if (!v) {
    FATAL("Run-time error in make_iv");
    exit(1);
  }
  return --v;
}

/*************************************************************/
/*     Create a bioReal precision matrix                      */
/*************************************************************/

#ifdef __STDC__
static bioReal **
make_dm(int rows, int cols)
#else
  static bioReal **
make_dm(rows, cols)
  int rows, cols;
#endif
{
  bioReal **temp;
  int i;

  if (rows == 0) rows=1;
  if (cols == 0) cols=1;
  temp=(bioReal **)calloc(rows,sizeof(bioReal *));
  if (!temp) {
    FATAL("Run-time error in make_dm");
    exit(1);
  }
  temp--;
  for (i=1; i<=rows; i++) {
    temp[i]=(bioReal *)calloc(cols,sizeof(bioReal));
    if (!temp[i]) {
      FATAL("Run-time error in make_dm");
      exit(1);
    }
    temp[i]--;
  }
  return temp;
}

/*************************************************************/
/*     Free a bioReal precision vector                        */
/*************************************************************/

#ifdef __STDC__
static void
free_dv(bioReal *v)
#else
  static void
free_dv(v)
  bioReal *v;
#endif
{
  free((char *) (v+1));
}

/*************************************************************/
/*     Free an integer vector                                */
/*************************************************************/

#ifdef __STDC__
static void
free_iv(int *v)
#else
  static void
free_iv(v)
  int *v;
#endif
{
  free((char *) (v+1));
}

/*************************************************************/
/*     Free a bioReal precision matrix                        */
/*************************************************************/

#ifdef __STDC__
static void
free_dm(bioReal **m,int rows)
#else
  static void
free_dm(m,rows)
  bioReal **m;
int rows;
#endif
{
  int i;

  if (!rows) rows=1;
  for (i=1; i<=rows; i++) free((char *) (m[i]+1));
  free ((char *) (m+1));
}

/*************************************************************/
/*     Converts matrix a into a form that can easily be      */
/*     passed to a FORTRAN subroutine.                       */
/*************************************************************/

#ifdef __STDC__
static bioReal *
convert(bioReal **a,int m,int n)
#else
  static bioReal *
convert(a,m,n)
  bioReal **a;
int m,n;
#endif
{
  bioReal *temp;
  int i,j;

  temp = make_dv(m*n);

  for (i=1; i<=n; i++)     /* loop thru columns */
    for (j=1; j<=m; j++)  /* loop thru row     */
      temp[(m*(i-1)+j)] = a[j][i];

  return temp;
}

