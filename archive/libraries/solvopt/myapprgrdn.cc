/**
This file has been adapted for g++ compiler by Michel Bierlaire
Thu Feb 21 08:35:48 2002
*/

#include <cstdlib>
#include "patMath.h"
#include "patType.h"

void apprgrdn ( unsigned short n,
                patReal g[],
                patReal x[],
                patReal f,
                patReal fun(patReal x[]),
                patReal deltax[],
                unsigned short obj
              )
{              
/* Function APPRGRDN performs the finite difference approximation 
   of the gradient <g> at a point <x>.
   f      is the calculated function value at a point <x>,
   <fun>  is the name of a function that calculates function values,
   deltax is an array of the relative stepsizes.
   obj    is the flag indicating whether the gradient of the objective
          function (1) or the constraint function (0) is to be calculated. 
*/
  patReal const lowbndobj=2.0e-10, lowbndcnt=5.0e-15, ten=10.0, half=0.5; 
  patReal d, y, fi;
  unsigned short i, j, center;
  for (i=0;i<n;i++)
  {   y=x[i];   d=patMax(lowbndcnt,fabs(y));  d*=deltax[i];
      if (obj)
      {  if (fabs(d)<lowbndobj) 
         {   if (deltax[i]<0.0) d=-lowbndobj; else d=lowbndobj;
             center=1;
         }    
         else  center=0;
      }
      else if (fabs(d)<lowbndcnt)
      {   if (deltax[i]<0.0) d=-lowbndcnt; else d=lowbndcnt;
      }
      x[i]=y+d; fi = fun(x);
      if (obj)
      {  if (fi==f)
         {  for (j=1;j<=3;j++)
            {   d*=ten; x[i]=y+d; fi = fun(x);
                if (fi!=f) break;
            }    
         }
      } 
      g[i]=(fi-f)/d;
      if (obj) 
      {  if (center) 
         {  x[i]=y-d; fi = fun(x);
            g[i]=half*(g[i]+(f-fi)/d);
         }
      }
      x[i]=y;
  }
}
