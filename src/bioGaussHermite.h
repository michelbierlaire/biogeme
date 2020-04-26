//-*-c++-*------------------------------------------------------------
//
// File name : bioGaussHermite.h
// Author :    Michel Bierlaire
// Date :      Thu Apr  8 10:39:09 2010
// Modified for biogemepython 3.0: Wed May  9 16:06:22 2018
//
//--------------------------------------------------------------------

#ifndef bioGaussHermite_h
#define bioGaussHermite_h

// Computes the integral from -infinity to +infinity of 
//
//   f(x) exp(-x*x)
//
// using the Gauss-Hermite quadrature method.
// If the integral of f(x) is needed, make sure to pre-multiply it by exp(x*x).

// This class is based on the  code found at 
//   http://mymathlib.webtrellis.net/quadrature/gauss/gauss_hermite.html

#include "bioTypes.h"
#include "bioGhFunction.h"

static const bioReal x[] = {
    1.10795872422439482889e-01,    3.32414692342231807054e-01,
    5.54114823591616988249e-01,    7.75950761540145781976e-01,
    9.97977436098105243902e-01,    1.22025039121895305882e+00,
    1.44282597021593278768e+00,    1.66576150874150946983e+00,
    1.88911553742700837153e+00,    2.11294799637118795206e+00,
    2.33732046390687850509e+00,    2.56229640237260802502e+00,
    2.78794142398198931316e+00,    3.01432358033115551667e+00,
    3.24151367963101295043e+00,    3.46958563641858916968e+00,
    3.69861685931849193984e+00,    3.92868868342767097205e+00,
    4.15988685513103054019e+00,    4.39230207868268401677e+00,
    4.62603063578715577309e+00,    4.86117509179121020995e+00,
    5.09784510508913624692e+00,    5.33615836013836049734e+00,
    5.57624164932992410311e+00,    5.81823213520351704715e+00,
    6.06227883261430263882e+00,    6.30854436111213512156e+00,
    6.55720703192153931598e+00,    6.80846335285879641431e+00,
    7.06253106024886543766e+00,    7.31965282230453531632e+00,
    7.58010080785748888415e+00,    7.84418238446082116862e+00,
    8.11224731116279191689e+00,    8.38469694041626507474e+00,
    8.66199616813451771409e+00,    8.94468921732547447845e+00,
    9.23342089021916155069e+00,    9.52896582339011480496e+00,
    9.83226980777796909401e+00,    1.01445099412928454695e+01,
    1.04671854213428121416e+01,    1.08022607536847145950e+01,
    1.11524043855851252649e+01,    1.15214154007870302416e+01,
    1.19150619431141658018e+01,    1.23429642228596742953e+01,
    1.28237997494878089065e+01,    1.34064873381449101387e+01
};

static const bioReal A[] = {
    2.18892629587439125060e-01,    1.98462850254186477710e-01,
    1.63130030502782941425e-01,    1.21537986844104181985e-01,
    8.20518273912244646789e-02,    5.01758126774286956964e-02,
    2.77791273859335142698e-02,    1.39156652202318064178e-02,
    6.30300028560805254921e-03,    2.57927326005909017346e-03,
    9.52692188548619117497e-04,    3.17291971043300305539e-04,
    9.51716277855096647040e-05,    2.56761593845490630553e-05,
    6.22152481777786331722e-06,    1.35179715911036728661e-06,
    2.62909748375372507934e-07,    4.56812750848493951350e-08,
    7.07585728388957290740e-09,    9.74792125387162124528e-10,
    1.19130063492907294976e-10,    1.28790382573155823282e-11,
    1.22787851441012497000e-12,    1.02887493735099254677e-13,
    7.54889687791524329227e-15,    4.82983532170303334787e-16,
    2.68249216476037608006e-17,    1.28683292112115327575e-18,
    5.30231618313184868536e-20,    1.86499767513025225814e-21,
    5.56102696165916731717e-23,    1.39484152606876708047e-24,
    2.91735007262933241788e-26,    5.03779116621318778423e-28,
    7.10181222638493422964e-30,    8.06743427870937717382e-32,
    7.27457259688776757460e-34,    5.11623260438522218054e-36,
    2.74878488435711249209e-38,    1.10047068271422366943e-40,
    3.18521787783591793076e-43,    6.42072520534847248278e-46,
    8.59756395482527161007e-49,    7.19152946346337102982e-52,
    3.45947793647555044453e-55,    8.51888308176163378638e-59,
    9.01922230369355617950e-63,    3.08302899000327481204e-67,
    1.97286057487945255443e-72,    5.90806786503120681541e-79
};

#define NUM_OF_POSITIVE_ZEROS  sizeof(x) / sizeof(bioReal)
#define NUM_OF_ZEROS           NUM_OF_POSITIVE_ZEROS+NUM_OF_POSITIVE_ZEROS


class bioGaussHermite {
  friend class bioGhFunction ;
 public:
  bioGaussHermite(bioGhFunction* f) ;
  std::vector<bioReal> integrate() ;
 private:
  void Gauss_Hermite_Coefs_100pts( bioReal coef[]) ;
  void Gauss_Hermite_Zeros_100pts( bioReal zeros[] ) ;
  bioGhFunction* theFunction ;
};

#endif
