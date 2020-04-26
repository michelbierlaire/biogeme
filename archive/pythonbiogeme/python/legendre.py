## \file
# Fonctions for transformed Legendre polynomials, orthonormal on [0,1]: \f[ L_n(x) = \frac{\sqrt{4n^2-1}}{n}(2x-1)L_{n-1}(x)-\frac{(n-1)\sqrt{2n+1}}{n \sqrt{2n-3}} L_{n-2}(x), \f] with \f$L_0(x)=1\f$, \f$L_1(x)=\sqrt{3}(2x-1)\f$ and \f$L_2(x) = \sqrt{5}(6x^2-6x+1)\f$. See \cite FosgBier07.
from math import sqrt

## Implements the transformed Legendre polynomials of degree 0 \f[ L_0(x) = 1 \f]
# \param x argument of the polynomial
# \return value of the polynomial
# \ingroup specs
def legendre00(x):
    return 1

## Implements the transformed Legendre polynomials of degree 1 \f[ L_1(x)=\sqrt{3}(2x-1) \f]
# \param x argument of the polynomial
# \return value of the polynomial
# \ingroup specs
def legendre01(x):
    return sqrt(3.0) * (2 * x - 1)

## Implements the transformed Legendre polynomials of degree 2 \f[ L_2(x)=\sqrt{5}(6x^2-6x+1) \f]
# \param x argument of the polynomial
# \return value of the polynomial
# \ingroup specs
def legendre02(x):
    return sqrt(5.0) * (6 * x * x - 6 * x + 1)

## Implements the transformed Legendre polynomials of degree 3 \f[ L_3(x)=\sqrt{7}(20 x^3 30 x^2+12x-1) \f]
# \param x argument of the polynomial
# \return value of the polynomial
# \ingroup specs
def legendre03(x):
    return sqrt(7.0) * (20 * x * x * x - 30 * x * x + 12 * x - 1)

## Implements the transformed Legendre polynomials of degree 4
# \param x argument of the polynomial
# \return value of the polynomial
# \ingroup specs
def legendre04(x):
    i = 4 
    t1 = sqrt(4 * i * i - 1) / i
    t2 = (i-1)*sqrt(2*i+1)/(i*sqrt(2*i-3))
    return t1 * (2*x-1) * legendre03(x) - t2 * legendre02(x)

## Implements the transformed Legendre polynomials of degree 5
# \param x argument of the polynomial
# \return value of the polynomial
# \ingroup specs
def legendre05(x):
    i = 5
    t1 = sqrt(4 * i * i - 1) / i
    t2 = (i-1)*sqrt(2*i+1)/(i*sqrt(2*i-3))
    return t1 * (2*x-1) * legendre04(x) - t2 * legendre03(x)

## Implements the transformed Legendre polynomials of degree 6
# \param x argument of the polynomial
# \return value of the polynomial
# \ingroup specs
def legendre06(x):
    i = 6
    t1 = sqrt(4 * i * i - 1) / i
    t2 = (i-1)*sqrt(2*i+1)/(i*sqrt(2*i-3))
    return t1 * (2*x-1) * legendre05(x) - t2 * legendre04(x)

## Implements the transformed Legendre polynomials of degree 7
# \param x argument of the polynomial
# \return value of the polynomial
# \ingroup specs
def legendre07(x):
    i = 7
    t1 = sqrt(4 * i * i - 1) / i
    t2 = (i-1)*sqrt(2*i+1)/(i*sqrt(2*i-3))
    return t1 * (2*x-1) * legendre06(x) - t2 * legendre05(x)

