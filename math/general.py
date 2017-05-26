# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:      general.py
# Purpose:   general ath Utility programs
#
# Author:      Indranil Sinharoy
#
# Created:        22/09/2012
# Last Modified:  04/24/2017
# Copyright:      (c) Indranil Sinharoy, 2012 - 2017
# Licence:        MIT License
#-------------------------------------------------------------------------------
"""General math utilities
"""
from __future__ import division, print_function
import numpy as _np
import sympy as _sym

# trig functions
sind = lambda x : _np.sin(_np.deg2rad(x))
cosd = lambda x : _np.cos(_np.deg2rad(x))
tand = lambda x : _np.tan(_np.deg2rad(x))

# inverse trig functions
arcsind = lambda x : _np.rad2deg(_np.arcsin(x))
arccosd = lambda x : _np.rad2deg(_np.arccos(x))
arctand = lambda x : _np.rad2deg(_np.arctan(x))

# docstrings of the trig functions
sind.__doc__ = "Sine of argument in degrees"
cosd.__doc__ = "Cosine of argument in degrees"
tand.__doc__ = "Tangent of argument in degrees"
arcsind.__doc__ = "Inverse sine in degrees"
arccosd.__doc__ = "Inverse cosine in degrees"
arctand.__doc__ = "Inverse tan in degrees"



def factorial(n):
    """return n factorial    

    The factorial is defined as n! = n*(n-1)!
    Note: The recursive factoial implementation will break down if n is too large
    """
    if(n==0):
        return 1
    else:
        return n*factorial(n-1)

def fibonacci(n):
    """returns a fibonacci sequence generator object for fibonacci sequence of n
    elements

    Examples
    --------
    >>> fs = fibonacci(2)
    >>> list(fs)
    [1, 1]
    >>> fs = fibonacci(10)
    >>> list(fs)
    [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    """
    low, high = 0, 1
    count = 0
    while count < n:
        yield high
        low, high = high, low + high
        count +=1

def golden_ratio(n):
    """returns the Golden Ratio to an accuracy of `n` decimal places

    Examples
    --------
    >>> print(golden_ratio(20))
    1.61803398874989484820
    >>> print(golden_ratio(50))
    1.61803398874989484820458683436563811772030917980576
    """
    x = _sym.symbols('x')
    gr = [sol for sol in _sym.solve(x**2 - x - 1) if sol > 0][0]
    return gr.evalf(n + 1)


def nCk(n, k):
    """return the number of possible combinations of `k` objects from a set of `n` objects
    (i.e., order does not matter).

    Other phrases include "`n` choose `k`" or "the number of combinations of `n` things 
    taken `k` at a time".

    Defined for positive integers `n>=k` as `nCk(n,k) = n!/(k!*(n-k)!)`

    nCk is often represented as  / n \
                                 |   |
                                 \ k /

    Examples
    --------
    >>> nCk(3, 2)
    3

    See Also
    --------
    itertools.combinations(iterable, r): It returns successive r-length combinations of 
    elements in the iterable. For example,
        `list(combinations(['A', 'B', 'C'], 2)) -->
        [('A', 'B'), ('A', 'C'), ('B', 'C')]`
    """
    return factorial(n)/(factorial(k)*factorial(n-k))

def nPk(n,k):
    """return the number of ways to arange `k` items out of `n` items. i.e., the order 
    matters 

    It is defined for positive integers `n>=k` as `nPk(n,k) = n!/(n-k)!`
    """
    return factorial(n)/factorial(n-k)

def binomial_distribution(n, p):
    """returns the Binomial probability distribution (PMF)

    Parameters
    ----------
    n : integer
        the number of samples or subexperiments or Bernoulli trials
    p : real
        the probability of success in each Bernoulli trial

    Notes
    -----
    The PMF for a binomial RV X is given as:
        p_x[k] = nCk* p^k * (1-p)^(n-k)
    """
    return [nCk(n,k)*(p**k)*((1-p)**(n-k)) for k in range(n+1)]

def cart2pol(X, Y, Z=None):
    """Transform Cartesian coordinates to polar or cylindrical

    cart2pol(X, Y [,Z]) -> rho, theta [,Z]

    Parameters
    ----------
    X  : Cartesian coordinate array
    Y  : Cartesian coordinate array
    Z  : (optional) if present, then the function converts 3D Cartesian coordinates
         to cylindrical coordinates.

    Returns
    -------
    rho   : distance from the origin to a point in the x-y plane
    theta : counterclockwise angular displacement in radians from the positive x-axis
    Z     : height above the x-y plane

    """
    if X.shape != Y.shape:
        raise ValueError("Input error: Expecting X.shape == Y.shape")
    rho = _np.sqrt(X**2.0 + Y**2.0)
    theta = _np.arctan2(Y,X)
    if Z:
        return rho, theta, Z
    else:
        return rho, theta

## Test the functions
def _test_factorial():
    print("test the factorial function")
    assert factorial(5)==120


def _test_nCk():
    print("test combination")
    print(nCk(10,2))
    #print nCk(2500,2)  #Right now this fails ... the recursion is too large...

def _test_cart2pol():
    print("test cart2pol()")
    X,Y = _np.mgrid[-2:2:10j,-2:2:10j]
    # Convert from cartesian to polar choosing the grid manually
    # Logic from http://scien.stanford.edu/pages/labsite/2003/psych221/projects/03/pmaeda/index_files/zernike.m
    # however there is something work
    # FIXME!! what is the correct logic for this. For now I assume that using atan2()
    # gives the correct results.
    rhoExp = _np.sqrt(X**2 + Y**2)
    mask = ((X >= 0) & (Y >= 0)) | ((X >= 0) & (Y < 0))
    thetaExp = _np.where(mask,_np.arctan(Y/(X+1e-15)),_np.pi + _np.arctan(Y/(X+1e-15)))
    rho, theta = cart2pol(X,Y)
    abs_diff_rho = _np.sum(_np.abs(rhoExp - rho))
    abs_diff_theta = _np.sum(_np.abs(thetaExp - theta))
    print(abs_diff_rho)
    print(abs_diff_theta)
    print(thetaExp)
    print(theta)

def _test_fibonacci():
    print("test fibonacci()")
    fs = fibonacci(10)
    fsl = list(fs)
    fsexp = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    _nt.assert_array_equal(fsl, fsexp)

def _test_goldenRatio():
    print("test goldenRatio()")
    gr15 = 1.618033988749895
    _nt.assert_almost_equal(golden_ratio(15), gr15, decimal=15)
    print(golden_ratio(50))

if __name__ == '__main__':
    import numpy.testing as _nt
    _test_factorial()
    _test_nCk()
    _test_cart2pol()
    _test_fibonacci()
    _test_goldenRatio()