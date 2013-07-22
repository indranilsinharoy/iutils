#-------------------------------------------------------------------------------
# Name:      mathUtils.py
# Purpose:   Math Utility programs
#
# Author:      Indranil Sinharoy
#
# Created:     22/09/2012
# Copyright:   (c) Indranil Sinharoy, 2012 - 2013
# Licence:     MIT License
#-------------------------------------------------------------------------------
import numpy as _np

#Import rogue library (test matrix library similar to Matlab's gallery()) if
#available
try:
    import rogues.matrices as rm
except ImportError:
    print "Did not import rogues.matrices"
else:
    print "Imported rogues.matrices as rm"
try:
    import rogues.utils as ru
except ImportError:
    print "Did not import rogues.utils"
else:
    print "Imported rogues.utils as ru"




## General math utilities
def factorial(n):
    """
    factorial is defined as n! = n*(n-1)!
    Note: The recursive factoial implementation will break down if n is too large
    """
    if(n==0):
        return 1
    else:
        return n*factorial(n-1)

def nCk(n,k):
    """
    nCk: nCk is n combination k or choose k items out of n items, is defined for
    positive integers n>=k as nCk(n,k) = n!/(k!*(n-k)!)
    """
    return factorial(n)/(factorial(k)*factorial(n-k))

def nPk(n,k):
    """
    nPk: nPk is n permutation k or arange k items out of n items, is defined for
    positive integers n>=k as nPk(n,k) = n!/(n-k)!
    """
    return factorial(n)/factorial(n-k)

def binomialDistribution(n,p):
    """
    Binomial distribution is the probability distribution
    Formula nCk* p^k * (1-p)^(n-k)
    inputs n = number of samples/experiments and p the probability of success
    """
    return [nCk(n,k)*(p**k)*((1-p)**(n-k)) for k in range(n+1)]

## Linear Algebra utilities

def gallery(n=3):
    """
    Similar to MATLAB's test matrix gallery(3) and gallery(5). gallery(3) returns
    a badly conditioned 3-by-3 matrix. gallery(5) is an interesting eigenvalue
    problem. The Matlab documentation says, "Try to find its EXACT eigenvalues
    and eigenvectors. 5-by-5 matrix. For other test matrices, import the rogues
    package.
    """
    if (n==5):
        return _np.matrix([[  -9,     11,    -21,     63,    -252],
                         [  70,    -69,    141,   -421,    1684],
                         [-575,    575,  -1149,   3451,  -13801],
                         [3891,  -3891,   7782, -23345,   93365],
                         [1024,  -1024,   2048,  -6144,   24572]])
    else:
        return _np.matrix([[-149,-50,-154],
                         [537,180,546],
                         [-27,-9,-25]])

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
    print "test the factorial function"
    assert factorial(5)==120


def _test_nCk():
    print "test combination"
    print nCk(10,2)
    #print nCk(2500,2)  #Right now this fails ... the recursion is too large...

def _test_gallery():
    print "test gallery(3)"
    print gallery(3)
    print "test gallery(5)"
    print gallery(5)

def _test_cart2pol():
    print "test cart2pol()"    
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

if __name__ == '__main__':
#    _test_factorial()
#    _test_nCk()
#    _test_gallery()
    _test_cart2pol()