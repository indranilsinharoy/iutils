#-------------------------------------------------------------------------------
# Name:          signals.py
# Purpose:       collection of signals. Most of these are not available directly from
#                scipy.signal or scipy.special.
#
# Author:        Indranil Sinharoy
#
# Created:       22/09/2012
# Last Modified:
# Copyright:     (c) Indranil Sinharoy 2012, 2013
# Licence:       MIT License
#-------------------------------------------------------------------------------
"""Collection of signals. Most of these are not available directly from scipy.signal 
or scipy.special.
"""
from __future__ import division, print_function
import warnings
import numpy as _np
import math as _math
import scipy.special as _sps

# Import jit and double from numba if available.
import iutils as iu
_hasNumba = iu._hasNumba
if _hasNumba:
    from numba import jit, double

def rect(x, a=1.0, sharp_transition=True):
    """One dimensional rectangular function of width `a`
    
    rect(x [, a, sharp_transition])->y

    Parameters
    ---------
    x : ndarray. input vector
    a : float. Width of the rect function  (Default a=1.0)
    sharp_transition: bool. if True (by default), the function forces the value to be
                      1.0 at the edges.
    
    Returns
    -------
    y : ndarray, the output vector

    Definition
    ---------
    rect(x/a) =  1 for abs(x) < a/2; 
               0.5 for abs(x) = a/2;
                 0 for abs(x) > a/2

    Examples
    --------
    >>> rect(np.linspace(-1,1,10))
    array([ 0., 0., 0., 1., 1., 1., 1., 0., 0., 0.])

    """
    y = _np.zeros(_np.shape(x))
    y[_np.abs(x)<=0.5*a] = 1.0
    # TODO !!! what should be the correct boundary conditions
    if not sharp_transition:
        y[_np.abs(x)==0.5*a] = 0.5  
    return y
    
def rect2d(X, Y, a=1.0, b=1.0, sharp_transition=True):
    """Two dimensional rectangular function of width `a` and height `b`.
    
    rect(x , y [, a, b, sharp_transition])->z

    Parameters
    ---------
    X : grid of x-corodinate (2-D ndarray, usually obtained from meshgrid)
    Y : grid of y-corodinate (2-D ndarray, usually obtained from meshgrid)
    a : float. Width (along x-axis) of the rect function  (Default a=1.0)
    b : float. Width (along y-axis) of the rect function  (Default b=1.0)
    sharp_transition: bool. if True (by default), the function forces the value to be
                      1.0 at the edges.
    
    Returns
    -------
    z : 2-D ndarray. The 2-D rectangular window

    Definition
    ---------
    rect2d(x,y) = rect(x)*rect(y)

    Given two vectors x and y, you can create a 2D rect as follows:
    X,Y = np.meshgrid(x,y);           # create the 2D coordinates
    z = rect(X/a)*rect(Y/b)           # rectangular function

    Examples
    --------

    Reference: "Fourier Analysis and Imaging," Ronald N. Bracewell
    """
    z = rect(X/a, sharp_transition)*rect(Y/b, sharp_transition)
    return z
    
def tri(x):
    """One dimensional triangle function
    tri(x)->y

    args: x = input vector
    ret:  y = output vector

    Definition:
    tri(x) =  1 - abs(x) for abs(x) <= 1.0; 0 otherwise
    """
    y = _np.zeros(_np.shape(x))
    y[_np.abs(x)<1.0]=1.0-_np.abs(x[_np.abs(x)<1.0])
    return y

def circ(r):
    """Circular function
    circ(r)->out

    Examples
    --------
    To create a circular array
    >>> x = np.linspace(-2, 2, 200); y = x
    >>> X,Y = np.meshgrid(x,y)
    >>> Z = fou.circ(np.sqrt(X**2 + Y**2))
    """
    out = _np.zeros(_np.shape(r))
    out[_np.abs(r)<=1]=1
    return out

def jinc(x, normalize=True):
    """Calculate the jinc of x. x can be a single number of an ndarray.
    The jinc function is defined as `J_1(x)/x` such that jinc(0)=0.5. The normalized 
    (default) jinc function is obtained by multiplying by 2, such that jinc(0)=1
    
    Parameters
    ----------
    x         : x may be a single point or an ndarray
    normalize : if true, the normalized jinc function is returned
    
    Returns
    -------
    result  : jinc(x)
    
    Reference: Weisstein, Eric W. "Jinc Function." From MathWorld--A Wolfram Web Resource. 
               http://mathworld.wolfram.com/JincFunction.html
    """
    # This is the interfact function. The actual function is implemented in _jinc()
    nFactor = 2.0 if normalize else 1.0
    numpyArray = isinstance(x,_np.ndarray)
    if not numpyArray:
        result = 0.5 if x==0.0 else _sps.j1(x)/(x)
    elif _hasNumba and numpyArray and False:  # For now the numba implementation is disabled as it is actually slower than the vectorized implementation.
        if x.ndim==2:
            fast_jinc = jit(double[:,:](double[:,:]))(_jinc2)
        elif x.ndim==1:
            fast_jinc = jit(double[:](double[:]))(_jinc2)
        result = fast_jinc(x)
    else:
        result = _jinc(x)    # Numba not present
    return nFactor*result

def jinc_goodman(rho):
    """Calculate the jinc function over a radial grid `rho`
    The `jinc(rho)` is defined as `2*(J_1(2*pi*rho)/2*pi*rho)` and `jinc(0)=1.0`.

    Parameters
    ----------
    rho  : ndarray
           The radial co-ordinate grid, which by be obtained from the cartesian
           co-ordinates as rho = np.hypot(X,Y)

    Returns
    -------
    result : ndarray
           The jinc function as defined in "Introduction to Fourier Optics", Joseph
           Goodman, 3rd Edition, Chapter-2, page-15.
    """
    return jinc(2.0*_math.pi*rho, normalize=True)

def _jinc(x):
    """Internal implementation of the `jinc` function. 
    The `jinc(x)` is defined as `J_1(x)/2x` and `jinc(0)=0.5`.
    """
    mask = x != 0.0
    result = 0.5*_np.ones(x.shape)
    result[mask] = _sps.j1(x[mask])/(x[mask])
    return result

def _jinc2(x):
    """Internal implementation of the jinc function for use with Numba @jit decorator
    The `jinc(x)` is defined as `J_1(x)/2x` and `jinc(0)=0.5`.
    NOTE: jinc(0) as not been implemented yet...
    """
    # TODO !!! implement jinc(0) = 0.5
    M, N = x.shape
    result = _np.empty((M,N))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(M):
            for j in range(N):
                result[i,j] = _sps.j1(x[i,j])/(x[i,j])
    return result

# ---------------------------
#   TEST FUNCTIONS
# ---------------------------

def _test_rect():
    """Test the rectangular function"""
    x = _np.linspace(-1,1,10)
    y1 = rect(x)
    y2 = _np.array([ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.]);
    nt.assert_array_equal(y1,y2)

    X,Y = _np.meshgrid(x,x)
    g = rect(X)*rect(Y)
    g_exp = _np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
                      [ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
                      [ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
                      [ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
                      [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    nt.assert_array_equal(g, g_exp)
    z = rect2d(X,Y)
    nt.assert_array_equal(z, g_exp)
    print("test_rect() is successful!")


def _test_circ():
    pass
    # TODO !!!
    
def _test_tri():
    pass
    # TODO !!!

def _test_jinc():
    """Test the jinc function"""
    # rho is numpy arrray scalar value 0
    nt.assert_equal(jinc_goodman(_np.ndarray([0])),1.0, err_msg="Expecting 1", verbose=True)
    # rho is numpy arrray scalar value 0.25
    nt.assert_almost_equal(jinc_goodman(_np.ndarray([0.25])), 0.72170284,
                           err_msg="Expecting 0.72170284", decimal=3 ,verbose=True)
    # rho is a vector (ndarray with 1 dimension)
    result = jinc_goodman(_np.array([0, 0.5, 0.75]))
    exp_jinc = array([ 1. ,  0.18119175, -0.11953933])
    nt.assert_array_almost_equal(exp_jinc, result, decimal=6)
    # rho is a 2-dimensional ndarray
    result = jinc_goodman(_np.array([[0, 0.5, 0.75],[0, 0.5, 0.75]]))
    exp_jinc = array([[ 1. ,  0.18119175, -0.11953933],
                      [ 1. ,  0.18119175, -0.11953933]])
    nt.assert_array_almost_equal(exp_jinc, result, decimal=6)
    # rho is 0 valued standard python variable
    result = jinc_goodman(0)
    assert result==1.0
    # rho is non zero standard python variable
    result = jinc_goodman(0.25)
    assert (result - 0.72170284) <= 1e-4
    # Test the actual jinc function
    result = jinc(0, False) # without normalization
    assert result==0.5
    print("test_jinc() is successful!")

def _timingtest_jinc():
    """Timing test to see speed increase by using numba"""
    x = _np.linspace(-1,1,500)
    y = x.copy()
    X, Y = _np.meshgrid(x,y)
    rho = _np.hypot(X,Y)
    start = time.time()
    result1 = _jinc(rho)
    duration = time.time() - start
    print(duration*1000.0)
    start, duration = 0.0, 0.0
    start = time.time()
    result2 = jinc(rho)
    duration = time.time() - start
    print(duration*1000.0)
    # verify equivalence
    print(result1[0:5,0:5])
    print(result2[0:5,0:5])
    print("timingtest_jinc completed")

if __name__ == '__main__':
    import time
    import numpy.testing as nt
    from numpy import array,set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    # Automatic tests
    _test_rect()
    _test_jinc()
    # Manual tests
    #_timingtest_jinc()