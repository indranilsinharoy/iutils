# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------------------
# Name:          signals.py
# Purpose:       collection of signals/functions. Most of these are either
#                not available directly from scipy.signal or scipy.special,
#                or are in a different form then my requirements
#
# Author:        Indranil Sinharoy
#
# Created:       09/22/2012
# Last Modified: 12/31/2013
# Copyright:     (c) Indranil Sinharoy 2012, 2013
# Licence:       MIT License
#-----------------------------------------------------------------------------------------
"""Collection of signals. Most of these are either not available directly
from scipy.signal or scipy.special or may have slightly different definition.
"""
from __future__ import division, print_function
import warnings as _warnings
import numpy as _np
import math as _math
import scipy.special as _sps

# Import jit and double from numba if available.
import iutils as _iu
_hasNumba = _iu._hasNumba
if _hasNumba:
    from numba import jit, double

def rect(x, a=1.0, sharp_transition=True):
    """One dimensional rectangular function of width ``a``

    Usage: ``rect(x [, a, sharp_transition]) -> y``

    Parameters
    ----------
    x : ndarray
        input vector
    a : float, optional
        width of the rect function  (Default ``a`` = 1.0)
    sharp_transition : bool, optional
        If True (by default), the function forces the value to be 1.0 at
        the edges.

    Returns
    -------
    y : ndarray
        the output vector

    Notes
    -----
    The `rect` function is defined as:

    ::

        rect(x/a) =  1 for abs(x) < a/2
                   0.5 for abs(x) = a/2
                     0 for abs(x) > a/2

    Using the above definition, another form of the `rect` function is

    ::

        rect(ax) =   1 for abs(x) < 1/2a
                   0.5 for abs(x) = 1/2a
                     0 for abs(x) > 1/2a

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
    """Two dimensional rectangular function of width ``a``, height ``b``

    Usage: ``rect(x , y [, a, b, sharp_transition]) -> z``

    Parameters
    ----------
    X : 2-D ndarray
        Grid of x-corodinate, usually obtained from meshgrid.
    Y : 2-D ndarray
        Grid of y-corodinate, usually obtained from meshgrid.
    a : float.
        Width (along x-axis) of the rect function  (Default ``a`` = 1.0)
    b : float.
        Width (along y-axis) of the rect function  (Default ``b`` = 1.0)
    sharp_transition : bool
        If True (by default), the function forces the value to be 1.0
        at the edges.

    Returns
    -------
    z : 2-D ndarray
        The 2-D rectangular window.

    Notes
    -----

    ``rect2d(x,y) = rect(x)*rect(y)``

    Given two vectors x and y, you can create a 2D rect as follows:
    ::

        X, Y = np.meshgrid(x, y)         # create the 2D coordinates
        z = rect(X/a)*rect(Y/b)          # rectangular function

    Examples
    --------

    References
    ----------
    "Fourier Analysis and Imaging," Ronald N. Bracewell
    """
    z = rect(X/a, sharp_transition)*rect(Y/b, sharp_transition)
    return z

def tri(x):
    """One dimensional triangle function.

    Usage: ``tri(x) -> y``

    Parameters
    ----------
    x : ndarray
        input vector

    Returns
    -------
    y : ndarray
        output vector

    Notes
    -----
    ::

        tri(x) =  1 - abs(x) for abs(x) <= 1.0;
                  0 otherwise
    """
    y = _np.zeros(_np.shape(x))
    y[_np.abs(x)<1.0]=1.0-_np.abs(x[_np.abs(x)<1.0])
    return y

def circR(r):
    """Circular function that creates a disk of unit radius as defined
    in "Introduction to Fourier Optics", by Goodman.

    Parameters
    ----------
    r : ndarray
        For example, a 2-D grid generated using ``r = np.hypot(xx, yy)``
        where ``xx`` and ``yy`` may be created using ``meshgrid(x, y)``.
        To specify a radius ``R`` different from unity, scale the
        parameter ``r`` as ``r/R``.

    Returns
    -------
    circ : ndarray

    Examples
    --------
    To create a circular array of unit radius:

    >>> x = np.linspace(-2, 2, 200); y = x
    >>> X, Y = np.meshgrid(x,y)
    >>> Z = circR(np.sqrt(X**2 + Y**2))

    Notes
    -----
    The circle function defined in Bracewell's "Fourier Analysis and
    Imaging Book", ``rect(r)``, defines a circle of unit diameter as
    opposed to unit radius. It is implemented in this module as
    ``circD(r)``

    See Also
    --------
    circD(r)
    """
    out = _np.zeros(_np.shape(r))
    out[_np.abs(r)<1.0] = 1.0
    out[_np.abs(r)==1.0] = 0.5
    return out

def circD(r):
    """Circular function that creates a disk of unit diameter as defined
    in "Fourier Analysis and Imaging", by Bracewell.

    Parameters
    ----------
    r : ndarray
        For example, a 2-D grid generated using ``r = np.hypot(xx, yy)``
        where, ``xx`` and ``yy`` may be created using ``meshgrid(x, y)``.
        To specify a diameter `D` different from unity, scale the
        parameter ``r`` as ``r/D`` where, ``D`` is the diameter.

    Returns
    -------
    circ : ndarray

    Examples
    --------
    To create a circular array of unit diameter:

    >>> x = np.linspace(-2, 2, 200); y = x
    >>> X, Y = np.meshgrid(x,y)
    >>> Z = circD(np.sqrt(X**2 + Y**2))

    See Also
    --------
    circR(r)
    """
    out = _np.zeros(_np.shape(r))
    out[_np.abs(r) < 0.5] = 1.0
    out[_np.abs(r)== 0.5] = 0.5
    return out

def jinc(x, normalize=True):
    """Evaluate the jinc function in the input corodinate ``x``

    Parameters
    ----------
    x : a single point or an ndarray
        the input corodinate
    normalize : bool, optional
        if ``True`` (default), the normalized jinc (``jinc(0)=1``)
        is returned

    Returns
    -------
    jincx : a single value or an ndarray
        The jinc function evaluated at ``x``.

    Notes
    -----
    The jinc function is defined as [#]_ :math:`J_1(x)/x`,
    where, :math:`J_1(x)` is a Bessel function of the first kind, such
    that ``jinc(0) = 0.5``. This function, by default, returns the
    normalized jinc function that is obtained by multiplying by 2, such
    that ``jinc(0) = 1``

    References
    ----------
    .. [#] Jinc Function, by Eric Weisstein, MathWorld. http://mathworld.wolfram.com/JincFunction.html
    """
    # This is the interface function. The actual function is implemented in
    # _jinc()
    nFactor = 2.0 if normalize else 1.0
    numpyArray = isinstance(x, _np.ndarray)
    if not numpyArray:
        result = 0.5 if x==0.0 else _sps.j1(x)/(x)
    elif _hasNumba and numpyArray and False:  # For now the numba implementation
    # is disabled as it is actually slower than the vectorized implementation.
        if x.ndim==2:
            fast_jinc = jit(double[:,:](double[:,:]))(_jinc2)
        elif x.ndim==1:
            fast_jinc = jit(double[:](double[:]))(_jinc2)
        result = fast_jinc(x)
    else:
        result = _jinc(x)    # Numba not present
    return nFactor*result

def jinc_goodman(rho, normalize=True):
    """Calculate the jinc function over a radial grid ``rho``

    Parameters
    ----------
    rho : ndarray
        the radial coordinate grid, which may be obtained from the
        cartesian coordinates as ``rho = np.hypot(X, Y)``
    normalize : bool, optional
        if ``False``, the values are scaled by :math:`\pi` such that
        the value at the origin is :math:`\pi`.

    Returns
    -------
    jincr : ndarray
        the jinc evaluated at radial coordinates ``r``

    Notes
    -----
    1. The :math:`jinc(\\rho)`, which is the Hankel transform of a circ
       function of unit-radius, is defined as [#]_

       .. math::
            2 \\frac{J_1(2 \pi \\rho) }{ 2 \pi \\rho}

       such that ``jinc(0) = 1.0``.

    2. If ``normalize`` is ``False`` then the values correspond to the
       un-normalized jinc function definition:
       ``jinc(rho) = J_1(2pi*rho)/rho`` and ``jinc(0) = pi``.
    3. The ``jinc_bracewell()`` and the ``jinc_goodman()`` (un-normalized)
       are related by the similarity theorem of the Hankel Transform.

    References
    ----------
    .. [#] "Introduction to Fourier Optics", Joseph Goodman, 3rd Edition,
           Chapter-2, page-15.

    See Also
    --------
    jinc_bracewell() :
        evaluates the un-normalized (``jinc(0) = pi``) jinc function.
    jinc() :

    """
    ji =  jinc(2.0*_math.pi*rho, True)
    if normalize:
        return ji
    else:
        return ji*_math.pi

def jinc_bracewell(rho):
    """Calculate the jinc function as defined in Bracewell's "Fourier
    Analysis and Imaging" over a radial grid ``rho``

    Parameters
    ----------
    rho : ndarray
        the radial coordinate grid, which may be obtained from the
        cartesian coordinates as ``rho = np.hypot(X, Y)``

    Returns
    -------
    jincr : ndarray

    Notes
    -----
    1. The jinc ``jinc(rho)``, which is the Hankel Transform of a circ
       function of unit-diameter, is defined as [#]_:

       .. math::
            \\frac{J_1(\\pi \\rho)}{2 \\rho}

       such that ``jinc(0)=pi/4``.

    2. The ``jinc_bracewell()`` and the ``jinc_goodman()`` (un-normalized)
       are related by the similarity theorem of the Hankel Transform.

    References
    ----------
    ..[#] Fourier Analysis and Imaging, Ronald Bracewell

    See Also
    --------
    jinc_goodman() :
        evaluates the un-normalized jinc function
    jinc()
    """
    return (_math.pi/2.0)*jinc(_math.pi*rho, False)

def _jinc(x):
    """Internal implementation of the `jinc` function.
    The `jinc(x)` is defined as `J_1(x)/2x` and `jinc(0)=0.5`.
    """
    mask = x != 0.0
    result = 0.5*_np.ones(x.shape)
    result[mask] = _sps.j1(x[mask])/(x[mask])
    return result

def _jinc2(x):
    """Internal implementation of the jinc function for use with
    Numba @jit decorator. The `jinc(x)` is defined as `J_1(x)/2x` and
    `jinc(0)=0.5`.
    NOTE: jinc(0) as not been implemented yet...
    """
    # TODO !!! implement jinc(0) = 0.5
    M, N = x.shape
    result = _np.empty((M,N))
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        for i in range(M):
            for j in range(N):
                result[i,j] = _sps.j1(x[i,j])/(x[i,j])
    return result

def dirichlet(x, n):
    """Returns the Dirichlet or periodic sinc function.

    Parameters
    ----------
    x : ndarray
        an ndarray of input elements
    n : non-zero integer

    Returns
    -------
    d : ndarray
        ``d``, having shape as ``x``, is the Dirichlet function of ``x``

    Notes
    -----
    The Dirichlet function is defined as:

    .. math::

        D(x) = \left \{ \\begin{array}{cc}
        \\frac{sin(N x/2)}{N sin(x/2)}   & x \\neq 2\pi k, \;
        k=0, \pm 1, \pm 2, \pm 3, \dots \\\
        (-1)^{k(N-1)}   & x = 2\pi k, \; k=0, \pm 1, \pm 2, \pm 3, \dots
        \end{array}\\right .\


    This function has period :math:`2\pi` for odd ``n``, and period of
    :math:`4\pi` for even ``n``. Its peak value is 1, and its minimum
    value is -1 for ``n`` even. The magnitude of this function is
    ``(1/n)`` times the magnitude of the discrete-time Fourier transform
    of the n-point rectangular window.
    """
    d = _np.zeros(x.shape, dtype=x.dtype)
    k = x//(2*_np.pi)
    mask_c2 = x/(2*_np.pi) - k == 0
    mask_c1 = _np.logical_not(mask_c2)
    d[mask_c1] = _np.sin(0.5*n*x[mask_c1])/(n*_np.sin(0.5*x[mask_c1]))
    d[mask_c2] = (-1.0)**(k[mask_c2]*(n-1))
    return d

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
    nt.assert_equal(jinc_goodman(_np.ndarray([0])),1.0, err_msg="Expecting 1",
                    verbose=True)
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
    result = jinc_goodman(0, False)
    assert (result - 3.14159265359) <= 1e-4
    # rho is non zero standard python variable
    result = jinc_goodman(0.25)
    assert (result - 0.72170284) <= 1e-4
    # Test the actual jinc function
    result = jinc(0, False) # without normalization
    assert result==0.5
    # Test the jinc_bracewell function
    result = jinc_bracewell(0)
    assert (result - _math.pi/4) <= 1e-4
    # Compare with Table 9-4, Fourier Analysis & Imaging, Bracewell
    result = jinc_bracewell(_np.array([[0.0, 0.01, 0.02, 0.03],
                                       [0.1, 0.11, 0.12, 0.13]]))
    exp_jinc = array([[0.7854, 0.7853, 0.7850, 0.7845],
                      [0.7757, 0.7737, 0.7715, 0.7691]])
    nt.assert_array_almost_equal(exp_jinc, result, decimal=4)
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

def _test_dirichlet():
    pass
    # TODO !!!

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