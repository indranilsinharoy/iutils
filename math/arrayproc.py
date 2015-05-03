# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          arrayproc.py
# Purpose:       Array Processing Utilities
#
# Author:        Indranil Sinharoy
#
# Created:       09/12/2014
# Last Modified:
# Copyright:     (c) Indranil Sinharoy 2014
# License:       MIT License
#-------------------------------------------------------------------------------
"""Utility functions for array (mostly Numpy arrays) processing. All examples
in docstrings assume that the module has been imported as ``apu``
"""

from __future__ import division, print_function
import numpy as _np


def rescale(arr, low=0, high=1, axis=None):
    """Return rescaled array, with values scaled between ``low`` and ``high``

    Parameters
    ----------
    arr : ndarray
        numpy array to be rescaled
    low : float (array scalar) or ndarray, optional
        specifies the lower value(s) of the rescaled array
    high : float (array scalar) or ndarray, optional
        specifies the upper value(s) of the rescaled array
    axis : integer or ``None``, optional
        if ``None``, the whole array is globally rescaled between ``low`` and
        ``high``;
        if ``0``, rescaling is individually done along the first dimension.
        i.e. if ``arr.ndim = 2``, then each column is individually rescaled
        if ``1`` rescaling is individually done along the second dimension.
        i.e. if ``arr.ndim = 2``, then each row is individually rescaled.

    Returns
    -------
    arrRescaled : ndarray
        the rescaled array

    Examples
    --------
    >>> a = np.arange(9).reshape(3,3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    >>> apu.rescale(a, low=10, high=26)
    array([[ 10.,  12.,  14.],
           [ 16.,  18.,  20.],
           [ 22.,  24.,  26.]])

    >>> apu.rescale(a, low=10, high=26, axis=0)
    array([[ 10.,  10.,  10.],
           [ 18.,  18.,  18.],
           [ 26.,  26.,  26.]])

    >>> apu.rescale(a, low=10, high=26, axis=1)
    array([[ 10.,  18.,  26.],
           [ 10.,  18.,  26.],
           [ 10.,  18.,  26.]])

    See Also
    --------
    normalize(),
    """
    kdflag = True if axis is not None else False
    arrMax = _np.max(arr, axis=axis, keepdims=kdflag)
    arrMin = _np.min(arr, axis=axis, keepdims=kdflag)
    arrScaled = (arr - arrMin)/(arrMax - arrMin)
    arrRescaled = low + arrScaled*(high - low)
    return arrRescaled

def normalize(arr, axis=None):
    """Return normalized array such that the sum of elements equals unity

    Parameters
    ----------
    arr : ndarray
        numpy array
    axis : integer or None, optional
        if ``None``, the array is globally normalized such that the sum of all
        the elements in the array equals one
        if ``0``, then each column is normalized individually
        if ``1``, then each row is normalized individually

    Returns
    -------
    arrNorm : ndarray
        normalized array

    Examples
    --------
    >>> a = np.arange(9).reshape(3,3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    >>> apu.normalize(a)
    array([[ 0.        ,  0.02777778,  0.05555556],
           [ 0.08333333,  0.11111111,  0.13888889],
           [ 0.16666667,  0.19444444,  0.22222222]])
    >>> np.sum(apu.normalize(a))
    1.0

    >>> apu.normalize(a, axis=0)
    array([[ 0.        ,  0.08333333,  0.13333333],
           [ 0.33333333,  0.33333333,  0.33333333],
           [ 0.66666667,  0.58333333,  0.53333333]])
    >>> np.sum(apu.normalize(a, axis=0), axis=0)
    array([ 1.,  1.,  1.])

    >>> apu.normalize(a, axis=1)
    array([[ 0.        ,  0.33333333,  0.66666667],
           [ 0.25      ,  0.33333333,  0.41666667],
           [ 0.28571429,  0.33333333,  0.38095238]])
    >>> np.sum(apu.normalize(a, axis=1), axis=1)
    array([ 1.,  1.,  1.])

    See Also
    --------
    rescale(),
    """
    kdflag = True if axis is not None else False
    arrSum = _np.sum(arr, axis=axis, keepdims=kdflag)
    arrNorm = arr/arrSum
    return arrNorm