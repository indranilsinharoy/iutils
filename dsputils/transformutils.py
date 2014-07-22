#--------------------------------------------------------------------------
# Name:          transformutils.py
# Purpose:       collection of utilities for transformations based
#                calculations such as FFT, etc.
#
# Author:        Indranil Sinharoy
#
# Created:       11/20/2013
# Last Modified: 11/20/2013
# Copyright:     (c) Indranil Sinharoy 2013
# Licence:       MIT License
#--------------------------------------------------------------------------
"""Collection of utilities for transformations used in signal processing
such as for FFT calculations, etc.
"""
from __future__ import division, print_function
import warnings
import numpy as _np
import math as _math
import scipy.special as _sps


def fft_freq_bin(N, sides='full', order=0, Fs=1):
    """Returns an array of frequency indices (bins) in cycles per unit.
    If `Fs=1`, the returned vector represents the normalized frequency

    Parameters
    ----------
    N : integer
        number of samples used to generate the indices for
        -0.5*Fs <= f*Fs < 0.5*Fs .
    sides : string
        representing the desired region of the frequency index to be
        returned.

        Options are:

        - full = ``N`` frequency indices between [-0.5, 0.5) are returned;
        - pos  = ``(N+1)//2`` frequency indices between [0, 0.5) are returned;
        - neg  = ``N//2`` frequency indices between [-0.5, 0) are returned;
        - both =  a tuple (neg_freq_indices, pos_freq_indices) is returned.
                  The returned vectors may be combined using
                  `np.hstack((fn, fp))`, for example.
    order : integer
        determines the way the frequency indices are arranged when `sides=full`
         0 = DC, pos_frequencyIndices, neg_frequencyIndices
         1 = neg_frequencyIndices, DC, pos_frequencyIndices
    Fs : float
        Sampling frequency (or 1/sample_spacing) its unit being samples/cycle

    Returns
    -------
    fc : ndarray or tuple of ndarrays
        frequency index vector, or a tuple of frequency index vectors
        (see above).

    Notes
    -----
    The default behavior of this function is similar to Numpy's
    ``fft.fftfreq`` function.

    Examples
    --------
    >>> fft_freq_bin(4)
    array([ 0.  ,  0.25, -0.5 , -0.25])
    >>> fft_freq_bin(5)
    array([ 0. ,  0.2,  0.4, -0.4, -0.2])
    >>> fft_freq_bin(4, order=1)
    array([-0.5 , -0.25,  0.  ,  0.25])
    >>> fft_freq_bin(4, 'both')
    (array([-0.5 , -0.25]), array([ 0.  ,  0.25]))
    >>> fft_freq_bin(4, order=1, Fs=10)
    array([-5. , -2.5,  0. ,  2.5])

    Reference: IPython notebook, "Understanding FFT frequency index"
    """
    if N%2: # odd
        fp = _np.arange(0, 0.5, 1.0/N)
        fn = -1.0*fp[-1:0:-1]
    else:
        fp = _np.arange(0, 0.5, 1.0/N)
        fn = _np.arange(-0.5, 0, 1.0/N)
    if sides == 'both':
        fc = (fn*Fs,fp*Fs)
    elif sides == 'pos':
        fc = fp*Fs
    elif sides == 'neg':
        fc = fn*Fs
    else: # 'full'
        if order:
            fc = _np.hstack((fn, fp))*Fs
        else:
            fc = _np.hstack((fp, fn))*Fs
    return fc




def _test_fft_freq_bin():
    """Test the fft_freq_bin() function"""
    freqInd = fft_freq_bin(4)  # test for Even number of bins
    nt.assert_equal(_np.array([ 0., 0.25, -0.5, -0.25]), freqInd)
    freqInd = fft_freq_bin(5)  # test for Odd number of bins
    nt.assert_equal(_np.array([ 0. ,  0.2,  0.4, -0.4, -0.2]), freqInd)
    freqInd = fft_freq_bin(4, 'both')
    nt.assert_equal(_np.array([-0.5 , -0.25]), freqInd[0])
    nt.assert_equal(_np.array([ 0.  ,  0.25]), freqInd[1])
    freqInd = fft_freq_bin(4, order=1, Fs=10)
    nt.assert_equal(_np.array([-5. , -2.5,  0. ,  2.5]), freqInd)
    print("fft_freq_bin test successful")


if __name__ == '__main__':
    import time
    import numpy.testing as nt
    from numpy import array,set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    # Automatic tests
    _test_fft_freq_bin()