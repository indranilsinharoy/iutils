# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          foptics.py
# Purpose:       Fourier Optics Utility Functions
#
# Author:        Indranil Sinharoy
#
# Created:       22/09/2012
# Last Modified:
# Copyright:     (c) Indranil Sinharoy 2012, 2013
# Licence:       MIT License
#-------------------------------------------------------------------------------
from __future__ import division, print_function
import numpy as np
from iutils.signalutils.signals import jinc as _jinc


def airyPattern(lamda, r, z, rho, normalization=1):
    """Returns the Fraunhoffer intensity diffraction pattern for a circular aperture.
    This is also known as the Airy pattern.
    
    Parameters
    ---------
    lamda  : wavelength in physical units (same units that of `r`, `z`, `rho`)
    r      : raduis of the aperture
    z      : distance to the screen/image plane from the aperture
    rho    : radial coordinate in the screen/image plane
    normalization : 0 = None
                    1 = Peak value is 1.0 (default)
                    2 = Sum of the area under PSF = 1.0   
    Returns
    -------
    pattern : Fraunhoffer diffraction pattern (Airy pattern)
    """
    pattern = ((np.pi*r**2/(lamda*z))*_jinc(2*np.pi*r*rho/(lamda*z)))**2
    if normalization==1:
        pattern = pattern/np.max(pattern)
    elif normalization==2:
        pattern = pattern/np.sum(pattern)
    return pattern


# ---------------------------
#   TEST FUNCTIONS
# ---------------------------

def _test_airyPattern():
    """Test airyPattern function against a known and verified output"""
    x = np.linspace(-1,1,5)
    y = x.copy()
    X, Y = np.meshgrid(x,y)
    radius = 5.0
    wavelength = 550e-6
    z = 25.0
    rho = np.hypot(X,Y)
    I = airyPattern(wavelength,radius,z,rho, normalization=0)
    expIntensity= np.array([[1.37798644e-03, 8.36156468e-04, 3.56139554e-05, 8.36156468e-04, 1.37798644e-03],
                            [8.36156468e-04, 1.77335279e-05, 4.89330106e-02, 1.77335279e-05, 8.36156468e-04],
                            [3.56139554e-05, 4.89330106e-02, 3.26267914e+07, 4.89330106e-02, 3.56139554e-05],
                            [8.36156468e-04, 1.77335279e-05, 4.89330106e-02, 1.77335279e-05, 8.36156468e-04],
                            [1.37798644e-03, 8.36156468e-04, 3.56139554e-05, 8.36156468e-04, 1.37798644e-03]])
    nt.assert_array_almost_equal(expIntensity, I, decimal=2)
    # check for normalization    
    I = airyPattern(wavelength,radius,z,rho)
    nt.assert_almost_equal(np.max(I), 1.0, decimal=6)
    I = airyPattern(wavelength,radius,z,rho, normalization=2)
    nt.assert_almost_equal(np.sum(I), 1.0, decimal=6)
    print("test_airyPatten() is successful")


if __name__ == '__main__':
    import numpy.testing as nt
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    _test_airyPattern()
