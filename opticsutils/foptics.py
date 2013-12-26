# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          foptics.py
# Purpose:       Fourier Optics Utility Functions
#
# Author:        Indranil Sinharoy
#
# Created:       09/22/2012
# Last Modified: 04/10/2013
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
    r      : radius of the aperture
    z      : distance to the screen/image plane from the aperture
    rho    : radial coordinate in the screen/image plane
    normalization : 0 = None
                    1 = Peak value is 1.0 (default)
                    2 = Sum of the area under PSF = 1.0   
    Returns
    -------
    pattern : Fraunhoffer diffraction pattern (Airy pattern)
    """
    # The jinc() function used here is defined as jinc(x) = J_1(x)/x, jinc(0) = 0.5
    pattern = ((np.pi*r**2/(lamda*z))*2*_jinc(2*np.pi*r*rho/(lamda*z), normalize=False))**2
    if normalization==1:
        pattern = pattern/np.max(pattern)
    elif normalization==2:
        pattern = pattern/np.sum(pattern)
    return pattern


def getDirCosinesFromZenithAndAzimuthAngles(zenith_angle, azimuth_angle, atype='deg'):
    """Returns the direction cosines cos_A, cos_B & cos_C
    of the direction vector described by zenith and azimuth angles

    Parameters
    ----------
    zenith_angle  : (real) angle of the direction vector with respect
                    to the positive z-axis. 0 <= zenith_angle <= 180 degrees
    azimuth_angle : (real) angle of the direction vector with respect
                    to the positive x-axis. 0 <= azimuth_angle <= 360 degrees
    atype         : angle unit in degree (default) or radians
    
    Returns
    -------
    Tuple - (cos_A, cos_B, cos_C). These are the direction cosines, sometimes
            represented as alhpa, beta, gamma. 

    Note
    ----
    1. The angle of elevation is given as 90 - zenith_angle. 
    2. Direction cosines are defined as follows:
       alpha = cos A = sin(zenith_angle)cos(azimuth_angle)
       beta  = cos B = sin(zenith_angle)sin(azimuth_angle)
       gamma = cos C = cos(zenith_angle)
       
    Example
    -------
    >>>getDirCosinesFromZenithAndAzimuthAngles(20.0, 10.0)
    (0.33682408883346515, 0.059391174613884698, 0.93969262078590843)
    """
    if atype=='deg':
        zenith_angle = np.deg2rad(zenith_angle)
        azimuth_angle = np.deg2rad(azimuth_angle)
    cos_A = np.sin(zenith_angle)*np.cos(azimuth_angle)
    cos_B  = np.sin(zenith_angle)*np.sin(azimuth_angle)
    cos_C = np.cos(zenith_angle)
    return cos_A, cos_B, cos_C

def get_alpha_beta_gamma_set(alpha=None, beta=None, gamma=None, force_zero='none'):
    """Function to return the complete set of direction cosines alpha, 
    beta, and gamma, given a partial set.
    
    Parameters
    ----------
    alpha      : real, direction cosine, cos(A); see Notes 3.
    beta       : real, direction cosine, cos(B); see Notes 3.
    gamma      : real, direction cosine, cos(C); see Notes 3.
    force_zero : force a particular direction cosine to be zero, in
                 order to calculate the other two direction cosines
                 when the wave vector is restricted to a either x-y, 
                 or y-z, or x-z plane.
    Returns
    -------
    tuple - (alpha, beta, gamma)
    
    Notes
    -----
    1. The function doesn't check for the validity of alpha, beta, 
       and gamma. 
    2. If the function returns (None, None, None), most likely 2 
       out of the 3 input direction cosines passed are zeros, and
       `force_zero` is `none`. Please provide the appropriate 
       value for the parameter `force_zero`.
    3. A, B, C are angles that the wave vector k makes with x, y,
       and z axis respectively.
    """
    def f(x,y):
        return np.sqrt(1.0 - x**2 - y**2)
    if force_zero == 'none':
        if alpha and beta:
            return alpha, beta, f(alpha, beta)
        elif alpha and gamma:
            return alpha, f(alpha, gamma), gamma
        elif beta and gamma:
            return f(beta, gamma), beta, gamma
        else: # Error case
            return None, None, None
    elif force_zero == 'alpha':
        if beta:
            return 0.0, beta, f(beta, 0)
        else:
            return 0.0, f(gamma, 0), gamma
    elif force_zero == 'beta':
        if alpha:
            return alpha, 0.0, f(alpha,0)
        else:
            return f(gamma, 0), 0.0, gamma
    else:  # force_zero='g'
        if alpha:
            return alpha, f(alpha,0), 0.0
        else:
            return f(beta, 0), beta, 0.0 

# ---------------------------
#   TEST FUNCTIONS
# ---------------------------

def _test_airyPattern():
    """Test airyPattern function against a known and verified (several times) output
    So, before doubting and changing the values of the `expIntensity`, be triple sure
    about what you are doing!!!    
    """
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

def _test_getDirCosinesFromZenithAndAzimuthAngles():
    """Test getDirCosinesFromZenithAndAzimuthAngles function"""
    a, b, g = getDirCosinesFromZenithAndAzimuthAngles(20.0, 10.0)
    exp_array = np.array([0.33682408883, 0.05939117461, 0.93969262078])
    nt.assert_array_almost_equal(np.array([a, b, g]), exp_array, decimal=8)
    a, b, g = getDirCosinesFromZenithAndAzimuthAngles(90.0, 0.0)
    exp_array = np.array([1.0, 0.0, 0.0])
    nt.assert_array_almost_equal(np.array([a, b, g]), exp_array, decimal=8)
    print("test_getDirCosinesFromZenithAndAzimuthAngles() is successful")
    
def _test_get_alpha_beta_gamma_set():
    """Test get_alpha_beta_gamma_set() function"""
    a, b, g = get_alpha_beta_gamma_set(0, 0, 1)     
    exp_array = np.array((None, None, None))
    nt.assert_array_equal(np.array((a, b, g)), exp_array) 
    a, b, g = get_alpha_beta_gamma_set(0, 0, 1, 'alpha')
    exp_array = np.array((0.0, 0.0, 1.0))
    nt.assert_array_almost_equal(np.array((a, b, g)), exp_array, decimal=8)
    a, b, g = get_alpha_beta_gamma_set(0, 0, 0.5, 'alpha')
    exp_array = np.array((0.0, 0.86602540378, 0.5)) 
    nt.assert_array_almost_equal(np.array((a, b, g)), exp_array, decimal=8)     
    a, b, g = get_alpha_beta_gamma_set(0, 0.0593911746139, 0.939692620786)
    exp_array = np.array((0.33682408883, 0.05939117461, 0.93969262078)) 
    nt.assert_array_almost_equal(np.array((a, b, g)), exp_array, decimal=8)   
    a, b, g = get_alpha_beta_gamma_set(0.33682408883320675, 0.0593911746139,0)
    exp_array = np.array((0.33682408883, 0.05939117461, 0.93969262078)) 
    nt.assert_array_almost_equal(np.array((a, b, g)), exp_array, decimal=8) 
    print("test_get_alpha_beta_gamma_set() is successful")

if __name__ == '__main__':
    import numpy.testing as nt
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    _test_airyPattern()
    _test_getDirCosinesFromZenithAndAzimuthAngles()
    _test_get_alpha_beta_gamma_set()