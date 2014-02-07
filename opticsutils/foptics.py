# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          foptics.py
# Purpose:       Fourier Optics Utility Functions
#
# Author:        Indranil Sinharoy
#
# Created:       09/22/2012
# Last Modified: 02/07/2014
# Copyright:     (c) Indranil Sinharoy 2012, 2013
# Licence:       MIT License
#-------------------------------------------------------------------------------
from __future__ import division, print_function
import numpy as np
from iutils.signalutils.signals import jinc as _jinc



def fresnelNumber(aperture_r, observ_plane_dist, wave_length=550e-6):
    """calculate the fresnel number, assuming circular aperture
    
    Parameters
    ----------
    aperture_r : float
        The radius of the aperture in units of length (usually mm)
    observ_plane_dist : float
        The distance of the observation plane from the aperture. This is equal to the focal length of the lens for infinite conjugate, or the image plane distance, in the same units of length as `aperture_r`
    wave_length : float
        The wavelength of light (default=550e-6, note that the default's unit is mm)
    
    Returns
    -------
    fresnel_number : float
    
    Note
    ----
    1. From the Huygens-Fresnel principle perspective, the Fresnel number represents the number of Fresnel zones in the aperture opening [Principles of Optics, Born and Wolf, 2011]
    """
    return (aperture_r**2.0)/(wave_length*observ_plane_dist)
    
def dlSpotSize(aperture_r, zi, wave_length=550e-6):
    """calculate the diffraction limited spot size, assuming circular aperture
    
    Parameters
    ----------
    aperture_r : float
        The radius of the aperture in units of length (usually mm)
    zi : float
        Image distance (or the distance of the observation plane) in the same units of length as `aperture_r`. For objects at infinity, `zi` is the focal length of the lens.
    wave_length : float
        The wavelength of light (default=550e-6, note that the default's unit is mm)
    
    Returns
    -------
    spot_size : float
        The diffraction limited spot size given as 2.44*lamda*f/#
    """
    return 1.22*wave_length*(zi/aperture_r)
    
def airyPattern(lamda, radius, zxp, rho, normalization=1):
    """Returns the Fraunhoffer intensity diffraction pattern for a circular aperture.
    This is also known as the Airy pattern.
    
    Parameters
    ---------
    lamda : Float 
        wavelength in physical units (same units that of `radius`, `zxp`, `rho`)
    radius : Float
        radius of the aperture
    zxp : Float 
        distance to the screen/image plane from the aperture/Exit pupil
    rho : ndarray
        radial coordinate in the screen/image plane (such as constructed using `meshgrid`)
    normalization : Integer (0 or 1 or 2)
        0 = None; 1 = Peak value is 1.0 (default); 2 = Sum of the area under PSF = 1.0   
    
    Returns
    -------
    pattern : ndarray
        Fraunhoffer diffraction pattern (Airy pattern)
    """
    # The jinc() function used here is defined as jinc(x) = J_1(x)/x, jinc(0) = 0.5
    pattern = ((np.pi*radius**2/(lamda*zxp))*2*_jinc(2*np.pi*radius*rho/(lamda*zxp), normalize=False))**2
    if normalization==1:
        pattern = pattern/np.max(pattern)
    elif normalization==2:
        pattern = pattern/np.sum(pattern)
    return pattern
    
def depthOfFocus(f_number_eff, wavelength=550e-6, first_zero=False, full=False):
    """Return the half diffractive optical focus depth, `delta_z`, in the image space.
    
    The total DOF is twice the returned value assuming symmetric intensity distribution about the point fo focus.
    
    Parameters
    ----------
    f_number_eff: float
        Effective F-number of the system, given by `f/D` for infinite conjugate imaging, and `zi/D` for finite conjugate imaging. Where `f` is the focal length, `zi` is the gaussian image distance, `D` is the entrance pupil diameter.
    wavelength : float
        The wavelength of light (default=550e-6, note that the default's unit is mm)
    first_zero : boolean
        Normally the DOF is the region between the focal point (max intensity) on the axis and a point along the axis where the intensity has fallen upto 20% of the max [Born and Wolf, Page 491, Handbook of Optical Systems. Aberration Theory, Vol-3, Gross, Page 126]. This is the region returned by default by this function. If `first_zero` is True, then the (half-) region returned is from the max intensity to the first zero along the axis.
        
    Return
    ------
    delta_z : float
        One-sided depth of focus (in the image space) in the units of the `wavelength`. The total DOF is twice the returned value assuming symmetric intensity distribution about the point fo focus.
    """
    if first_zero:
        delta_z = 8.0*wavelength*f_number_eff**2
    else:
        delta_z = (6.4*wavelength*f_number_eff**2)/np.pi
    if full:
        delta_z = delta_z*2.0
    return delta_z


def depthOfField(focal_length, f_number, obj_dist, wavelength=550e-6, first_zero=False):
    """Returns the diffraction based depth of field in the object space

    Parameters
    ----------
    focal_length : float
        Focal length of the lens in the length units (usually in mm)
    f_number: float
        F-number of the system, given by `f/D`, where `f` is the focal length, and `D` is the pupil diameter. It is not the effective f-number, as the effective f-number is calculated within the function based on the `focal_length` and the `obj_dist`.
    obj_dist : float
        Distance of the object plane from the lens in the same units as `focal_length`
    wavelength : float
        The wavelength of light in the same units of `focal_length` (default=550e-6, note that the default's unit is mm)
    first_zero : boolean
        Normally the DOF in the image space is the region between the focal point (max intensity) on the axis and a point along the axis where the intensity has fallen upto 20% of the max [Born and Wolf, Page 491, Handbook of Optical Systems. Aberration Theory, Vol-3, Gross, Page 126]. This is the region returned by this function in the default mode. If `first_zero` is `True`, then the region returned is from the max intensity to the first zeros along the axis on either side of the optical axis.
        
    Returns
    -------
    The function returns a 3-tuple of the following elements:

    dof_t : float
        total DOF range     
    dof_ext_f : float
        far extent of the plane in acceptable focus (criterion set by the state of `first_zero`)
    dof_ext_n : float
        near extent of the plane in acceptable focus
    """
    img_dist = obj_dist*focal_length/(obj_dist - focal_length) # geometric image point
    f_num_eff = (img_dist/focal_length)*f_number
    delta_z = depthOfFocus(f_num_eff, wavelength, first_zero)   # half diffraction DOF
    # far (w.r.t. lens) extent of the plane in acceptable focus    
    dof_ext_f =  (img_dist - delta_z)*focal_length/((img_dist - delta_z) - focal_length)
    # near (w.r.t. lens) extent of the plane in acceptable focus    
    dof_ext_n =  (img_dist + delta_z)*focal_length/((img_dist + delta_z) - focal_length)
    # total DOF extent    
    dof_t = dof_ext_f - dof_ext_n
    return dof_t, dof_ext_f, dof_ext_n


def geometricDepthOfField(focal_length, f_number, obj_dist, coc, grossExpr=False):
    """Returns the geometric depth of field value
    
    Parameters
    ----------
    focal_length : float 
        focal length in length units (usually in mm)
    f_number : float
        F/# of the optical system
    obj_dist : float
        distance of the object from the lens in focus in the same unit as `f`
    coc : float
        circle of confusion in the same length-unit as `f`
    grossExpr : boolean
        whether or not to use the expression from Gross' Handbook of Optical systems (see the notes). This is an approximation.
    
    Returns
    -------
    dof : float 
        length of the total depth of field zone in the same unit as `f`
    
    Notes
    -----
    The expression for the geometric depth of field is given as:
    
    $DOF_{\text{geom}} = \frac{2 f^2 N c u^2 }{f^4 - N^2 c^2 u^2 }$
    
    The expression for the geometric depth of filed using Gross Expression, which is derived from the expression given in 30.8 of Handbook of Optical Systems, vol 3, Gross is as follows:
    $DOF_{\text{geom}} = \frac{2 c N (u - f)^2}{f^2}$
    """
    if grossExpr:
        dof = (2.0*coc*f_number*(obj_dist-focal_length)**2)/focal_length**2
    else:
        dof = (2.0*focal_length**2*f_number*coc*obj_dist**2)/(focal_length**4 - (f_number*coc*obj_dist)**2)
    return dof

#----------------------------------
# Aberration calculation functions
#----------------------------------

def defocus(w020, aperture_r, zi):
    """Return the amount focus shift or defocus, delta_z 
    
    Parameters
    ----------
    w020 : float
        Wavefront error coefficient for defocus. w020 is the maximum wavefront error measured at the edge of the pupil.
    aperture_r : float
        The radius of the aperture in units of length (usually mm)
    zi : float
        Image distance (or the distance of the observation plane) in the same units of length as `aperture_r`. For objects at infinity, `zi` is the focal length of the lens.  
    
    Returns
    -------
    delta_z : float
        The amount of defocus along the optical axis corresponding to the given wavefront error.
        
    Note
    ----    
    The relation between the wavefront error and the defocus as derived using paraxial assumption and scalar diffraction theory. It is given as:
    
    `W020 = (delta_z*a^2)/(2*zi(zi + delta_z))` 
    
    which may also be approximated as `W020 = delta_z/8N^2`, where `N` is the f-number.
    
    See also w020FromDefocus().
    """
    return (2.0*zi**2*w020)/(aperture_r**2 - 2.0*zi*w020)

def w020FromDefocus(aperture_r, zi, delta_z, wave_length=1.0):
    """Return the maximum wavefront error corresponding to defocus amount `delta_z`  
    
    Parameters
    ----------
    aperture_r : float
        The radius of the aperture in units of length (usually mm)
    zi : float
        Image distance (or the distance of the observation plane) in the same units of length as `aperture_r`. For objects at infinity, `zi` is the focal length of the lens. 
    delta_z : float
        The amount of defocus/focus-shift along the optical axis. Generally `delta_z = z_a - zi` where `z_a` is the point of convergence of the actual/aberrated wavefront on the optical axis.
    wave_length : float
        The `wave_length` is used to specify a wave length if the coefficient w020 needs to be 'relative to the wavelength'
    
    Returns
    -------
    w020 : float
        Wavefront error coefficient for defocus. w020 is the maximum wavefront error, which is measured at the edge of the pupil.
        
    Note
    ----    
    The relation between the wavefront error and the defocus as derived using paraxial assumption and scalar diffraction theory. It is given as:
    
    `W020 = (delta_z*a^2)/(2*zi(zi + delta_z))` 
    
    which may also be approximated as `W020 = delta_z/8N^2`, where `N` is the f-number.
    
    See also defocus().
    """
    w020 = (delta_z*aperture_r**2)/(2.0*zi*(zi + delta_z))
    return w020/wave_length

def seidel_5(u0, v0, X, Y, wd=0, w040=0, w131=0, w222=0, w220=0, w311=0):
    """Computer wavefront OPD for first 5 Seidel wavefront aberration coefficients plus defocus.
    
    Parameters
    ----------
    u0, v0 : float
        normalized image plane coordinate along the u-axis and v-axis respectively
    X, Y : ndarray
        normalized pupil coordinate array (usually from meshgrid)
    wd, w040, w131, w222, w220, w311  : float
        defocus, spherical, coma, astigmatism, field-curvature, distortion aberration coefficients
        
    Returns
    -------
    w : ndarray
        wavefront OPD at the given image plane coordinate.
        
    Note
    ----
    This function is exactly implemented as is from 'Computational Fourier Optics', David Voelz
    """
    theta = np.arctan2(v0, u0)       # image rotation angle
    u0r = np.sqrt(u0**2 + v0**2)     # image height 
    # rotate pupil grid
    Xr = X*np.cos(theta) + Y*np.sin(theta)
    Yr = -X*np.sin(theta) + Y*np.cos(theta)
    rho2 = Xr**2 + Yr**2
    w = (  wd*rho2         +    # defocus
         w040*rho2**2      +    # spherical
         w131*u0r*rho2*Xr  +    # coma
         w222*u0r**2*Xr**2 +    # astigmatism
         w220*u0r**2*rho2  +    # field curvature
         w311*u0r**3*Xr     )   # distortion
    return w
    

#-------------------------------------
# Some ray optics helper functions
#-------------------------------------
# Don't move the following fuctions to another module ... these functions are also
# useful for Fourier Optics calculations too, such as direction cosine calculations.

def getDirCosinesFromZenithAndAzimuthAngles(zenith_angle, azimuth_angle, atype='deg'):
    """Returns the direction cosines cos_A, cos_B & cos_C
    of the direction vector described by zenith and azimuth angles

    Parameters
    ----------
    zenith_angle : Float 
        Angle of the direction vector with respect to the positive z-axis. 0 <= zenith_angle <= 180 degrees
    azimuth_angle : Float 
        Angle of the direction vector with respect to the positive x-axis. 0 <= azimuth_angle <= 360 degrees
    atype : String ('rad' or 'deg')
        Angle unit in degree (default) or radians
    
    Returns
    -------
    direction_cosines : Tuple (cos_A, cos_B, cos_C) 
        These are the direction cosines, sometimes represented as alhpa, beta, gamma. 

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
    
    See also `get_alpha_beta_gamma_set`.
    """
    if atype=='deg':
        zenith_angle = np.deg2rad(zenith_angle)
        azimuth_angle = np.deg2rad(azimuth_angle)
    cos_A = np.sin(zenith_angle)*np.cos(azimuth_angle)
    cos_B  = np.sin(zenith_angle)*np.sin(azimuth_angle)
    cos_C = np.cos(zenith_angle)
    return (cos_A, cos_B, cos_C)

def get_alpha_beta_gamma_set(alpha=None, beta=None, gamma=None, force_zero='none'):
    """Function to return the complete set of direction cosines alpha, 
    beta, and gamma, given a partial set.
    
    Parameters
    ----------
    alpha : Float 
        Direction cosine, cos(A); see Notes 3.
    beta : Float 
        Direction cosine, cos(B); see Notes 3.
    gamma : Float 
        Direction cosine, cos(C); see Notes 3.
    force_zero : String ('none' or 'alpha' or 'beta' or 'gamma')
        Force a particular direction cosine to be zero, in order to calculate the other two direction cosines when the wave vector is restricted to a either x-y, or y-z, or x-z plane.
    
    Returns
    -------
    direction_cosines: tuple - (alpha, beta, gamma)
    
    Notes
    -----
    1. The function doesn't check for the validity of alpha, beta, and gamma. 
    2. If the function returns (None, None, None), most likely 2 out of the 3 input direction cosines passed are zeros, and `force_zero` is `none`. Please provide the appropriate value for the parameter `force_zero`.
    3. A, B, C are angles that the wave vector k makes with x, y, and z axis respectively.
    
    See also `getDirCosinesFromZenithAndAzimuthAngles`.
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

def _test_fresnelNumber():
    """Test fresnelNumber function"""
    fresnelNum = fresnelNumber(10, 550e-6, 500)
    nt.assert_almost_equal(fresnelNum, 363.636363636, decimal=4)
    print("test_fresnelNumber is successful")

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
    I = airyPattern(wavelength, radius, z, rho, normalization=0)
    expIntensity= np.array([[1.37798644e-03, 8.36156468e-04, 3.56139554e-05, 8.36156468e-04, 1.37798644e-03],
                            [8.36156468e-04, 1.77335279e-05, 4.89330106e-02, 1.77335279e-05, 8.36156468e-04],
                            [3.56139554e-05, 4.89330106e-02, 3.26267914e+07, 4.89330106e-02, 3.56139554e-05],
                            [8.36156468e-04, 1.77335279e-05, 4.89330106e-02, 1.77335279e-05, 8.36156468e-04],
                            [1.37798644e-03, 8.36156468e-04, 3.56139554e-05, 8.36156468e-04, 1.37798644e-03]])
    nt.assert_array_almost_equal(expIntensity, I, decimal=2)
    # check for normalization    
    I = airyPattern(wavelength, radius, z, rho)
    nt.assert_almost_equal(np.max(I), 1.0, decimal=6)
    I = airyPattern(wavelength, radius, z, rho, normalization=2)
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

def _test_seidel_5():
    """Test seidel_5 function"""
    #TODO!!!    
    pass

def _test_dSpotSize():
    #TODO!!!
    pass

def _test_w020FromDefocus():
    #TODO!!!
    pass

def _test_depthOfFocus():
    """Test depthOfFocus(), which returns the diffraction based DOF"""
    wavelength = 500e-6  # mm  
    fnumber = [2, 2.8, 4.0, 5.6, 8.0, 10, 11.0, 16.0, 22.0]
    dz = [depthOfFocus(N, wavelength) for N in fnumber]
    exp_array = np.array((0.004074366543152521, 0.00798575842457894, 0.016297466172610083, 0.03194303369831576, 0.06518986469044033, 0.10185916357881303, 0.12324958793036377, 0.2607594587617613, 0.49299835172145506))
    nt.assert_array_almost_equal(np.array(dz), exp_array, decimal=6)
    print("test_depthOfFocus() is successful")

def _test_depthOfField():
    #TODO!!!
    pass

def _test_geometricDepthOfField():
    #TODO!!!
    pass

if __name__ == '__main__':
    import numpy.testing as nt
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    _test_fresnelNumber()    
    _test_airyPattern()
    _test_getDirCosinesFromZenithAndAzimuthAngles()
    _test_get_alpha_beta_gamma_set()
    _test_depthOfFocus()