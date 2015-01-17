# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          foptics.py
# Purpose:       Fourier Optics Utility Functions
#
# Author:        Indranil Sinharoy
#
# Created:       09/22/2012
# Last Modified: 07/09/2014
# Copyright:     (c) Indranil Sinharoy 2012, 2013, 2014
# Licence:       MIT License
#-------------------------------------------------------------------------------
"""foptics module contains some useful functions for Fourier Optics based
calculations.
"""
from __future__ import division, print_function
import math as _math
import numpy as _np
from iutils.signalutils.signals import jinc as _jinc
import iutils.opticsutils.imager as _imgr
#import warnings as _warnings
import collections as _co

def fresnel_number(r, z, wl=550e-6, approx=False):
    """calculate the fresnel number

    Parameters
    ----------
    r : float
        radius of the aperture in units of length (usually mm)
    z : float
        distance of the observation plane from the aperture. this is equal
        to the focal length of the lens for infinite conjugate, or the
        image plane distance, in the same units of length as ``r``
    wl : float, optional
        wavelength of light (default=550e-6 mm)
    approx : boolean, optional
        if ``True``, uses the approximate expression (default is ``False``)

    Returns
    -------
    fN : float
        fresnel number

    Notes
    -----
    1. The Fresnel number is calculated based on a circular aperture or a
       an unaberrated rotationally symmetric beam with finite extent
       [Zemax]_.
    2. From the Huygens-Fresnel principle perspective, the Fresnel number
       represents the number of annular Fresnel zones in the aperture
       opening [Wolf2011]_, or from the center of the beam to the edge in
       case of a propagating beam [Zemax]_.

    References
    ----------
    .. [Zemax] Zemax manual

    .. [Born&Wolf2011] Principles of Optics, Born and Wolf, 2011
    """
    if approx:
        return (r**2)/(wl*z)
    else:
        return 2.0*(_math.sqrt(z**2 + r**2) - z)/wl

def diffraction_spot_size(fnum=None, r=None, zi=None, wl=550e-6):
    """calculate the diffraction limited spot size, assuming circular
    aperture. Specify either ``fnum`` or ``r`` and ``zi``

    Parameters
    ----------
    fnum : float, optional
        F/# or the effective F/# of the system. If ``fnum`` is specified, then
        the spot size shall be calculated using the ``fnum`` and the wavelength
    r : float, optional
        radius of the aperture in units of length (usually mm). If ``fnum`` is
        also specified, then ``r`` is ignored.
    zi : float, optional
        image distance (or the distance of the observation plane) in the
        same units of length as ``r``. For objects at infinity, ``zi`` is
        the focal length of the lens. If ``fnum`` is also specified, then ``r``
        is ignored.
    wl : float, optional
        wavelength of light (default=550e-6 mm)

    Returns
    -------
    spot_size : float
        The diffraction limited spot size given as 2.44*lambda*f/#
    """
    spotSize = 2.44*wl*fnum if fnum else 1.22*wl*(zi/r)
    return spotSize

def effective_fnumber(fnum, m=None, zxp=None, f=None):
    """Calculate the effective F/#

    Parameters
    ----------
    fnum : float
        F/# (specified at infinite conjugates) of the lens
    m : float, optional
        magnification (defined as image_distance/object_distance)
    zxp : float, optional
        distance to the screen/image plane from the aperture/Exit pupil i.e.
        the image distance
    f : float, optional
        focal length

    Returns
    --------
    fnumEff : float
        The effective F/#

    Notes
    -----
    If ``m`` is given, then only the magnification parameter is used.
    """
    fnumEff = 0.0
    if m is None:
        fnumEff = (zxp/f)*fnum
    else:
        fnumEff = (1.0 - m)*fnum
    return fnumEff


def airy_pattern(wl, r, zxp, rho, norm=1):
    """Fraunhoffer intensity diffraction pattern for a circular aperture

    Parameters
    ----------
    wl : float
        wavelength in physical units (same unit of ``r``, ``zxp``, ``rho``)
    r : float
        radius of the aperture
    zxp : float
        distance to the screen/image plane from the aperture/Exit pupil
    rho : ndarray
        radial coordinate in the screen/image plane (such as constructed
        using ``meshgrid``)
    norm : integer (0 or 1 or 2), optional
        0 = None;
        1 = Peak value is 1.0 (default);
        2 = Sum of the area under PSF=1.0

    Returns
    -------
    pattern : ndarray
        Fraunhoffer diffraction pattern (Airy pattern)

    Examples
    --------
    This example creates an airy pattern for light wave of 0.55 microns
    wavelength diffracting through an unaberrated lens of focal length
    50 mm, and epd 20 mm. The airy pattern is generated on a spatial grid
    which extends between -5*lambda and 5*lambda along both X and Y

    >>> lamba = 550e-6
    >>> M, N = 513, 513            # odd grid for symmetry
    >>> dx = 10*lamba/N
    >>> dy = 10*lamba/M
    >>> x = (np.linspace(0, N-1, N) - N//2)*dx
    >>> y = (np.linspace(0, M-1, M) - M//2)*dy
    >>> xx, yy = np.meshgrid(x, y)
    >>> r = np.hypot(xx, yy)
    >>> w = 10                    # radius, in mm
    >>> z = 50                    # z-distance, in mm
    >>> ipsf = fou.airy_pattern(lamba, w, z, r, 1)
    """
    # The jinc() function used here is defined as jinc(x) = J_1(x)/x, jinc(0) = 0.5
    pattern = ((_np.pi*r**2/(wl*zxp))
               *2*_jinc(2*_np.pi*r*rho/(wl*zxp), normalize=False))**2
    if norm==1:
        pattern = pattern/_np.max(pattern)
    elif norm==2:
        pattern = pattern/_np.sum(pattern)
    return pattern

def depth_of_focus(effFNum, wavelength=550e-6, firstZero=False, full=False):
    """half diffractive optical focus depth, ``delta_z``, in the image
    space.

    The total DOF is twice the returned value assuming symmetric intensity
    distribution about the point fo focus.

    Parameters
    ----------
    effFNum: float
        effective F-number of the system, given by ``f/D`` for infinite
        conjugate imaging, and ``zi/D`` for finite conjugate imaging;
        ``f`` is the focal length, ``zi`` is the gaussian image distance,
        ``D`` is the entrance pupil diameter.
    wavelength : float, optional
        wavelength of light (default=550e-6, note that the default's
        unit is mm)
    firstZero : boolean, optional
        Normally the DOF is the region between the focal point (max
        intensity) on the axis and a point along the axis where the
        intensity has fallen upto 20% of the max (see [Wolf2011]_ Page 491,
        and [Gross2007]_ Page 126). This is the region returned by default
        by this function. If ``firstZero`` is True, then the (half-)
        region returned is from the max intensity to the first zero along
        the axis.

    Returns
    -------
    deltaZ : float
        one-sided depth of focus (in the image space) in the units of the
        ``wavelength``. The total DOF is twice the returned value assuming
        symmetric intensity distribution about the point of focus.

    References
    ----------
    .. [Gross2007] Handbook of Optical Systems. Aberration Theory, Vol-3
    """
    if firstZero:
        deltaZ = 8.0*wavelength*effFNum**2
    else:
        deltaZ = (6.4*wavelength*effFNum**2)/_np.pi
    if full:
        deltaZ = deltaZ*2.0
    return deltaZ

def depth_of_field(focalLength, fNumber, objDist, wavelength=550e-6, firstZero=False):
    """Returns the diffraction based depth of field in the object space

    Parameters
    ----------
    focalLength : float
        focal length of the lens in the length units (usually in mm)
    fNumber: float
        F-number of the system, given by ``f/D``, where ``f`` is the focal
        length, and ``D`` is the pupil diameter. It is not the effective
        f-number, as the effective f-number is calculated within the
        function based on the ``focalLength`` and the ``objDist``.
    objDist : float
        distance of the object plane from the lens in the same units
        as ``focalLength``
    wavelength : float
        the wavelength of light in the same units of ``focalLength``
        (default=550e-6, note that the default's unit is mm)
    firstZero : boolean, optional
        Normally the DOF is the region between the focal point (max
        intensity) on the axis and a point along the axis where the
        intensity has fallen upto 20% of the max (see [Wolf2011]_ Page 491,
        and [Gross2007]_ Page 126). This is the region returned by default
        by this function. If ``firstZero`` is True, then the (half)
        region returned is from the max intensity to the first zero along
        the axis.

    Returns
    -------
    dofTotal : float
        total DOF range
    dofFar : float
        far extent of the plane in acceptable focus (criterion set by the
        state of ``firstZero``)
    dofNear : float
        near extent of the plane in acceptable focus
    """
    imgDist = objDist*focalLength/(objDist - focalLength) # geometric image point
    effFNum = (imgDist/focalLength)*fNumber
    deltaZ = depth_of_focus(effFNum, wavelength, firstZero)   # half diffraction DOF
    # far (w.r.t. lens) extent of the plane in acceptable focus
    dofFar =  (imgDist - deltaZ)*focalLength/((imgDist - deltaZ) - focalLength)
    # near (w.r.t. lens) extent of the plane in acceptable focus
    dofNear =  (imgDist + deltaZ)*focalLength/((imgDist + deltaZ) - focalLength)
    # total DOF extent
    dofTotal = dofFar - dofNear
    return dofTotal, dofFar, dofNear

def geometric_depth_of_field(focalLength, fNumber, objDist, coc, grossExpr=False):
    """returns the geometric depth of field value

    Parameters
    ----------
    focalLength : float
        focal length in length units (usually in mm)
    fNumber : float
        F/# of the optical system
    objDist : float
        distance of the object from the lens in focus in the same unit
        as ``f``
    coc : float
        circle of confusion in the same length-unit as ``f``
    grossExpr : boolean
        whether or not to use the expression from Gross' Handbook of
        Optical systems (see the notes). This is an approximation.

    Returns
    -------
    dof : float
        length of the total depth of field zone in the same unit as ``f``

    Notes
    -----
    This function is implemented in the module ``iutils.opticsutils.imager``
    """
    gdof = _imgr.geometric_depth_of_field(focalLength, fNumber, objDist, coc,
                                          grossExpr)
    return gdof



#----------------------------------
# Aberration calculation functions
#----------------------------------

def defocus(w020, radius, zi):
    """Return the amount focus shift or defocus, deltaZ

    Parameters
    ----------
    w020 : float
        Wavefront error coefficient for defocus. w020 is the maximum
        wavefront error measured at the edge of the pupil.
    radius : float
        The radius of the aperture in units of length (usually mm)
    zi : float
        Image distance (or the distance of the observation plane) in the
        same units of length as ``radius``. For objects at infinity,
        ``zi`` is the focal length of the lens.

    Returns
    -------
    deltaZ : float
        The amount of defocus along the optical axis corresponding to the
        given wavefront error.

    Notes
    -----
    The relation between the wavefront error and the defocus as derived
    using paraxial assumption and scalar diffraction theory.
    It is given as:

    .. math::

        W_{020} = \\frac{\\delta_z a^2}{2 z_i(z_i + \delta_z)}

    which may also be approximated as :math:`W_{020} = \delta_z/8N^2`, where
    `N` is the f-number.

    See Also
    --------
    w020FromDefocus()
    """
    return (2.0*zi**2*w020)/(radius**2 - 2.0*zi*w020)

def w020_from_defocus(radius, zi, deltaZ, waveLength=1.0):
    """Return the maximum wavefront error corresponding to defocus amount
    `deltaZ`

    Parameters
    ----------
    radius : float
        The radius of the aperture in units of length (usually mm)
    zi : float
        Image distance (or the distance of the observation plane) in the
        same units of length as ``radius``. For objects at infinity,
        ``zi`` is the focal length of the lens.
    deltaZ : float
        The amount of defocus/focus-shift along the optical axis.
        Generally ``deltaZ = za - zi`` where ``za`` is the point of
        convergence of the actual/aberrated wavefront on the optical axis.
    waveLength : float
        The ``waveLength`` is used to specify a wave length if the
        coefficient w020 needs to be "relative to the wavelength"

    Returns
    -------
    w020 : float
        Wavefront error coefficient for defocus. w020 is the maximum
        wavefront error, which is measured at the edge of the pupil.

    Notes
    -----
    The relation between the wavefront error and the defocus as derived
    using paraxial assumption and scalar diffraction theory.
    It is given as:

    .. math::

        W_{020} = \\frac{\\delta_z a^2}{2 zi(z_i + \\delta_z)}

    which may also be approximated as :math:`W_{020} = \delta_z/8N^2`, where
    ``N`` is the f-number.

    See Also
    --------
    defocus()
    """
    w020 = (deltaZ*radius**2)/(2.0*zi*(zi + deltaZ))
    return w020/waveLength

def seidel_5(u0, v0, X, Y, wd=0, w040=0, w131=0, w222=0, w220=0, w311=0):
    """Computer wavefront OPD for first 5 Seidel wavefront aberration
    coefficients plus defocus.

    Parameters
    ----------
    u0, v0 : float
        normalized image plane coordinate along the u-axis and v-axis
        respectively
    X, Y : ndarray
        normalized pupil coordinate array (usually from meshgrid)
    wd, w040, w131, w222, w220, w311  : floats
        defocus, spherical, coma, astigmatism, field-curvature, distortion
        aberration coefficients

    Returns
    -------
    w : ndarray
        wavefront OPD at the given image plane coordinate.

    Notes
    -----
    This function is exactly implemented as is from
    'Computational Fourier Optics', David Voelz
    """
    theta = _np.arctan2(v0, u0)       # image rotation angle
    u0r = _np.sqrt(u0**2 + v0**2)     # image height
    # rotate pupil grid
    Xr =  X*_np.cos(theta) + Y*_np.sin(theta)
    Yr = -X*_np.sin(theta) + Y*_np.cos(theta)
    rho2 = Xr**2 + Yr**2
    w = ( wd*rho2            # defocus
        + w040*rho2**2       # spherical
        + w131*u0r*rho2*Xr   # coma
        + w222*u0r**2*Xr**2  # astigmatism
        + w220*u0r**2*rho2   # field curvature
        + w311*u0r**3*Xr )   # distortion
    return w


#-------------------------------------
# Some ray optics helper functions
#-------------------------------------

def get_dir_cos_from_zenith_azimuth(zenith, azimuth, atype='deg', tol=1e-12):
    """Returns the direction cosines alpha, beta & gamma
    of the direction vector described by zenith and azimuth angles

    Parameters
    ----------
    zenith : float
        angle of the direction vector with respect to the positive z-axis.
        :math:`0 \leq \\theta \leq \pi`
    azimuth : float
        angle of the direction vector with respect to the positive x-axis.
        :math:`0 \leq \phi \leq 2\pi`
    atype : string ('rad' or 'deg')
        angle unit in degree (default) or radians
    tol : float (very small number)
        tol (default=1e-12) is the absolute value below which the
        direction cosine value is set to zero.

    Returns
    -------
    direction_cosines : tuple
        (alpha, beta, gamma) are the direction cosines, which are
        cos(A), cos(B), cos(C). Where A, B, C are angles that the wave 
        vector ``k`` makes with x, y, and z axis respectively.

    Notes
    -----
    1. The zenith angle is also known as the inclination angle or polar
       angle.
    2. The angle of elevation (i.e. the angle that the ray makes with
       the x-y axis) is given as 90 - zenith (:math:`\\theta`).
    3. Direction cosines are defined as follows:

       - :math:`\\alpha = cos(A) = sin(\\theta)cos(\phi)`
       - :math:`\\beta  = cos(B) = sin(\\theta)sin(\phi)`
       - :math:`\\gamma = cos(C) = cos(\\theta)`
    3. The use of :math:`\\theta` and :math:`\phi` to represent zenith and
       azimuth angles follow the convension specified by ISO standard
       80000-2 :2009 [1]_

    Examples
    --------
    >>> get_dir_cos_from_zenith_azimuth(20.0, 10.0)
    (0.33682408883346515, 0.059391174613884698, 0.93969262078590843)

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Spherical_coordinate_system

    See Also
    --------
    get_zenith_azimuth_from_dir_cos(),
    get_alpha_beta_gamma_set()
    """
    if atype=='deg':
        zenith = _np.deg2rad(zenith)
        azimuth = _np.deg2rad(azimuth)
    cosA = _np.sin(zenith)*_np.cos(azimuth)
    cosB  = _np.sin(zenith)*_np.sin(azimuth)
    cosC = _np.cos(zenith)
    # set extremely small values to zero
    cosA = 0 if abs(cosA) < tol else cosA
    cosB = 0 if abs(cosB) < tol else cosB
    cosC = 0 if abs(cosC) < tol else cosC
    dirCosines = _co.namedtuple('dirCosines', ['alpha', 'beta', 'gamma'])
    return dirCosines(cosA, cosB, cosC)

def get_zenith_azimuth_from_dir_cos(gamma, alpha, beta, atype='deg'):
    """Returns the zenith and azimuth angles of the direction vector
    given the direction cosines gamma, alpha and/or beta

    Parameters
    ----------
    gamma : float
        direction cosine, i.e. cos(C)
    alpha : float
        direction cosine, i.e. cos(A)
    beta : float
        direction cosine, i.e. cos(B)
    atype : string
        string code indicating whether to return the angles in
        degrees ('deg') or radians ('rad')

    Returns
    -------
    zenith : float
        angle of the direction vector with respect to the positive z-axis
        :math:`0 \leq \\theta \leq \pi`
    azimuth : float
        angle of the direction vector with respect to the positive x-axis
        :math:`0 \leq \phi \leq 2\pi`

    Examples
    --------
    >>> alpha, beta, gamma = 0.33682408883, 0.05939117461, 0.93969262078
    >>> get_zenith_azimuth_from_dir_cos(gamma, alpha, beta)
    (20.000000001 10.0000000188)

    Notes
    -----
    The direction cosines (:math:`\\alpha, \\beta, \\gamma`),
    azimuth (:math:`\phi`) and zenith (:math:`\\theta`) are related
    as follows:

        - :math:`\\alpha = cos(A) = sin(\\theta)cos(\phi)`
        - :math:`\\beta  = cos(B) = sin(\\theta)sin(\phi)`
        - :math:`\\gamma = cos(C) = cos(\\theta)`
    where, A, B, C are angles that the wave vector ``k`` makes with
    x, y, and z axis respectively.

    See Also
    --------
    get_dir_cos_from_zenith_azimuth()
    """
    if atype not in ['deg', 'rad']:
        raise ValueError, "Invalid angle type specification"
    zenith = _np.arccos(gamma)
    azimuth = _np.arctan2(beta, alpha)
    rayAngle = _co.namedtuple('rayAngle', ['zenith', 'azimuth'])
    if atype=='deg':
        return rayAngle(_np.rad2deg(zenith), _np.rad2deg(azimuth))
    else:
        return rayAngle(zenith, azimuth)

def get_alpha_beta_gamma_set(alpha=None, beta=None, gamma=None, forceZero='none'):
    """Function to return the complete set of direction cosines alpha,
    beta, and gamma, given a partial set.

    Parameters
    ----------
    alpha : float
        Direction cosine, i.e. cos(A); see Notes.
    beta : float
        Direction cosine, i.e. cos(B); see Notes.
    gamma : float
        Direction cosine, i.e. cos(C); see Notes.
    forceZero : string ('none' or 'alpha' or 'beta' or 'gamma')
        Force a particular direction cosine to be zero, in order to
        calculate the other two direction cosines when the wave vector
        is restricted to a either x-y, or y-z, or x-z plane.

    Returns
    -------
    direction_cosines : tuple
        (alpha, beta, gamma)

    Warnings
    --------
    Only the positive values of the direction cosines are returned by this
    function. A sign ambiguity is always present

    Notes
    -----
    1. The function doesn't check for the validity of alpha, beta, and
       gamma.
    2. If the function returns (None, None, None), most likely 2 out of
       the 3 input direction cosines given are zeros, and ``forceZero``
       is ``none``. Please provide the appropriate value for the parameter
       ``forceZero``.
    3. A, B, C are angles that the wave vector ``k`` makes with x, y, and
       z axis respectively.

    See Also
    --------
    get_dir_cos_from_zenith_azimuth()
    get_zenith_azimuth_from_dir_cos()
    """
    def f(x,y):
        return _np.sqrt(1.0 - x**2 - y**2)
    if forceZero == 'none':
        if alpha and beta:
            return alpha, beta, f(alpha, beta)
        elif alpha and gamma:
            return alpha, f(alpha, gamma), gamma
        elif beta and gamma:
            return f(beta, gamma), beta, gamma
        else: # Error case
            return None, None, None
    elif forceZero == 'alpha':
        if beta:
            return 0.0, beta, f(beta, 0)
        else:
            return 0.0, f(gamma, 0), gamma
    elif forceZero == 'beta':
        if alpha:
            return alpha, 0.0, f(alpha,0)
        else:
            return f(gamma, 0), 0.0, gamma
    else:  # forceZero='g'
        if alpha:
            return alpha, f(alpha,0), 0.0
        else:
            return f(beta, 0), beta, 0.0


#--------------------------------
# Miscellaneous helper functions
#---------------------------------



# ---------------------------
#   TEST FUNCTIONS
# ---------------------------

def _test_fresnel_number():
    """Test fresnel_number function"""
    fresnelNum = fresnel_number(10, 500, 550e-6)
    nt.assert_almost_equal(fresnelNum, 363.6000072709645, decimal=4)
    fresnelNum = fresnel_number(10, 500, 550e-6, approx=True)
    nt.assert_almost_equal(fresnelNum, 363.636363636, decimal=4)
    print("test_fresnelNumber() is successful")

def _test_airy_pattern():
    """Test air_pattern function against a known and verified (several times)
    output. So, before doubting and changing the values of the
    ``expIntensity``, be triple sure about what you are doing!!!
    """
    x = _np.linspace(-1,1,5)
    y = x.copy()
    X, Y = _np.meshgrid(x,y)
    radius = 5.0
    wavelength = 550e-6
    z = 25.0
    rho = _np.hypot(X,Y)
    I = airy_pattern(wavelength, radius, z, rho, norm=0)
    expIntensity= _np.array([[1.37798644e-03, 8.36156468e-04, 3.56139554e-05,
                              8.36156468e-04, 1.37798644e-03],
                            [8.36156468e-04, 1.77335279e-05, 4.89330106e-02,
                             1.77335279e-05, 8.36156468e-04],
                            [3.56139554e-05, 4.89330106e-02, 3.26267914e+07,
                             4.89330106e-02, 3.56139554e-05],
                            [8.36156468e-04, 1.77335279e-05, 4.89330106e-02,
                             1.77335279e-05, 8.36156468e-04],
                            [1.37798644e-03, 8.36156468e-04, 3.56139554e-05,
                             8.36156468e-04, 1.37798644e-03]])
    nt.assert_array_almost_equal(expIntensity, I, decimal=2)
    # check for normalization
    I = airy_pattern(wavelength, radius, z, rho)
    nt.assert_almost_equal(_np.max(I), 1.0, decimal=6)
    I = airy_pattern(wavelength, radius, z, rho, norm=2)
    nt.assert_almost_equal(_np.sum(I), 1.0, decimal=6)
    print("test_airy_patten() is successful")

def _test_get_dir_cos_from_zenith_azimuth():
    """Test get_dir_cos_from_zenith_azimuth function"""
    a, b, g = get_dir_cos_from_zenith_azimuth(20.0, 10.0)
    exp_array = _np.array([0.33682408883, 0.05939117461, 0.93969262078])
    nt.assert_array_almost_equal(_np.array([a, b, g]), exp_array, decimal=8)
    a, b, g = get_dir_cos_from_zenith_azimuth(90.0, 0.0)
    exp_array = _np.array([1.0, 0.0, 0.0])
    nt.assert_array_almost_equal(_np.array([a, b, g]), exp_array, decimal=8)
    print("test_getDirCosinesFromZenithAndAzimuthAngles() is successful")

def _test_get_zenith_azimuth_from_dir_cos():
    """Test get_zenith_azimuth_from_dir_cos() function"""
    alpha, beta, gamma = 0.33682408883, 0.05939117461, 0.93969262078
    zenith, azimuth = get_zenith_azimuth_from_dir_cos(gamma, alpha, beta)
    nt.assert_array_almost_equal(_np.array((zenith, azimuth)),
                                 _np.array((20.0, 10.0)), decimal=7)
    zenith, azimuth = get_zenith_azimuth_from_dir_cos(gamma, alpha, beta,
                                                      atype='rad')
    nt.assert_array_almost_equal(_np.array((zenith, azimuth)),
                                 _np.array((0.349065850416, 0.17453292518)),
                                 decimal=7)
    try:
        zenith, azimuth = get_zenith_azimuth_from_dir_cos(gamma, alpha, beta,
                                                          atype='invalid')
    except ValueError: # as e:
        #nt.assert_string_equal(e, 'Invalid angle type specification')
        pass
    zenith, azimuth = get_zenith_azimuth_from_dir_cos(0.9950371902099892,
                                                      0.0,
                                                      0.099503719020998957)
    nt.assert_almost_equal(azimuth, 90.0, decimal=8)
    print("test_get_zenith_azimuth_from_dir_cos() is successful")

def _test_get_alpha_beta_gamma_set():
    """Test get_alpha_beta_gamma_set() function"""
    a, b, g = get_alpha_beta_gamma_set(0, 0, 1)
    exp_array = _np.array((None, None, None))
    nt.assert_array_equal(_np.array((a, b, g)), exp_array)
    a, b, g = get_alpha_beta_gamma_set(0, 0, 1, 'alpha')
    exp_array = _np.array((0.0, 0.0, 1.0))
    nt.assert_array_almost_equal(_np.array((a, b, g)), exp_array, decimal=8)
    a, b, g = get_alpha_beta_gamma_set(0, 0, 0.5, 'alpha')
    exp_array = _np.array((0.0, 0.86602540378, 0.5))
    nt.assert_array_almost_equal(_np.array((a, b, g)), exp_array, decimal=8)
    a, b, g = get_alpha_beta_gamma_set(0, 0.0593911746139, 0.939692620786)
    exp_array = _np.array((0.33682408883, 0.05939117461, 0.93969262078))
    nt.assert_array_almost_equal(_np.array((a, b, g)), exp_array, decimal=8)
    a, b, g = get_alpha_beta_gamma_set(0.33682408883320675, 0.0593911746139,0)
    exp_array = _np.array((0.33682408883, 0.05939117461, 0.93969262078))
    nt.assert_array_almost_equal(_np.array((a, b, g)), exp_array, decimal=8)
    print("test_get_alpha_beta_gamma_set() is successful")

def _test_seidel_5():
    """Test seidel_5 function"""
    #TODO!!!
    pass

def _test_diffraction_spot_size():
    #TODO!!!
    pass

def _test_w020_from_defocus():
    #TODO!!!
    pass

def _test_depth_of_focus():
    """Test depthOfFocus(), which returns the diffraction based DOF"""
    wavelength = 500e-6  # mm
    fnumber = [2, 2.8, 4.0, 5.6, 8.0, 10, 11.0, 16.0, 22.0]
    dz = [depth_of_focus(N, wavelength) for N in fnumber]
    exp_array = _np.array((0.004074366543152521, 0.00798575842457894,
                           0.016297466172610083, 0.03194303369831576,
                           0.06518986469044033, 0.10185916357881303,
                           0.12324958793036377, 0.2607594587617613,
                           0.49299835172145506))
    nt.assert_array_almost_equal(_np.array(dz), exp_array, decimal=6)
    print("test_depthOfFocus() is successful")

def _test_depth_of_field():
    #TODO!!!
    pass

def _test_geometric_depth_of_field():
    #TODO!!!
    pass

if __name__ == '__main__':
    import numpy.testing as nt
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    _test_fresnel_number()
    _test_airy_pattern()
    _test_get_dir_cos_from_zenith_azimuth()
    _test_get_zenith_azimuth_from_dir_cos()
    _test_get_alpha_beta_gamma_set()
    _test_depth_of_focus()