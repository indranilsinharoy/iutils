# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------------------
# Name:          foptics.py
# Purpose:       Fourier Optics Utility Functions
#
# Author:        Indranil Sinharoy
#
# Created:       09/22/2012
# Last Modified: 04/30/2015
# Copyright:     (c) Indranil Sinharoy 2012, 2013, 2014, 2015
# License:       MIT License
#-----------------------------------------------------------------------------------------
"""foptics module contains some useful functions for Fourier Optics based
calculations.
"""
from __future__ import division, print_function
import math as _math
import numpy as _np
from iutils.signal.signals import jinc as _jinc
import iutils.optics.imager as _imgr
import warnings as _warnings
import collections as _co

#%% Module helper functions

def _set_small_values_to_zero(tol, *values):
    """helper function to set infinitesimally small values to zero
    
    Parameters
    ----------
    tol : float
        threshold. All numerical values below abs(tol) is set to zero
    *values : unflattened sequence of values
        
    Returns
    -------
    
    Example
    -------
    >>> tol = 1e-12
    >>> a, b, c, d = _set_small_values_to_zero(tol, 1.0, 0.0, tol, 1e-13)
    >>> a 
    1.0
    >>> b 
    0.0
    >>> c 
    1e-12
    >>> d
    0.0
    """
    return [0 if abs(value) < tol else value for value in values]

def _is_dir_cos_valid(alpha, beta, gamma, tol=1e-12):
    """test the validity of direction cosines
    returns ``True`` if valid, ``False`` if not
    """
    absdiff = abs(alpha**2 + beta**2 + gamma**2 - 1)
    if absdiff > tol:
        print('The sum of squares of the direction cosines is not approximately '
              'equal to 1. The absolute difference is {:2.4E}'.format(absdiff))
        return False
    else:
        return True
        
def _first(iterable, what, test='equality'):
    """return the index of the first occurance of ``what`` in ``iterable``
    """
    if test=='equality':
        for index, item in enumerate(iterable):
            if item == what:
                break
        else:
                index = None
    else:
        raise NotImplementedError
    return index

#%% Fourier optics related functions

def fresnel_number(r, z, wavelen=550e-6, approx=False):
    """calculate the fresnel number

    Parameters
    ----------
    r : float
        radius of the aperture in units of length (usually mm)
    z : float
        distance of the observation plane from the aperture. this is equal
        to the focal length of the lens for infinite conjugate, or the
        image plane distance, in the same units of length as ``r``
    wavelen : float, optional
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
       represents the number of annular "Fresnel zones" in the aperture
       opening [Wolf2011]_, or from the center of the beam to the edge in
       case of a propagating beam [Zemax]_. Fresnel zones are the radial 
       zones where the phase as seen from the observation point changes by π.
    3. The Fresnel number is also used to determine if the observation point
       is in the near or far field from the position of current propagation.
       If Fresnel number << 1, the observation point is said to be in the
       far-field relative to the propagation of the present location of 
       the beam [Zemax]_. 

    References
    ----------
    .. [Zemax] Zemax manual

    .. [Born&Wolf2011] Principles of Optics, Born and Wolf, 2011
    """
    if approx:
        return (r**2)/(wavelen*z)
    else:
        return 2.0*(_math.sqrt(z**2 + r**2) - z)/wavelen
        
def fraunhofer_distance(d=1.0, wavelen=550e-6):
    """computes the Fraunhofer distance from a diffracting aperture. It is the
    limit between the near and far field
    
    Parameters
    ----------
    d : float
        largest dimension of the aperture. i.e. aperture width/diameter (in mm)
    wl : float
        wavelength of light (in mm)
    
    Returns
    -------
    fdist : float
        the Fraunhofer distance in mm
        
    Notes
    -----
    This is the more stringent condition for the Fraunhofer Approximation. 
    Therefore the far-field is the region where ``z > fraunhofer_distance``
    """
    return 2.0*d**2 / wavelen

def diffraction_spot_size(fnum=None, r=None, zi=None, wavelen=550e-6):
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
    wavelen : float, optional
        wavelength of light (default=550e-6 mm)

    Returns
    -------
    spot_size : float
        The diffraction limited spot size given as 2.44*lambda*f/#
    """
    spotSize = 2.44*wavelen*fnum if fnum else 1.22*wavelen*(zi/r)
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


def airy_pattern(r, zxp, rho, wavelen, norm=1):
    """Fraunhofer intensity diffraction pattern for a circular aperture

    Parameters
    ----------
    r : float
        radius of the aperture
    zxp : float
        distance to the screen/image plane from the aperture/Exit pupil
    rho : ndarray
        radial coordinate in the screen/image plane (such as constructed
        using ``meshgrid``)
    wavelen : float
        wavelength in physical units (same unit of ``r``, ``zxp``, ``rho``)
    norm : integer (0 or 1 or 2), optional
        0 = None;
        1 = Peak value is 1.0 (default);
        2 = Sum of the area under PSF=1.0

    Returns
    -------
    pattern : ndarray
        Fraunhofer diffraction pattern (Airy pattern)

    Examples
    --------
    This example creates an airy pattern for light of 0.55 μm wavelength
    diffracting through an unaberrated lens of focal length 50 mm, and 
    EPD 20 mm. The airy pattern is generated on a spatial grid that extends 
    between -5λ and 5λ along both x and y

    >>> lamba = 550e-6
    >>> M, N = 513, 513            # odd grid for symmetry
    >>> dx = 10*lamba/N
    >>> dy = 10*lamba/M
    >>> x = (np.linspace(0, N-1, N) - N//2)*dx
    >>> y = (np.linspace(0, M-1, M) - M//2)*dy
    >>> xx, yy = np.meshgrid(x, y)
    >>> rho = np.hypot(xx, yy)
    >>> w = 10                    # radius, in mm
    >>> z = 50                    # z-distance, in mm
    >>> ipsf = fou.airy_pattern(w, z, rho, lamba, 1)
    """
    #TODO!!! (remove this warning in future)
    if r < 1e-3 :  # was possibly was for wavelength
        _warnings.warn('\nAPI has changed. Please ensure parameter validity!\n')
    # The jinc() function used here is defined as jinc(x) = J_1(x)/x, jinc(0) = 0.5
    pattern = ((_np.pi*r**2/(wavelen*zxp))
               *2*_jinc(2*_np.pi*r*rho/(wavelen*zxp), normalize=False))**2
    if norm==1:
        pattern = pattern/_np.max(pattern)
    elif norm==2:
        pattern = pattern/_np.sum(pattern)
    return pattern

def depth_of_focus(effFNum, wavelen=550e-6, firstZero=False, full=False):
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
    wavelen : float, optional
        wavelength of light (default=550e-6 mm)
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
        deltaZ = 8.0*wavelen*effFNum**2
    else:
        deltaZ = (6.4*wavelen*effFNum**2)/_np.pi
    if full:
        deltaZ = deltaZ*2.0
    return deltaZ

def depth_of_field(focalLength, fNumber, objDist, wavelen=550e-6, firstZero=False):
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
    wavelen : float
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
    deltaZ = depth_of_focus(effFNum, wavelen, firstZero)   # half diffraction DOF
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



#%% Aberration calculation functions

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

def w020_from_defocus(radius, zi, deltaZ, wavelen=1.0):
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
    wavelen : float
        The ``waveLength`` is used to specify a wavelength if the
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
    return w020/wavelen

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


#%% Some ray optics helper functions

def dir_cos_from_zenith_azimuth(zenith, azimuth, atype='deg', tol=1e-12):
    """returns the direction cosines alpha, beta & gamma
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
    >>> dir_cos_from_zenith_azimuth(20.0, 10.0)
    (0.33682408883346515, 0.059391174613884698, 0.93969262078590843)

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Spherical_coordinate_system

    See Also
    --------
    zenith_azimuth_from_dir_cos(),
    get_alpha_beta_gamma_set()
    """
    if atype=='deg':
        zenith = _np.deg2rad(zenith)
        azimuth = _np.deg2rad(azimuth)
    cosA = _np.sin(zenith)*_np.cos(azimuth)
    cosB  = _np.sin(zenith)*_np.sin(azimuth)
    cosC = _np.cos(zenith)
    assert _is_dir_cos_valid(cosA, cosB, cosC)
    # set extremely small values to zero
    cosA, cosB, cosC = _set_small_values_to_zero(tol, cosA, cosB, cosC)
    dirCosines = _co.namedtuple('dirCosines', ['alpha', 'beta', 'gamma'])
    return dirCosines(cosA, cosB, cosC)

def zenith_azimuth_from_dir_cos(gamma, alpha, beta, atype='deg'):
    """returns the zenith and azimuth angles of the direction vector given the 
    direction cosines gamma, alpha and/or beta

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
    >>> zenith_azimuth_from_dir_cos(gamma, alpha, beta)
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
    dir_cos_from_zenith_azimuth()
    dir_cos_from_angles()
    angles_from_dir_cos()
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

def angles_from_dir_cos(gamma, alpha, beta, atype='deg', tol=1e-12):
    """returns the angles of the direction vector w.r.t. the z, x, and y, axis 
    given the direction cosines gamma, alpha and/or beta

    Parameters
    ----------
    gamma : float
        direction cosine, i.e. cos(theta_z)
    alpha : float
        direction cosine, i.e. cos(theta_x)
    beta : float
        direction cosine, i.e. cos(theta_y)
    atype : string
        string code indicating whether to return the angles in
        degrees ('deg') or radians ('rad')
    tol : float (very small number), optional
        tol (default=1e-12) is the absolute value below which the direction 
        cosine value is set to zero.

    Returns
    -------
    theta_z : float
        angle w.r.t. z-axis
    theta_x : float
        angle w.r.t. x-axis
    theta_y : float
        angle w.r.t. y-axis
        
    See Also
    --------
    zenith_azimuth_from_dir_cos()
    angles_from_spatial_freq()
    """
    assert _is_dir_cos_valid(alpha, beta, gamma)
    theta_z, theta_x, theta_y = [_math.acos(dc) for dc in (gamma, alpha, beta)]
    theta_z, theta_x, theta_y = _set_small_values_to_zero(tol, theta_z, theta_x, theta_y)
    if atype == 'deg':
       theta_z, theta_x, theta_y = [t*180/_math.pi for t in (theta_z, theta_x, theta_y)]
    angles = _co.namedtuple('angles', ['theta_z', 'theta_x', 'theta_y'])
    return angles(theta_z, theta_x, theta_y)

def get_alpha_beta_gamma_set(alpha=None, beta=None, gamma=None, forceZero='none'):
    """returns the complete set of direction cosines {α, β, γ} given a partial set. Use 
    `None` for the unknowns

    Parameters
    ----------
    alpha : float
        x-direction cosine, cos(θ_x)
    beta : float
        y-direction cosine, cos(θ_y)
    gamma : float
        z-direction cosine, cos(θ_z)
    forceZero : string ('none', 'alpha', 'beta' or 'gamma')
        Force a particular direction cosine to be zero, in order to
        calculate the other two direction cosines when the wavevector (ḵ)
        is restricted to either x-y, or y-z, or x-z plane -- α=0 if ḵ is
        in y-z plane, β=0 if ḵ is in x-z plane, and γ=0 if ḵ is in x-y plane

    Returns
    -------
    dircosines : tuple of floats
        (alpha, beta, gamma)

    Warnings
    --------
    Only the positive values of the direction cosines are returned by this
    function. A sign ambiguity is always present

    Notes
    -----
    1. The function doesn't check for the validity of {α, β, γ}. Use 
       `fou._is_dir_cos_valid()` if you need to.
    2. θ_x, θ_y, θ_z are angles that the wavevector ``k`` makes with 
       x, y, and z axis respectively.

    See Also
    --------
    dir_cos_from_zenith_azimuth(), zenith_azimuth_from_dir_cos()
    """
    def f(x,y):
        return _np.sqrt(1.0 - x**2 - y**2)
    if forceZero == 'none':
        if alpha is not None and beta is not None:
            return alpha, beta, f(alpha, beta)
        elif alpha is not None and gamma is not None:
            return alpha, f(alpha, gamma), gamma
        elif beta is not None and gamma is not None:
            return f(beta, gamma), beta, gamma
        #else: # Error case
        #    return None, None, None
    elif forceZero == 'alpha':
        if beta is not None:
            return 0.0, beta, f(beta, 0)
        else:
            return 0.0, f(gamma, 0), gamma
    elif forceZero == 'beta':
        if alpha is not None:
            return alpha, 0.0, f(alpha,0)
        else:
            return f(gamma, 0), 0.0, gamma
    else:  # forceZero='g'
        if alpha is not None:
            return alpha, f(alpha,0), 0.0
        else:
            return f(beta, 0), beta, 0.0

def spatial_freq_from_zenith_azimuth(zenith, azimuth, wavelen=550e-6, atype='deg', tol=1e-12):
    """returns the spatial frequencies associated with a plane wave with wavevector 
    having zenith and azimuth angle
    
    Parameters
    ----------
    zenith : float
        angle of the direction vector with respect to the positive z-axis.
        :math:`0 \leq \\theta \leq \pi`
    azimuth : float
        angle of the direction vector with respect to the positive x-axis.
        :math:`0 \leq \phi \leq 2\pi`
    wavelen : float, optional
        wavelength in millimeters
    atype : string  ('rad' or 'deg'), optional
        angle unit in degree (default) or radians
    tol : float (very small number), optional
        tol (default=1e-12) is the absolute value below which the
        direction cosine value is set to zero.
        
    Returns
    -------
    fx, fy, fz : tuple
        spatial frequencies along x, y and z directions in cycles/mm
    """
    dirCos = dir_cos_from_zenith_azimuth(zenith, azimuth, atype, tol)
    fx = dirCos.alpha/wavelen
    fy = dirCos.beta/wavelen
    fz = dirCos.gamma/wavelen
    fx, fy, fz = _set_small_values_to_zero(tol, fx, fy, fz)
    sfreq = _co.namedtuple('spatialFrequency', ['fx', 'fy', 'fz'])
    return sfreq(fx, fy, fz)

def spatial_freq_from_dir_cos(gamma=None, alpha=None, beta=None, wavelen=550e-6, tol=1e-12):
    """returns the spatial frequencies associated with a plane wave with direction 
    cosines gamma, alpha and/or beta

    Parameters
    ----------
    gamma : float or None
        direction cosine, i.e. cos(theta_z)
    alpha : float or None
        direction cosine, i.e. cos(theta_x)
    beta : float or None
        direction cosine, i.e. cos(theta_y)
    wavelen : float, optional
        wavelength in millimeters
    tol : float (very small number), optional
        tol (default=1e-12) is the absolute value below which the direction 
        cosine value is set to zero.
        
    Returns
    -------
    fx, fy, fz : tuple
        spatial frequencies along x, y and z directions in cycles/mm
    
    Notes
    -----
    1. At least 2 out of the 3 direction cosines must be specified.
    """
    if not all([alpha, beta, gamma]):
        alpha, beta, gamma = get_alpha_beta_gamma_set(alpha, beta, gamma)
    assert _is_dir_cos_valid(alpha, beta, gamma)
    fx, fy, fz = _set_small_values_to_zero(tol, alpha/wavelen, beta/wavelen, gamma/wavelen)
    sfreq = _co.namedtuple('spatialFrequency', ['fx', 'fy', 'fz'])
    return sfreq(fx, fy, fz)  

def spatial_freq_from_angles(theta_z, theta_x, theta_y, wavelen=550e-6, atype='deg', tol=1e-12):
    """returns the spatial frequencies associated with a plane wave whose wave vector
    makes angle theta_z w.r.t z-axis, theta_x w.r.t. x-axis, & theta_y w.r.t. y-axis

    Parameters
    ----------
    theta_z : float
        angle w.r.t. z-axis
    theta_x : float
        angle w.r.t. x-axis
    theta_y : float
        angle w.r.t. y-axis
    wavelen : float, optional
        wavelength in millimeters
    atype : string
        string code indicating whether the specified angles is in degrees ('deg') 
        or radians ('rad')
    tol : float (very small number), optional
        tol (default=1e-12) is the absolute value below which the direction 
        cosine value is set to zero.
        
    Returns
    -------
    fx, fy, fz : tuple
        spatial frequencies along x, y and z directions in cycles/mm
    """
    if atype == 'deg':
        theta_z, theta_x, theta_y = [t*_math.pi/180 for t in (theta_z, theta_x, theta_y)]
    gamma, alpha, beta = [_math.cos(t) for t in (theta_z, theta_x, theta_y)]
    return spatial_freq_from_dir_cos(gamma, alpha, beta, wavelen, tol)
    
def zenith_azimuth_from_spatial_freq(fx, fy, fz, wavelen=550e-6, atype='deg', tol=1e-12):
    """returns the zenith and azimuth angles associated with spatial frequencies

    Parameters
    ----------
    fx : float
        spatial frequency along x-axis (unit=cycles/mm)
    fy : float
        spatial frequency along y-axis (unit=cycles/mm)
    fz : float
        spatial frequency along z-axis (unit=cycles/mm)
    wavelen : float, optional
        wavelength in millimeters
    atype : string
        string code indicating whether the specified angles is in degrees ('deg') 
        or radians ('rad')
    tol : float (very small number), optional
        tol (default=1e-12) is the absolute value below which the direction 
        cosine value is set to zero.
        
    Returns
    -------
    zenith : float
        angle of the direction vector with respect to the positive z-axis
        :math:`0 \leq \\theta \leq \pi`
    azimuth : float
        angle of the direction vector with respect to the positive x-axis
        :math:`0 \leq \phi \leq 2\pi`
    """
    raise NotImplementedError
    
def dir_cos_from_spatial_freq(fx=None, fy=None, fz=None, wavelen=550e-6, tol=1e-12):
    """returns the direction cosines of the direction vector of a plane wave 
    of given spatial frequency
    
    Parameters
    ----------
    fx : float
        spatial frequency along x-axis (unit=cycles/mm)
    fy : float
        spatial frequency along y-axis (unit=cycles/mm)
    fz : float
        spatial frequency along z-axis (unit=cycles/mm)
    wavelen : float, optional
        wavelength in millimeters
    tol : float (very small number), optional
        tol (default=1e-12) is the absolute value below which the direction 
        cosine value is set to zero.

    Returns
    -------
    direction_cosines : tuple
        (alpha, beta, gamma) are the direction cosines, which are
        cos(A), cos(B), cos(C). Where A, B, C are angles that the wave 
        vector ``k`` makes with x, y, and z axis respectively.
        
    See Also
    --------
    angles_from_spatial_freq(), 
    """
    spatialFreqNone = _first([fx, fy, fz], None)
    if spatialFreqNone is not None:
        if spatialFreqNone == 2:
            a, b = wavelen*fx, wavelen*fy
            alpha, beta, gamma = get_alpha_beta_gamma_set(alpha=a, beta=b, gamma=None)
        elif spatialFreqNone == 1:
            a, g = wavelen*fx, wavelen*fy
            alpha, beta, gamma = get_alpha_beta_gamma_set(alpha=a, beta=None, gamma=g)
        else:
            b, g = wavelen*fy, wavelen*fz
            alpha, beta, gamma = get_alpha_beta_gamma_set(alpha=None, beta=b, gamma=g)
    else:
        alpha, beta, gamma = wavelen*fx, wavelen*fy, wavelen*fz
    assert _is_dir_cos_valid(alpha, beta, gamma)
    dirCosines = _co.namedtuple('dirCosines', ['alpha', 'beta', 'gamma'])
    return dirCosines(alpha, beta, gamma)
    
def angles_from_spatial_freq(fx=None, fy=None, fz=None, wavelen=550e-6, atype='deg', tol=1e-12):
    """returns the angles w.r.t. x, y and z of the direction vector of a plane wave 
    of given spatial frequency
    
    Parameters
    ----------
    fx : float
        spatial frequency along x-axis (unit=cycles/mm)
    fy : float
        spatial frequency along y-axis (unit=cycles/mm)
    fz : float
        spatial frequency along z-axis (unit=cycles/mm)
    wavelen : float, optional
        wavelength in millimeters
    atype : string, optional
        string code indicating whether the returned angles is in degrees ('deg') 
        or radians ('rad')
    tol : float (very small number), optional
        tol (default=1e-12) is the absolute value below which the direction 
        cosine value is set to zero.

    Returns
    -------
    theta_z : float
        angle w.r.t. z-axis
    theta_x : float
        angle w.r.t. x-axis
    theta_y : float
        angle w.r.t. y-axis
        
    See Also
    --------
    angles_from_dir_cos(),
    spatial_freq_from_angles()
    """
    dirCos = dir_cos_from_spatial_freq(fx, fy, fz, wavelen, tol)
    return angles_from_dir_cos(dirCos.gamma, dirCos.alpha, dirCos.beta, atype, tol)

def grating_refracted_angle(d, thetai, wavelen=550e-6, m=1, n1=1.0, n2=1.0, atype='deg'):
    """returns the refracted angle using the grating equation
    
    Parameters
    ----------
    d : float
        grating spacing in micrometer. 1/d is the grating frequency in lines/micrometer
    thetai : float
        incident angle
    wavelen : float, optional
        wavelength in millimeters
    m : integer, optional
        grating order
    n1, n2 : float, optional
        refractive indices at the input and output side respectively
    atype : string, optional
        whether the angle is specified (and returned) in degrees ('deg') or radians 
        ('rad')
        
    Returns
    -------
    thetar : float
        refracted angle
    """
    wavelen = wavelen*1000.0 # wavelength in microns
    thetai = _math.radians(thetai) if atype=='deg' else thetai
    thetar = _math.asin((n1*_math.sin(thetai) + m*wavelen/d)/n1)
    thetar = _math.degrees(thetar) if atype=='deg' else thetar
    return thetar
    
#%% Miscellaneous helper functions
    

#%% TEST FUNCTIONS

def _test_set_small_values_to_zero():
    """Test helper function _set_small_values_to_zero()"""
    tol = 1e-12
    a, b, c, d = _set_small_values_to_zero(tol, 1.0, 0.0, tol, 1e-13)
    assert a == 1.0
    assert b == 0.0
    assert c == tol
    assert d == 0.0
    a, b, c, d = _set_small_values_to_zero(tol, -1.0, -0.0, -tol, -1e-13)
    assert a == -1.0
    assert b == -0.0
    assert c == -tol
    assert d == 0.0
    print("test_set_small_values_to_zero() successful")
    
def _test_is_dir_cos_valid():
    """Test helper function _is_dir_cos_valid()"""
    assert _is_dir_cos_valid(1, 0, 0) == True
    assert _is_dir_cos_valid(1.01, 0, 0) == False
    print("test_is_dir_cos_valid() successful")

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
    I = airy_pattern(radius, z, rho, wavelength, norm=0)
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
    I = airy_pattern(radius, z, rho, wavelength)
    nt.assert_almost_equal(_np.max(I), 1.0, decimal=6)
    I = airy_pattern(radius, z, rho, wavelength, norm=2)
    nt.assert_almost_equal(_np.sum(I), 1.0, decimal=6)
    print("test_airy_patten() is successful")

def _test_dir_cos_from_zenith_azimuth():
    """Test dir_cos_from_zenith_azimuth function"""
    a, b, g = dir_cos_from_zenith_azimuth(20.0, 10.0)
    exp_array = _np.array([0.33682408883, 0.05939117461, 0.93969262078])
    nt.assert_array_almost_equal(_np.array([a, b, g]), exp_array, decimal=8)
    a, b, g = dir_cos_from_zenith_azimuth(90.0, 0.0)
    exp_array = _np.array([1.0, 0.0, 0.0])
    nt.assert_array_almost_equal(_np.array([a, b, g]), exp_array, decimal=8)
    print("test_dir_cos_from_zenith_azimuth() is successful")

def _test_zenith_azimuth_from_dir_cos():
    """Test zenith_azimuth_from_dir_cos() function"""
    alpha, beta, gamma = 0.33682408883, 0.05939117461, 0.93969262078
    zenith, azimuth = zenith_azimuth_from_dir_cos(gamma, alpha, beta)
    nt.assert_array_almost_equal(_np.array((zenith, azimuth)),
                                 _np.array((20.0, 10.0)), decimal=7)
    zenith, azimuth = zenith_azimuth_from_dir_cos(gamma, alpha, beta,
                                                      atype='rad')
    nt.assert_array_almost_equal(_np.array((zenith, azimuth)),
                                 _np.array((0.349065850416, 0.17453292518)),
                                 decimal=7)
    try:
        zenith, azimuth = zenith_azimuth_from_dir_cos(gamma, alpha, beta,
                                                          atype='invalid')
    except ValueError: # as e:
        #nt.assert_string_equal(e, 'Invalid angle type specification')
        pass
    zenith, azimuth = zenith_azimuth_from_dir_cos(0.9950371902099892,
                                                  0.0,
                                                  0.099503719020998957)
    nt.assert_almost_equal(azimuth, 90.0, decimal=8)
    print("test_zenith_azimuth_from_dir_cos() is successful")

def _test_get_alpha_beta_gamma_set():
    """Test get_alpha_beta_gamma_set() function"""
    a, b, g = get_alpha_beta_gamma_set(None, None, 1, forceZero='alpha')
    exp_array = _np.array((0.0, 0.0, 1.0))
    nt.assert_array_almost_equal(_np.array((a, b, g)), exp_array, decimal=8)
    a, b, g = get_alpha_beta_gamma_set(None, None, 1, forceZero='alpha')
    exp_array = _np.array((0.0, 0.0, 1.0))
    nt.assert_array_almost_equal(_np.array((a, b, g)), exp_array, decimal=8) 
    a, b, g = get_alpha_beta_gamma_set(None, None, 0.5, forceZero='alpha')
    exp_array = _np.array((0.0, 0.86602540378, 0.5))
    nt.assert_array_almost_equal(_np.array((a, b, g)), exp_array, decimal=8)
    a, b, g = get_alpha_beta_gamma_set(None, 0.0593911746139, 0.939692620786)
    exp_array = _np.array((0.33682408883, 0.05939117461, 0.93969262078))
    nt.assert_array_almost_equal(_np.array((a, b, g)), exp_array, decimal=8)
    a, b, g = get_alpha_beta_gamma_set(0.33682408883320675, 0.0593911746139, None)
    exp_array = _np.array((0.33682408883, 0.05939117461, 0.93969262078))
    nt.assert_array_almost_equal(_np.array((a, b, g)), exp_array, decimal=8)
    print("test_get_alpha_beta_gamma_set() is successful")
    
def _test_spatial_freq_from_zenith_azimuth():
    #TODO!!!
    pass

def _test_spatial_freq_from_dir_cos():
    #TODO!!!
    pass

def _test_spatial_freq_from_angles():
    #TODO!!!
    pass 

def _test_angles_from_dir_cos():
    #TODO!!!
    pass

def _test_zenith_azimuth_from_spatial_freq():
    #TODO!!!
    pass

def _test_angles_from_spatial_freq():
    #TODO!!!
    pass

def _test_dir_cos_from_spatial_freq():
    #TODO!!!
    pass

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

def _test_grating_refracted_angle():
    #TODO!!!
    pass

if __name__ == '__main__':
    import numpy.testing as nt
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    _test_set_small_values_to_zero()
    _test_is_dir_cos_valid()
    _test_fresnel_number()
    _test_airy_pattern()
    _test_dir_cos_from_zenith_azimuth()
    _test_zenith_azimuth_from_dir_cos()
    _test_get_alpha_beta_gamma_set()
    _test_depth_of_focus()
