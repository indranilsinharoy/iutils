# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          imager.py
# Purpose:       camera and imaging system utility functions
#
# Author:        Indranil Sinharoy
#
# Created:       02/08/2014
# Last Modified: 07/04/2014
# Copyright:     (c) Indranil Sinharoy 2014
# Licence:       MIT License
#-------------------------------------------------------------------------------
from __future__ import division, print_function
import numpy as _np
import math as _math
import collections as _co

#----------------------------------
# Digital sensor related functions
#----------------------------------

def pixel_pitch(width, height, megapixels):
    """Return the pixel pitch for digital a sensor of given ``width``,
    ``height`` and ``megapixels``

    It assumes 100% fill factor.

    Parameters
    ----------
    width : float
        width of the sensor in millimeters
    height : float
        height of the sensor in millimeters
    megapixels : float
        number of megapixels in the sensor

    Returns
    -------
    pixelPitch : float
        pixel pitch in microns
    """
    return _np.sqrt(width*height/megapixels)

def sensor_nyqusit_frequency(pixelPitch):
    """Return the nyquist frequency of the sensor

    Parameters
    ----------
    pixelPitch : float
        pixel ptich of the sensor in microns

    Returns
    -------
    nyqFreq : float
        sensor Nyquist frequency in line-pair-per-millimeter (LPMM)

    Notes
    -----
    .. math::

        F_{nyquist} = \\frac{F_s}{2 } = \\frac{1}{2*pixelPitch}
    """
    return 500.0/pixelPitch

def lpph2lpmm(lpph, n, pixelPitch):
    """Convert resolution specified in Line-Pair per Picture Height (LPPH)
    to Line-Pair per Milli Meters (LPMM)

    Parameters
    ----------
    lpph : float
        resolution in terms of Line-Pair per Picture Height (LPPH)
    n : integer
        Number of pixels in the picture along the dimension
        (height or width) of consideration
    pixelPitch : float
        pixel pitch of the sensor in microns

    Returns
    -------
    lpmm : float
        resolution in terms of Line-Pair per Milli-Meters (LPMM)
    """
    return lpph*1000.0/(n*pixelPitch)

def lpmm2lpph(lpmm, n, pixelPitch):
    """Convert resolution specified in Line-Pair per Milli Meters (LPMM) to
    Line-Pair per Picture Height (LPPH)

    Parameters
    ----------
    lpmm : float
        resolution in terms of Line-Pair per Milli-Meters (LPMM)
    n : integer
        Number of pixels in the picture along the dimension
        (height or width) of consideration
    pixelPitch : float
        pixel pitch of the sensor in microns

    Returns
    -------
    lpph : float
        resolution in terms of Line-Pair per Picture Height (LPPH)
    """
    return lpmm*n*pixelPitch/1000.0


#---------------------------------------
# Imaging functions
#---------------------------------------
def gaussian_lens_formula(u=None, v=None, f=None, infinity=10e20):
    """return the third value of the Gaussian lens formula, given any two

    Parameters
    ----------
    u : float, optional
        object distance
    v : float, optional
        image distance
    f : float, optional
        focal length
    infinity : float
        numerical value to represent infinity (default=10e20)

    Returns
    -------
    value : float
        the third value given the other two of the Gaussian lens formula

    Examples
    --------
    >>> gaussian_lens_formula(u=1e20, f=10)
    10.0
    """
    if u:
        if v:
            f = (u*v)/(u+v)
            return f
        elif f:
            try:
                v = (u*f)/(u - f)
            except ZeroDivisionError:
                v = infinity
            return v
    else:
        try:
            u = (v*f)/(v - f)
        except ZeroDivisionError:
            u = infinity
        return u

def geometric_depth_of_field(focalLength, fNumber, objDist, coc, grossExpr=False):
    """Returns the geometric depth of field value

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
    The expression for the geometric depth of field is given as:

    .. math::

        DOF = \\frac{2 N c f^2 u^2}{f^4 - N^2 c^2 u^2}

    The expression for the geometric depth of filed using Gross
    Expression, which is derived from the expression given in
    30.8 of Handbook of Optical Systems, vol 3, Gross is as follows:

    .. math::

        DOF = \\frac{2 c N (u - f)^2}{f^2}
    """
    if grossExpr:
        dof = (2.0*coc*fNumber*(objDist-focalLength)**2)/focalLength**2
    else:
        dof = ((2.0*focalLength**2*fNumber*coc*objDist**2)/
               (focalLength**4 - (fNumber*coc*objDist)**2))
    return dof


#---------------------------------------
# simple Scheimpflug camera
#---------------------------------------
# This is a very simple rudimentary version for quick calculations ...
# will extended it as we go ... may even break it up into components like
# lens, sensor, etc.

class Scheimpflug(object):
    """Class for quick calculations of geometric scheimpflug imaging
    """
    infty = 10e22    # value of infinity
    def __init__(self, f, u=None, v=None, alpha=None, beta=None, atype='d'):
        """Scheimpflug camera class

        Parameters
        ----------
        f : float
            focal length, in length units (usually in mm)
        u : float, optional
            object to lens distance along the optical axis, in the same units
            as of ``f``
        v : float, optional
            lens to image plane distance along the optical axis
        alpha : float, optional
            angle between lens standard and image standard. Units specified
            by the parameter ``atype``. Specify either ``alpha`` or ``beta``
        beta : float, optional
            anglebetween the PoSF and lens standard. Units specified by the
            parameter ``atype``. Specify either ``alpha`` or ``beta``
        atype : string, optional
            string code to specify angle. Use ``d`` for degrees and ``r``
            for radians.

        Notes
        -----
        1. If both ``alpha`` and ``beta`` are not specified during the object
           creation, both of them are set to 0 (i.e. a rigid camera).
        2. If both ``u`` and ``v`` are not specified during the object creation,
           ``v`` is set equal to ``f``, the focal length.
        3. If ``v``, the image plane distance is set or re-set, the corresponding
           ``u`` is calculated, and then the value of ``beta`` is reevaluated
           keeping ``alpha`` constant. This is consistant with how we may
           adjust the standards in real life.
        4. Internally, the class represents angles in radians
        """
        if atype not in ['d', 'r']:
            raise TypeError, 'Invalid angle type specified'
        if (alpha is not None) and (beta is not None):
            raise TypeError, 'Specify either alpha or beta but not both'
        self._atype = atype

        # Set the focal length
        self._f = f

        # set the object and image distances
        if u is not None:
            self._u = u
            self._v = gaussian_lens_formula(u=u, v=None, f=f)
        elif v is not None:
            self._v = v
            self._u = gaussian_lens_formula(u=None, v=v, f=f)
        else:
            self._v = f
            self._u = gaussian_lens_formula(u=None, v=f, f=f,
                                            infinity=Scheimpflug.infty)

        # set the angles
        if alpha is not None:
            self._alpha = Scheimpflug._deg2rag(alpha) if atype == 'd' else alpha
            self._beta = Scheimpflug._alpha2beta(self._alpha, self._u, self._f)
        elif beta is not None:
            self._beta = Scheimpflug._deg2rag(beta) if atype == 'd' else beta
            self._alpha = Scheimpflug._beta2alpha(self._beta, self._u, self._f)
        else:
            self._alpha = 0
            self._beta = 0

    def __repr__(self):
        alpha = self._alpha
        beta = self._beta
        alpha = Scheimpflug._rad2deg(alpha) if self._atype == 'd' else alpha
        beta = Scheimpflug._rad2deg(beta) if self._atype == 'd' else beta
        return ("Scheimpflug(f={:2.2f}, alpha={:2.2f}, beta={:2.2f}, u={:2.2f},"
                " v={:2.2f})".format(self._f, alpha, beta, self._u, self._v))

    @property
    def f(self):
        """gets focus, ``f``
        """
        return self._f

    @f.setter
    def f(self, val):
        """sets focus, ``f``, and recomputes ``u`` and ``beta``
        """
        self._f = val
        self._u = gaussian_lens_formula(u=None, v=self._v, f=val)
        self._beta = Scheimpflug._alpha2beta(self._alpha, self._u, self._f)

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, val):
        """sets object distance, ``u``, and recomputes ``v`` and ``alpha``
        """
        self._u = val
        self._v = gaussian_lens_formula(u=val, v=None, f=self._f)
        self._alpha = Scheimpflug._beta2alpha(self._beta, self._u, self._f)

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, val):
        """sets image plane distance, ``v``, and recomputes ``u`` and ``beta``
        """
        self._v = val
        self._u = gaussian_lens_formula(u=None, v=val, f=self._f)
        # recalculate beta
        self._beta = Scheimpflug._alpha2beta(self._alpha, self._u, self._f )

    @property
    def alpha(self):
        ang = self._alpha
        ang = Scheimpflug._rad2deg(ang) if self._atype == 'd' else ang
        return ang

    @alpha.setter
    def alpha(self, val):
        self._alpha = Scheimpflug._deg2rag(val) if self._atype == 'd' else val
        self._beta = Scheimpflug._alpha2beta(self._alpha, self._u, self._f )

    @property
    def beta(self):
        ang = self._beta
        ang = Scheimpflug._rad2deg(ang) if self._atype == 'd' else ang
        return ang

    @beta.setter
    def beta(self, val):
        self._beta = Scheimpflug._deg2rag(val) if self._atype == 'd' else val
        self._alpha = Scheimpflug._beta2alpha(self._beta, self._u, self._f)

    @staticmethod
    def _alpha2beta(alpha, u, f):
        """
        alpha : in radians
        """
        tanAlpha = _math.tan(alpha)
        tanBeta = (u/f - 1)*tanAlpha
        return _math.atan(tanBeta)

    @staticmethod
    def _beta2alpha(beta, u, f):
        """
        beta : in radians
        """
        tanBeta = _math.tan(beta)
        tanAlpha = (f/(u - f))*tanBeta
        return _math.atan(tanAlpha)

    @staticmethod
    def _rad2deg(x):
        return x*180.0/_math.pi

    @staticmethod
    def _deg2rag(x):
        return x*_math.pi/180.0

    @staticmethod
    def cop_restore(alpha, h, k, atype='d'):
        """calculate lateral and longitudinal shifts required to restore the
        COP in cameras with base or asymmetrical tilts and swings.

        PARAMETERS
        ----------
        alpha : float
            angle of the lens tilt with proper sign convention. clockwise is +ve.
            By default, it is assumed that the angle is specified in degrees.
            If the angle ``alpha`` is specified in radians, use ``r`` for the
            parameter ``atype``
        h : float
            the distance from the center of the lens standard (O) to the pivot.
            If the pivot point lies above the lens center, ``h`` is positive.
        k : float
            the distance from the lens center to the COP. If the COP lies to
            the right of the lens center, then ``k`` is positive.


        Returns
        -------
        deltaZ : float
            shift along the longitudinal direction required to restore COP
            position
        deltaY : float
            shift along the y-axis (vertical direction) required to restore COP
            position

        Notes
        -----
        1. Currently the function returns COP restoration only for lens tilts
           and not for swings.
        2. The following figure shows a case when both ``h`` and ``k`` are
           positive
           ::

                |
                |
                +  <- pivot
                |
                |  h
                |
              O |-------o <- COP
                |   k
                |
                |
                |
                |

        """
        assert atype in ['d', 'r']
        alpha = alpha if atype=='r' else _np.deg2rad(alpha)
        deltaZ = h*_math.sin(alpha) + k*(1.0 - _math.cos(alpha))
        deltaY = k*_math.sin(alpha) - h*(1.0 - _math.cos(alpha))
        shift = _co.namedtuple('shift', ['deltaZ', 'deltaY'])
        return shift(deltaZ, deltaY)


# ###########################################
#   TEST FUNCTIONS
# ##########################################

def _test_gaussian_lens_formula():
    """Test gaussian_lens_formula function"""
    v = gaussian_lens_formula(u=10e20, f=10)
    _nt.assert_equal(v, 10.0)
    v = gaussian_lens_formula(u=5000.0, f=100)
    _nt.assert_almost_equal(v, 102.04081632653062, decimal=5)
    u = gaussian_lens_formula(v=200, f=200)
    _nt.assert_equal(u, 10e20)
    f = gaussian_lens_formula(u=10e20, v=40)
    _nt.assert_almost_equal(f, 40, decimal=5)
    print("test_gaussian_lens_formula() is successful")

def _test_scheimpflug():
    """Test the schimpflug camera class and methods"""
    # Try creating Scheimpflug with both alpha and beta specified
    try:
        Scheimpflug(f=180, u=3000, alpha=10, beta=20)
    except Exception as e:
        _nt.assert_equal(e.__class__, TypeError)
    cam0 = Scheimpflug(f=180)
    _nt.assert_equal(cam0.u, Scheimpflug.infty)
    _nt.assert_equal(cam0.v, 180)
    _nt.assert_equal(cam0.alpha, 0)
    _nt.assert_equal(cam0.beta, 0)
    # Create Scheimpflug camera object with only f, and u specified
    cam1 = Scheimpflug(f=180, u=3000)
    _nt.assert_equal(cam1.u, 3000)
    _nt.assert_almost_equal(cam1.v, 191.4893617)
    _nt.assert_equal(cam1.alpha, 0)
    _nt.assert_equal(cam1.beta, 0)
    # Create 2 Scheimpflug cameras with f and u and alpha specified in radians
    # and degrees respectively
    ang_d = 3.0
    ang_r = _np.deg2rad(ang_d)
    cam2 = Scheimpflug(f=180, u=3000, alpha=ang_r, atype='r')
    cam3 = Scheimpflug(f=180, u=3000, alpha=3)
    _nt.assert_almost_equal(_np.rad2deg(cam2.alpha), cam3.alpha)
    _nt.assert_almost_equal(cam3.alpha, 3)
    _nt.assert_almost_equal(cam3.beta, 39.387884973)
    cam3.alpha = 5
    _nt.assert_almost_equal(cam3.beta, 53.8863395127)

if __name__ == '__main__':
    import numpy.testing as _nt
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    #_test_gaussian_lens_formula()
    _test_scheimpflug()