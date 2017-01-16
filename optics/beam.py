# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          beamoptics.py
# Purpose:       Beam Optics Utility Functions
#
# Author:        Indranil Sinharoy
#
# Created:       06/19/2014
# Last Modified: 06/28/2015
# Copyright:     (c) Indranil Sinharoy, 2014, 2015
# Licence:       MIT License
#-------------------------------------------------------------------------------
from __future__ import division, print_function
import math as _math
import numpy as _np
from iutils.pyutils.general import approx_equal as _approx_equal

class GaussianBeam(object):
    """Gaussian Beam (perfect TEM00 mode) class"""
    def __init__(self, waistDiameter=1, wavelength=635e-6, power=1):
        """
        Parameters
        ----------
        waistDiameter : float, optional
            waist diameter (``2*w0`` or spot size or minimum beam diameter)
            in mm, where ``w0`` is called the "waist" (waist radius) in Zemax
        wavelength : float, optional
            wavelength in mm, default=635e-6 mm
        power : float, optional
            power in milli-watts
        """
        # protected variables
        self._wavelen = None
        self._waistDia = None
        self._power = None

        self.wavelength = wavelength
        self.waistDiameter = waistDiameter
        self.power = power                  # power in milli Watts

    @property
    def wavelength(self):
        return self._wavelen

    @wavelength.setter
    def wavelength(self, value):
        self._wavelen = value    

    @property
    def waistDiameter(self):
        return self._waistDia

    @waistDiameter.setter
    def waistDiameter(self, value):
        self._waistDia = value

    @property
    def power(self):
        return self._power
    
    @power.setter
    def power(self, value):
        self._power = value

    @property
    def w0(self):
        """waist radius"""
        return self._waistDia/2

    @property
    def rayleigh(self):
        """rayleigh range in mm
        """
        return (_math.pi*self.w0**2)/self._wavelen

    @property
    def divergence(self):
        """far-field beam divergence (half-angle) in radians
        """
        return _math.atan(self.w0/self.rayleigh)

    @property
    def divergenceInDeg(self):
        """far-field beam divergence (half angle) in degrees
        """
        return self.divergence*180/_math.pi

    @property
    def dof(self):
        """depth of field or confocal parameter
        """
        return 2*self.rayleigh

    @property
    def bpp(self):
        """beam parameter product (BPP) in mm mrad.
        It is a quality measure -- higher the BPP, lower the quality
        """
        return self.w0 * self.divergence * 1000

    def get_beam_diameter(self, z):
        """beam diameter defined by `1/e^2` intensity at a distance ``z``
        mm from the waist. Half of this quantity is also called the "Size"
        in Zemax

        Parameters
        ----------
        z : float
            distance z, in mm, from the waist

        Returns
        -------
        beam_diameter : float
            the diameter, in mm, of the beam at a distance ``z`` from the
            waist
        """
        return 2*self.w0*_math.sqrt(1 + (z/self.rayleigh)**2)

    def get_phase_roc(self, z):
        """beam's phase radius of curvature at a distance of ``z`` mm from
        waist. This is called as the "Radius" in Zemax

        Parameters
        ----------
        z : float
            distance z, in mm, from the waist

        Returns
        -------
        phaseRoc : float
            the phase radius of curvature in mm at a distance ``z`` from
            the waist
        """
        try:
            phaseRoc = z*(1 + (self.rayleigh/z)**2)
        except ZeroDivisionError:
            phaseRoc = _np.inf
        return phaseRoc

    def get_gouy_shift(self, z):
        """Gouy shift at a distance ``z`` mm from the waist

        Parameters
        ----------
        z : float
            distance z, in mm, from the waist

        Returns
        -------
        gouy_shift : float
            the gouy shift in phase at a distance z mm from the waist
        """
        return _math.atan2(z/self.rayleigh)

    def get_intensity(self, rho=0, z=0):
        """beam intensity

        Parameters
        ----------
        rho : float, optional
            radial distance from the beam axis in mm
        z : float, optional
            distance in mm

        Returns
        -------
        intensity : float
            intensity at radial distance ``rho`` and axial distance ``z``

        Notes
        -----
        The total optical power needs to be defined
        """
        w_z = self.get_beam_diameter(z)/2
        return (2*self._power/_math.pi*w_z**2)*_math.exp(-2*rho**2 / w_z**2)



class HeNe(GaussianBeam):
    """Helium-Neon (HeNe) Laser"""
    def __init__(self, waistDiameter, power=None):
        """HeNe laser

        Parameters
        ----------
        waistDiameter : float
            waist diameter (2*W_0 or spot size or minimum beam diameter) in mm
        power : float, optional
            total power
        """
        super(HeNe, self).__init__(waistDiameter, 632.8e-6, power)


#%% TESTS
def _test_GaussianBeam():
    beam = GaussianBeam(waistDiameter=2, wavelength=1064e-6,)
    # the following values were verified with the calculator at
    # http://www.rp-photonics.com/gaussian_beams.html
    assert _approx_equal(beam.rayleigh, 2.95262467443e3, 1e-10)
    assert _approx_equal(beam.divergence, 0.00033868170595, 1e-10)
    assert _approx_equal(beam.bpp, 0.33868170595, 1e-10)
    assert _approx_equal(beam.get_phase_roc(1000), 9.71799246803e3, 1e-10) # phase ROC @ 1 m
    assert _approx_equal(beam.dof, 5.9052493489e3, 1e-10)
    # Test change of parameters (values verified with Zemax)
    gauss = GaussianBeam(2, 550e-6)
    assert gauss.wavelength == 550e-6
    assert gauss.w0 == 1.0
    assert _approx_equal(gauss.divergence, 1.75070435612e-4, 1e-10)
    assert _approx_equal(gauss.rayleigh, 5.71198664289e3, 1e-10)
    assert _approx_equal(gauss.get_phase_roc(1000), 3.36267914086e4, 1e-10) # phase ROC @ 1 m
    beamDiam1m = gauss.get_beam_diameter(1000)
    assert _approx_equal(beamDiam1m, 2.0304183392, 1e-10)
    gauss.waistDiameter = 0.2 # change beam parameter using property
    assert gauss.w0 == 0.1
    assert _approx_equal(gauss.divergence, 1.750702585398e-3, 1e-10)
    beamDiam1m = gauss.get_beam_diameter(1000)
    assert _approx_equal(beamDiam1m, 3.50711608315, 1e-10)
    # test changing wavelength using properties
    gauss.wavelength = 500e-6
    assert _approx_equal(gauss.divergence, 1.59154808711e-3, 1e-10)
    print('test_GaussianBeam Class successful')

def _test_HeNe():
    heNe = HeNe(0.1, 1)
    assert _approx_equal(heNe.divergence, 4.02850812668e-3, 1e-10)
    heNe2 = HeNe(0.48, 0.8)
    assert _approx_equal(heNe2.divergence, 8.39276869513e-4, 1e-10)
    print('test_HeNe Class successful')

if __name__ == "__main__":
    #import numpy.testing as nt
    _test_GaussianBeam()
    _test_HeNe()