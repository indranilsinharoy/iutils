# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          beamoptics.py
# Purpose:       Beam Optics Utility Functions
#
# Author:        Indranil Sinharoy
#
# Created:       06/19/2014
# Last Modified: 06/20/2014
# Copyright:     (c) Indranil Sinharoy, 2014
# Licence:       MIT License
#-------------------------------------------------------------------------------
from __future__ import division, print_function
import math as _math
import numpy as _np

class GaussianBeam(object):
    """Gaussian Beam (perfect TEM00 mode) class"""
    def __init__(self, wavelen=635e-6, waistDia=1, power=None):
        """
        Parameters
        ----------
        wavelen : float, optional
            wavelength in mm, default=635e-6 mm
        waistDia : float, optional
            waist diameter (``2*w0`` or spot size or minimum beam diameter)
            in mm, where ``w0`` is called the "waist" (waist radius) in
            Zemax
        power : float, optional
            Total power
        """
        self._wavelen = wavelen
        self._waistDia = waistDia
        parameters = _calculate_parameters(waistDia, wavelen)
        self._waistRad, self._rayl, self._div = parameters
        self._power = power  # power in milli Watts

    @property
    def w0(self):
        """waist"""
        return self._waistRad

    @property
    def waistDiameter(self):
        return self._waistDia

    @waistDiameter.setter
    def waistDiameter(self, value):
        self._waistDia = value
        parameters = _calculate_parameters(value, self._wavelen)
        self._waistRad, self._rayl, self._div = parameters

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, value):
        self._power = value

    @property
    def rayleigh(self):
        """rayleigh range
        """
        return self._rayl

    @property
    def divergence(self):
        """far-field beam divergence (half-angle) in radians
        """
        return self._div

    @property
    def divergenceInDeg(self):
        """far-field beam divergence (half angle) in degrees
        """
        return self._div*180/_math.pi

    @property
    def dof(self):
        """depth of field or confocal parameter
        """
        return 2*self._rayl

    @property
    def bpp(self):
        """beam parameter product (BPP) in mm mrad.
        It is a quality measure -- higher the BPP, lower the quality
        """
        return self._waistRad*self._div*1000

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
        return 2*self._waistRad*_math.sqrt(1 + (z/self._rayl)**2)

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
            phaseRoc = z*(1 + (self._rayl/z)**2)
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
        return _math.atan2(z/self._rayl)

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
        w_z = self._get_beam_diameter(z)/2
        p = self._power
        if p:
            return (2*p/_math.pi*w_z**2)*_math.exp(-2*rho**2 / w_z**2)


# Helper functions

def _calculate_parameters(waistDia, wavelen):
    waist_rad = waistDia/2.0
    rayl = (_math.pi*waist_rad**2)/wavelen
    div = _math.atan(waist_rad/rayl)
    return waist_rad, rayl, div

class HeNe(GaussianBeam):
    """Helium-Neon (HeNe) Laser"""
    def __init__(self, waistDia, power=None):
        """HeNe laser

        Parameters
        ----------
        waistDia : float, optional
            waist diameter (2*W_0 or spot size or minimum beam diameter)
            in mm
        power : float, optional
            Total power
        """
        super(HeNe, self).__init__(632.8e-6, waistDia, power)

## TESTS
def _test_GaussianBeam():
    beam = GaussianBeam(wavelen=1064e-6, waistDia=2)
    # the following values were verified with the calculator at
    # http://www.rp-photonics.com/gaussian_beams.html
    rayl_rng_diff = abs(beam.rayleigh - 2952.62467443)
    assert rayl_rng_diff < 1e-5, 'Value: {}'.format(rayl_rng_diff)
    bdiv_diff = abs(beam.divergence - 0.00033868170595)
    assert bdiv_diff < 1e-5
    bpp_diff = abs(beam.bpp - 0.33868170595)
    assert bpp_diff < 1e-5
    assert beam.get_phase_roc(0) == _np.inf # phase ROC @ waist
    proc_diff = abs(beam.get_phase_roc(1000) - 9717.99246803)
    assert proc_diff < 1e-5 # phase ROC @ 1 m
    dof_diff = abs(beam.dof - 5905.24934885)
    assert dof_diff < 1e-5

def _test_HeNe():
    heNe = HeNe(0.1, 1)
    print("HeNe Laser, divergence = ", heNe.divergence)
    heNe2 = HeNe(0.48, 0.8)
    print("HeNe 2 Laser, divergence = ", heNe2.divergence)

if __name__ == "__main__":
    #import numpy.testing as nt
    _test_GaussianBeam()
    _test_HeNe()