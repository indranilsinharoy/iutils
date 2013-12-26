# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          genutils.py
# Purpose:       General (Python) Utility Functions
#
# Author:        Indranil Sinharoy
#
# Created:       07/25/2013
# Last Modified: 11/20/2013
#                1. Moved find_zero_crossings() from here to plottingUtils.py
# Copyright:     (c) Indranil Sinharoy 2013
# Licence:       MIT License
#-------------------------------------------------------------------------------
from __future__ import division, print_function
import sys


def is64bit():
    """Returns True if 64 bit, False if not (i.e. if 32 bit Python environment)
    
    Usage: is64bit()->ret

    Parameters
    ----------
    None

    Returns
    -------
    ret : bool
        True if 64 bit environment, False otherwise.    
    """
    # As per the discussion at (http://stackoverflow.com/questions/1842544/how-do-i-
    # detect-if-python-is-running-as-a-64-bit-application) I think the following is
    # the best way to determine the "bitness" of the system.
    return sys.maxsize > 2**31 - 1


def find_zero_crossings(f, a, b, func_args=(), n=100):
    """Moved to plottingUtils.py
    """
    print("Function moved to plottingUtils")
    return None


# ---------------------------
#   TEST FUNCTIONS
# ---------------------------
def _test_is64bit():
    """For obvious reasons, this is not an automated test. i.e. it requires a visual inspection"""
    print("\nTest for 32/64 bitness of Python system")
    bitness = 64 if is64bit() else 32    
    print("This is %s bit system" % bitness)
 
 
    
if __name__=="__main__":
    import numpy.testing as nt
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    _test_is64bit()