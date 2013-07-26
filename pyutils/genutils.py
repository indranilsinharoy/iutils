# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          genutils.py
# Purpose:       General (Python) Utility Functions
#
# Author:        Indranil Sinharoy
#
# Created:       25/07/2013
# Last Modified:
# Copyright:     (c) Indranil Sinharoy 2013
# Licence:       MIT License
#-------------------------------------------------------------------------------
from __future__ import division, print_function
import sys


def is64bit():
    """Returns True if 64 bit, False if not (i.e. if 32 bit Python environment)"""
    # As per the discussion at (http://stackoverflow.com/questions/1842544/how-do-i-
    # detect-if-python-is-running-as-a-64-bit-application) I think the following is
    # the best way to determine the "bitness" of the system.
    return sys.maxsize > 2**31 - 1








# ---------------------------
#   TEST FUNCTIONS
# ---------------------------
def _test_is64bit():
    """For obvious reasons, this is not an automated test"""
    bitness = 64 if is64bit() else 32    
    print("This is %s bit system" % bitness)
    
    
if __name__=="__main__":
    _test_is64bit()
