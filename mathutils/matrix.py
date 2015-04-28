# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:      matrix.py
# Purpose:   matrix and linear algebra based math utility programs
#
# Author:      Indranil Sinharoy
#
# Created:        22/09/2012
# Last Modified:  12/24/2014
# Copyright:      (c) Indranil Sinharoy, 2012 - 2015
# Licence:        MIT License
#-------------------------------------------------------------------------------
""" matrix and linear algebra based math utility programs
"""
from __future__ import division, print_function
import numpy as _np
import sympy as _sym

#Import rogue library (test matrix library similar to Matlab's gallery()) if
#available
try:
    import rogues.matrices as rm
except ImportError:
    print("Did not import rogues.matrices")
else:
    print("Imported rogues.matrices as rm")
try:
    import rogues.utils as ru
except ImportError:
    print("Did not import rogues.utils")
else:
    print("Imported rogues.utils as ru")

def gallery(n=3):
    """
    Similar to MATLAB's test matrix gallery(3) and gallery(5). gallery(3) returns
    a badly conditioned 3-by-3 matrix. gallery(5) is an interesting eigenvalue
    problem. The Matlab documentation says, "Try to find its EXACT eigenvalues
    and eigenvectors. 5-by-5 matrix. For other test matrices, import the rogues
    package.
    """
    if (n==5):
        return _np.matrix([[  -9,     11,    -21,     63,    -252],
                         [  70,    -69,    141,   -421,    1684],
                         [-575,    575,  -1149,   3451,  -13801],
                         [3891,  -3891,   7782, -23345,   93365],
                         [1024,  -1024,   2048,  -6144,   24572]])
    else:
        return _np.matrix([[-149,-50,-154],
                         [537,180,546],
                         [-27,-9,-25]])

## Test the functions
def _test_gallery():
    print("test gallery(3)")
    print(gallery(3))
    print("test gallery(5)")
    print(gallery(5))

if __name__ == '__main__':
    import numpy.testing as _nt
    _test_gallery()
