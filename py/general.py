# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------------------
# Name:          genutils.py
# Purpose:       General (Python) Utility Functions
#
# Author:        Indranil Sinharoy
#
# Created:       07/25/2013
# Last Modified: 04/30/2015
# Copyright:     (c) Indranil Sinharoy 2013 - 2015
# Licence:       MIT License
#-----------------------------------------------------------------------------------------
from __future__ import division, print_function
import sys as _sys
from os import listdir as _listdir, getcwd as _getcwd
import hashlib as _hashlib
from subprocess import call as _call
import random as _random


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
    return _sys.maxsize > 2**31 - 1


def find_zero_crossings(f, a, b, func_args=(), n=100):
    """Moved to plottingUtils.py
    """
    print("Function moved to plottingUtils")
    return None

def c_binary_string(n, numBits=32):
    """Return `n` as a clean 32-bit/64-bit binary number, without a leading '0b'.

    Usage: c_binary_string(n [,numBits])->binary_string

    Parameters
    ----------
    n : positive or negative integer
    numBits (optional) : 32 or 64 indicating the string length.

    Returns
    -------
    binary_string (string) : the binary string representation of the number `n`.

    Note
    ----
    This function has been borrowed from Brandon Rhodes' great talk called
    "The Mighty Dictionary" in PyCon 2010.
    Link: http://pyvideo.org/video/276/the-mighty-dictionary-55
    """
    if n < 0:
        n = 2 ** numBits + n
    if numBits == 32:
        return '{0:0>32}'.format(bin(n)[2:])
    else:
        return '{0:0>64}'.format(bin(n)[2:])

def _convert(f):
    """return 1 on success, 0 on fail"""
    try:
        cmd = "ipython nbconvert --to html --quiet \"{fn}\"".format(fn=f)
        #print(cmd) # for debugging
        # Wait for the command(s) to get executed ...
        _call(cmd, shell=True)
    except:
        return 0
    else:
        return 1

def _getMD5(filename):
    """returns the MD5 signature of the file"""
    with open(filename) as f:
        d = f.read()
    h = _hashlib.md5(d).hexdigest() # HEX string representation of the hash
    return h


def nbconvert():
    # Get the current directory
    #cdir = path.dirname(path.realpath(__file__))
    cdir =  _getcwd()
    # Get a list of .ipynb files
    files2convert = [f for f in _listdir(cdir) if f.endswith('.ipynb')]
    # Convert the files within a try-except block
    count_files_successfully_converted = 0
    failedFiles = []
    for i, f in enumerate(files2convert):
        _getMD5(f)
        print(">>> [{}] Converting file ... ".format(i+1))
        if _convert(f):
            count_files_successfully_converted += 1
        else:
            failedFiles.append(f)

    # Print some human readable feedback
    print("\n")
    print("*******************************************")
    print("                  REPORT                   ")
    print("*******************************************")
    print("\nCurrent Directory: ", cdir)
    print("Number of IPython notebooks found: ", len(files2convert))
    print("Number of files successfully converted to html:",
    count_files_successfully_converted)
    print("Number of files failed to convert to html:", len(failedFiles))
    if failedFiles:
        print("Files that failed to convert:")
        for f in failedFiles:
            print(f)
    print("\nDONE!")
    raw_input("Press ENTER to close the appliation ...")

def remove_duplicates_in_list(seq):
    """Removes and returns a new list with duplicate elements removed and the
    order of elements in the sequence is preserved

    Parameters
    ----------
    seq : list
        the list

    Returns
    -------
    newSeq : list
        the new list with duplicate elements removed

    Examples
    --------
    >>> a = [1, 2, 4, 1, 2, 3, 6, 2, 2, 5, 5, 10, 3, 20, 21, 20, 8, 6]
    >>> gnu.remove_duplicates_in_list(a)
    [1, 2, 4, 3, 6, 5, 10, 20, 21, 8]
    """
    seen = set()
    seen_add = seen.add
    newSeq = [x for x in seq if x not in seen and not seen_add(x)]
    return newSeq


def set_small_values_to_zero(tol, *values):
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
    return [0.0 if abs(value) < tol else value for value in values]
    
def approx_equal(x, y, tol=1e-12):
    """compare two float values using relative difference as measure
    
    Parameters
    ----------
    x, y : floats
        floating point values to be compared
    tol : float
        tolerance
    
    Returns
    -------
    rel_diff : bool
        ``True`` if ``x`` and ``y`` are approximately equal within the tol   
    
    Notes
    -----
    1. relative difference: http://en.wikipedia.org/wiki/Relative_change_and_difference
    3. In future, this function could be replaced by a standard library function. See
       PEP0485 for details. https://www.python.org/dev/peps/pep-0485/
    """
    return abs(x - y) <= max(abs(x), abs(y)) * tol

#%% TEST FUNCTIONS

def _test_is64bit():
    """For obvious reasons, this is not an automated test. i.e. it requires a visual 
    inspection"""
    print("\nTest for 32/64 bitness of Python system")
    bitness = 64 if is64bit() else 32
    print("This is %s bit system" % bitness)

def _test_remove_duplicates_in_list():
    a = [1, 2, 4, 1, 2, 3, 6, 2, 2, 5, 5, 10, 3, 20, 21, 20, 8, 6]
    exp_ret = [1, 2, 4, 3, 6, 5, 10, 20, 21, 8]
    ret = remove_duplicates_in_list(a)
    _nt.assert_array_equal(exp_ret, ret)
    print("test_remove_duplicates_in_list() successful")
    
def _test_set_small_values_to_zero():
    """Test helper function _set_small_values_to_zero()
    """
    tol = 1e-12
    a, b, c, d = set_small_values_to_zero(tol, 1.0, 0.0, tol, 1e-13)
    assert a == 1.0
    assert b == 0.0
    assert c == tol
    assert d == 0.0
    a, b, c, d = set_small_values_to_zero(tol, -1.0, -0.0, -tol, -1e-13)
    assert a == -1.0
    assert b == -0.0
    assert c == -tol
    assert d == 0.0
    print("test_set_small_values_to_zero() successful")
    
def _test_approx_equal():
    """test for function test_approx_equal()
    """
    a = _random.random()*100.0
    b = a + 1e-15
    assert approx_equal(a, b)
    c = a + 1e-3
    assert not approx_equal(a, c)
    print("test_approx_equal() successful")
    


if __name__=="__main__":
    import numpy.testing as _nt
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    _test_is64bit()
    _test_remove_duplicates_in_list()
    _test_set_small_values_to_zero()
    _test_approx_equal()