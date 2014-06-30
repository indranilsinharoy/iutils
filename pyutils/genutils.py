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
from os import listdir, path, getcwd
import hashlib
from subprocess import call


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

def trial():
     cdir = getcwd()
     print(cdir)


def _convert(f):
    """return 1 on success, 0 on fail"""
    try:
        cmd = "ipython nbconvert --to html --quiet \"{fn}\"".format(fn=f)
        #print(cmd) # for debugging
        # Wait for the command(s) to get executed ...
        call(cmd, shell=True)
    except:
        return 0
    else:
        return 1

def _getMD5(filename):
    """returns the MD5 signature of the file"""
    with open(filename) as f:
        d = f.read()
    h = hashlib.md5(d).hexdigest() # HEX string representation of the hash


def nbconvert():
    # Get the current directory
    #cdir = path.dirname(path.realpath(__file__))
    cdir =  getcwd()
    # Get a list of .ipynb files
    files2convert = [f for f in listdir(cdir) if f.endswith('.ipynb')]
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
    s = raw_input("Press ENTER to close the appliation ...")

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