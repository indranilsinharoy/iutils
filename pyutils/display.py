# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          display.py
# Purpose:       Display (Python) Utility Functions
#
# Author:        Indranil Sinharoy
#
# Created:       04/20/2014
# Copyright:     (c) Indranil Sinharoy 2014
# Licence:       MIT License
#-------------------------------------------------------------------------------
from __future__ import division, print_function
import Tkinter

def getPrimaryScreenResolution():
    root = Tkinter.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    return screen_width, screen_height


# TEST FUNCTIONS
def _test_getPrimaryScreenResolution():
    width, height = getPrimaryScreenResolution()
    print("Width: ", width)
    print("Height: ", height)



if __name__ == '__main__':
    import numpy.testing as nt
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    _test_getPrimaryScreenResolution()
