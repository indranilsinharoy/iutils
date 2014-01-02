# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          opticsPlottingUtils.py
# Purpose:       collection of plotting and visualization utilities for optics.
#
# Author:        Indranil Sinharoy
#
# Created:       01/02/2014
# Last Modified: 01/02/2014
# Copyright:     (c) Indranil Sinharoy 2014
# Licence:       MIT License
#-------------------------------------------------------------------------------
"""collection of plotting and visualization utilities for optics.
"""
from __future__ import division, print_function
import warnings
import numpy as _np
import matplotlib as _mplb
import matplotlib.pylab as _plt

def intensityPSF_Blues(N=1000):
    """Return ColorBrewer 2.0 9-class Blues based matplotlib colormap optimized for showing intensity PSF
    
    This colormap is a sequential colormap, however it has more color distribution near zero.
    
    Parameters
    ----------
    N : Integer
        Length of the colormap
        
    Returns
    -------
    psfblues : matplotlib colormap
    """
    col_seq = [(  0/255.,  20/255.,  80/255.), (  8/255.,  48/255., 107/255.), 
               (  8/255.,  81/255., 156/255.), ( 33/255., 113/255., 181/255.), 
               ( 66/255., 146/255., 198/255.), (107/255., 174/255., 214/255.), 
               (158/255., 202/255., 225/255.), (198/255., 219/255., 239/255.), 
               (222/255., 235/255., 247/255.), (247/255., 251/255., 255/255.)]

    cdict = {'red':   ((0.00, col_seq[0][0], col_seq[0][0]), 
                       (0.02, col_seq[1][0], col_seq[1][0]), 
                       (0.06, col_seq[2][0], col_seq[2][0]), 
                       (0.10, col_seq[3][0], col_seq[3][0]), 
                       (0.20, col_seq[4][0], col_seq[4][0]), 
                       (0.30, col_seq[5][0], col_seq[5][0]), 
                       (0.50, col_seq[6][0], col_seq[6][0]), 
                       (0.75, col_seq[7][0], col_seq[7][0]), 
                       (0.90, col_seq[8][0], col_seq[8][0]),
                       (1.00, col_seq[9][0], col_seq[9][0])),
             'green': ((0.00, col_seq[0][1], col_seq[0][1]), 
                       (0.02, col_seq[1][1], col_seq[1][1]), 
                       (0.06, col_seq[2][1], col_seq[2][1]), 
                       (0.10, col_seq[3][1], col_seq[3][1]), 
                       (0.20, col_seq[4][1], col_seq[4][1]), 
                       (0.30, col_seq[5][1], col_seq[5][1]), 
                       (0.50, col_seq[6][1], col_seq[6][1]), 
                       (0.75, col_seq[7][1], col_seq[7][1]), 
                       (0.90, col_seq[8][1], col_seq[8][1]),
                       (1.00, col_seq[9][1], col_seq[9][1])),    
             'blue':  ((0.00, col_seq[0][2], col_seq[0][2]), 
                       (0.02, col_seq[1][2], col_seq[1][2]), 
                       (0.06, col_seq[2][2], col_seq[2][2]), 
                       (0.10, col_seq[3][2], col_seq[3][2]), 
                       (0.20, col_seq[4][2], col_seq[4][2]), 
                       (0.30, col_seq[5][2], col_seq[5][2]), 
                       (0.50, col_seq[6][2], col_seq[6][2]), 
                       (0.75, col_seq[7][2], col_seq[7][2]), 
                       (0.90, col_seq[8][2], col_seq[8][2]),
                       (1.00, col_seq[9][2], col_seq[9][2]))}
                       
    psfblues = _mplb.colors.LinearSegmentedColormap('psfblues', cdict, N)
    return psfblues
    
def intensityPSF_BlRd(N=1000):
    """Return cool to warm colormap based matplotlib colormap optimized for showing intensity PSF
    
    This colormap is a diverging colormap based on "Diverging Maps for Scientific Visualization," 
    by K Moreland; however it has more color distribution near zero and some values have been 
    modified.
    
    Parameters
    ----------
    N : Integer
        Length of the colormap
        
    Returns
    -------
    psfblrd : matplotlib colormap
    """  
    col_seq = [( 59/255.,  76/255., 192/255.), ( 68/255.,  90/255., 204/255.), 
               ( 77/255., 104/255., 215/255.), ( 87/255., 117/255., 225/255.), 
               ( 98/255., 130/255., 234/255.), (108/255., 142/255., 241/255.), 
               (119/255., 154/255., 247/255.), (130/255., 165/255., 251/255.), 
               (141/255., 176/255., 254/255.), (152/255., 185/255., 255/255.),
               (163/255., 194/255., 255/255.), (174/255., 201/255., 253/255.),
               (184/255., 208/255., 249/255.), (194/255., 213/255., 244/255.),
               (204/255., 217/255., 238/255.), (213/255., 219/255., 230/255.),
               (221/255., 221/255., 221/255.), (229/255., 216/255., 209/255.),
               (236/255., 211/255., 197/255.), (241/255., 204/255., 185/255.),
               (245/255., 196/255., 173/255.), (247/255., 187/255., 160/255.),
               (247/255., 177/255., 148/255.), (247/255., 166/255., 135/255.),
               (244/255., 154/255., 123/255.), (241/255., 141/255., 111/255.),
               (236/255., 127/255.,  99/255.)]
    
    cdict = {'red':   ((0.00000000, col_seq[0][0], col_seq[0][0]), 
                       (0.00769231, col_seq[1][0], col_seq[1][0]), 
                       (0.01538462, col_seq[2][0], col_seq[2][0]), 
                       (0.02307692, col_seq[3][0], col_seq[3][0]), 
                       (0.03076923, col_seq[4][0], col_seq[4][0]), 
                       (0.03846154, col_seq[5][0], col_seq[5][0]), 
                       (0.04615385, col_seq[6][0], col_seq[6][0]), 
                       (0.05384615, col_seq[7][0], col_seq[7][0]), 
                       (0.06153846, col_seq[8][0], col_seq[8][0]),
                       (0.06923077, col_seq[9][0], col_seq[9][0]),
                       (0.07692308, col_seq[10][0], col_seq[10][0]), 
                       (0.08461538, col_seq[11][0], col_seq[11][0]), 
                       (0.09230769, col_seq[12][0], col_seq[12][0]), 
                       (0.10000000, col_seq[13][0], col_seq[13][0]), 
                       (0.10769231, col_seq[14][0], col_seq[14][0]), 
                       (0.18205128, col_seq[15][0], col_seq[15][0]), 
                       (0.25641026, col_seq[16][0], col_seq[16][0]), 
                       (0.33076923, col_seq[17][0], col_seq[17][0]), 
                       (0.40512821, col_seq[18][0], col_seq[18][0]),
                       (0.47948718, col_seq[19][0], col_seq[19][0]),
                       (0.55384615, col_seq[20][0], col_seq[20][0]), 
                       (0.62820513, col_seq[21][0], col_seq[21][0]), 
                       (0.70256410, col_seq[22][0], col_seq[22][0]), 
                       (0.77692308, col_seq[23][0], col_seq[23][0]), 
                       (0.85128205, col_seq[24][0], col_seq[24][0]), 
                       (0.92564103, col_seq[25][0], col_seq[25][0]), 
                       (1.00000000, col_seq[26][0], col_seq[26][0])),
             'green': ((0.00000000, col_seq[0][1], col_seq[0][1]), 
                       (0.00769231, col_seq[1][1], col_seq[1][1]), 
                       (0.01538462, col_seq[2][1], col_seq[2][1]), 
                       (0.02307692, col_seq[3][1], col_seq[3][1]), 
                       (0.03076923, col_seq[4][1], col_seq[4][1]), 
                       (0.03846154, col_seq[5][1], col_seq[5][1]), 
                       (0.04615385, col_seq[6][1], col_seq[6][1]), 
                       (0.05384615, col_seq[7][1], col_seq[7][1]), 
                       (0.06153846, col_seq[8][1], col_seq[8][1]),
                       (0.06923077, col_seq[9][1], col_seq[9][1]),
                       (0.07692308, col_seq[10][1], col_seq[10][1]), 
                       (0.08461538, col_seq[11][1], col_seq[11][1]), 
                       (0.09230769, col_seq[12][1], col_seq[12][1]), 
                       (0.10000000, col_seq[13][1], col_seq[13][1]), 
                       (0.10769231, col_seq[14][1], col_seq[14][1]), 
                       (0.18205128, col_seq[15][1], col_seq[15][1]), 
                       (0.25641026, col_seq[16][1], col_seq[16][1]), 
                       (0.33076923, col_seq[17][1], col_seq[17][1]), 
                       (0.40512821, col_seq[18][1], col_seq[18][1]),
                       (0.47948718, col_seq[19][1], col_seq[19][1]),
                       (0.55384615, col_seq[20][1], col_seq[20][1]), 
                       (0.62820513, col_seq[21][1], col_seq[21][1]), 
                       (0.70256410, col_seq[22][1], col_seq[22][1]), 
                       (0.77692308, col_seq[23][1], col_seq[23][1]), 
                       (0.85128205, col_seq[24][1], col_seq[24][1]), 
                       (0.92564103, col_seq[25][1], col_seq[25][1]), 
                       (1.00000000, col_seq[26][1], col_seq[26][1])),    
             'blue':  ((0.00000000, col_seq[0][2], col_seq[0][2]), 
                       (0.00769231, col_seq[1][2], col_seq[1][2]), 
                       (0.01538462, col_seq[2][2], col_seq[2][2]), 
                       (0.02307692, col_seq[3][2], col_seq[3][2]), 
                       (0.03076923, col_seq[4][2], col_seq[4][2]), 
                       (0.03846154, col_seq[5][2], col_seq[5][2]), 
                       (0.04615385, col_seq[6][2], col_seq[6][2]), 
                       (0.05384615, col_seq[7][2], col_seq[7][2]), 
                       (0.06153846, col_seq[8][2], col_seq[8][2]),
                       (0.06923077, col_seq[9][2], col_seq[9][2]),
                       (0.07692308, col_seq[10][2], col_seq[10][2]), 
                       (0.08461538, col_seq[11][2], col_seq[11][2]), 
                       (0.09230769, col_seq[12][2], col_seq[12][2]), 
                       (0.10000000, col_seq[13][2], col_seq[13][2]), 
                       (0.10769231, col_seq[14][2], col_seq[14][2]), 
                       (0.18205128, col_seq[15][2], col_seq[15][2]), 
                       (0.25641026, col_seq[16][2], col_seq[16][2]), 
                       (0.33076923, col_seq[17][2], col_seq[17][2]), 
                       (0.40512821, col_seq[18][2], col_seq[18][2]),
                       (0.47948718, col_seq[19][2], col_seq[19][2]),
                       (0.55384615, col_seq[20][2], col_seq[20][2]), 
                       (0.62820513, col_seq[21][2], col_seq[21][2]), 
                       (0.70256410, col_seq[22][2], col_seq[22][2]), 
                       (0.77692308, col_seq[23][2], col_seq[23][2]), 
                       (0.85128205, col_seq[24][2], col_seq[24][2]), 
                       (0.92564103, col_seq[25][2], col_seq[25][2]), 
                       (1.00000000, col_seq[26][2], col_seq[26][2]))}
                   
    psfblrd = _mplb.colors.LinearSegmentedColormap('psfblrd', cdict, N)
    return psfblrd



    
    

