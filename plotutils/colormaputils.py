# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          colormaputils.py
# Purpose:       collection of utility functions related to colormaps.
#
# Author:        Indranil Sinharoy
#
# Created:       01/03/2014
# Last Modified: 01/03/2014
# Copyright:     (c) Indranil Sinharoy, 2014
# Licence:       MIT License
#-------------------------------------------------------------------------------
"""collection of utility functions related to colormaps
"""
from __future__ import division, print_function
#import warnings
import numpy as _np
import matplotlib as _mpl
import matplotlib.pyplot as _plt
import matplotlib.colors as _mplc
from mpl_toolkits.mplot3d import Axes3D

def trace_colormap(cmap, figsize=(7,6), bcolor='w', show_vals=False, infoInTitle=None):
    """function to plot the trace of RGB values in the given matplotlib
    compatible colormap

    Parameters
    ----------
    cmap : colormap
      the colormap must be matplotlib compatible. i.e. it must be a
      matplotlib colormap or created using
      ``matplotlib.colors.LinearSegmentedColormap()``.
    figsize : tuple
        tuple indicating figure size
    bcolor : color
        figure background color
    show_vals : bool
        if True, corresponding values between 0 and 1 will be shown along
        the plot
    infoInTitle : string
        extra information to append to the standard title

    Returns
    -------
    None

    Notes
    -----
    The function creates a matplotlib figure with 3d axis  and traces
    the RGB values of the colormap.

    Examples
    --------
    traceColormap(mpl.cm.jet)
    """
    #Set figure characteristics
    fig = _plt.figure(figsize=figsize, facecolor=bcolor)
    #ax = fig.gca(projection='3d',axis_bgcolor='black')
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], axisbg=bcolor, projection='3d')

    # Ideally I would like to do a lot more with the grid. However, not all commands
    # are functioning as exected as it is still the early stages of development
    ax.grid(b=True)
    X = [cmap(i)[0] for i in range(cmap.N)]
    Y = [cmap(i)[1] for i in range(cmap.N)]
    Z = [cmap(i)[2] for i in range(cmap.N)]
    C = zip(X, Y, Z)

    for i in range(cmap.N - 1):
        c = C[i]
        ax.plot([X[i], X[i+1]], [Y[i], Y[i+1]], [Z[i], Z[i+1]],
                color=c, lw=(3.75-(500./256 -1)*0.25))
    if show_vals:
        for i in range(cmap.N - 1):
            if _np.mod(i, 10) == 0:
                ax.text(X[i], Y[i], Z[i], '{:1.3f}'.format(i/cmap.N), fontsize=8)
        i += 1
        ax.text(X[i], Y[i], Z[i], '{:1.3f}'.format(i/cmap.N), fontsize=8)

    # Figure decoration
    ax.set_xticks(_np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    ax.set_yticks(_np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    ax.set_zticks(_np.array([0.0, 0.25, 0.5, 0.75, 1.0]))

    # Mark the origin of the coordinate system
    ax.scatter3D([0], [0], s=30, c='#555555', zorder=20)

    # The planes on the side
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))

    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)

    ax.xaxis.set_tick_params(colors='red')
    ax.yaxis.set_tick_params(colors='lime')
    ax.zaxis.set_tick_params(colors='blue')

    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('lime')
    ax.zaxis.label.set_color('blue')

    ax.set_xlabel("Red", fontsize=13)
    ax.set_ylabel("Green", fontsize=13)
    ax.set_zlabel("Blue", fontsize=13)

    if infoInTitle:
        title = 'cmap: {} ({})'.format(cmap.name, infoInTitle)
    else:
        title = 'cmap: {}'.format(cmap.name)
    ax.set_title(title, color='#555555', fontsize=13)
    ax.azim = -40
    ax.dist = 11.2  #Required to do in order to prevent the labels from being cut off. (default is 10)
    _plt.show()


def get_colormap(name='moreland', N=256, **kwargs):
    """Function returns matplotlib compatible colormap of name `name`.

    Parameters
    ----------
    name : string, optional
      name of the colormap, e.g. ``moreland``, ``iron``, ``fire``, etc.
    N : integer, optional
      specifies the number of colors in the colormap
    kwargs : keyword arguments
        examples are 'sat' for saturation value between 0 and 1 for cplothsv
        colormap, etc.

    Returns
    -------
    colormap : matplotlib colormap

    """
    if name =='moreland':
        return _get_moreland(N)
    elif name =='iron':
        return _get_iron(N)
    elif name == 'fire':
      return _get_fire(N)
    elif name == 'cplothsv':
        sat = kwargs['sat'] if 'sat' in kwargs else 1.0 # saturation
        lmap = kwargs['linearPhaseMap'] if 'linearPhaseMap' in kwargs else True
        return _get_complex_function_hsv(N, sat, lmap)
    else:
        raise ValueError('Invalid colormap specified')

def get_colormap_description():
    cmap_str_des = """
    1. moreland : "Diverging Maps for Scientific Visualization," Kenneth Moreland
    2. iron : cool warm colormap used for thermographic imaging
    3. fire : diverging color map from ImageJ
    4. cplothsv : colormap for complex function plot using hsv domain mapping
    """
    return cmap_str_des

def _get_moreland(N):
    """Helper function for generating cool warm diverging colormap.

    Parameters
    ----------
    N : integer
        Number of colors in the colormap

    Returns
    -------
    colormap :
      matplotlib colormap

    References
    ----------
    "Diverging Maps for Scientific Visualization," Kenneth Moreland.
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
               (236/255., 127/255.,  99/255.), (229/255., 112/255.,  88/255.),
               (222/255.,  96/255.,  77/255.), (213/255.,  80/255.,  66/255.),
               (203/255.,  62/255.,  56/255.), (192/255.,  40/255.,  47/255.),
               (180/255.,   4/255.,  38/255.)]

    cdict = {'red':   ((0.00000, col_seq[0][0], col_seq[0][0]),
                       (0.03125, col_seq[1][0], col_seq[1][0]),
                       (0.06250, col_seq[2][0], col_seq[2][0]),
                       (0.09375, col_seq[3][0], col_seq[3][0]),
                       (0.12500, col_seq[4][0], col_seq[4][0]),
                       (0.15625, col_seq[5][0], col_seq[5][0]),
                       (0.18750, col_seq[6][0], col_seq[6][0]),
                       (0.21875, col_seq[7][0], col_seq[7][0]),
                       (0.25000, col_seq[8][0], col_seq[8][0]),
                       (0.28125, col_seq[9][0], col_seq[9][0]),
                       (0.31250, col_seq[10][0], col_seq[10][0]),
                       (0.34375, col_seq[11][0], col_seq[11][0]),
                       (0.37500, col_seq[12][0], col_seq[12][0]),
                       (0.40625, col_seq[13][0], col_seq[13][0]),
                       (0.43750, col_seq[14][0], col_seq[14][0]),
                       (0.46875, col_seq[15][0], col_seq[15][0]),
                       (0.50000, col_seq[16][0], col_seq[16][0]),
                       (0.53125, col_seq[17][0], col_seq[17][0]),
                       (0.56250, col_seq[18][0], col_seq[18][0]),
                       (0.59375, col_seq[19][0], col_seq[19][0]),
                       (0.62500, col_seq[20][0], col_seq[20][0]),
                       (0.65625, col_seq[21][0], col_seq[21][0]),
                       (0.68750, col_seq[22][0], col_seq[22][0]),
                       (0.71875, col_seq[23][0], col_seq[23][0]),
                       (0.75000, col_seq[24][0], col_seq[24][0]),
                       (0.78125, col_seq[25][0], col_seq[25][0]),
                       (0.81250, col_seq[26][0], col_seq[26][0]),
                       (0.84375, col_seq[27][0], col_seq[27][0]),
                       (0.87500, col_seq[28][0], col_seq[28][0]),
                       (0.90625, col_seq[29][0], col_seq[29][0]),
                       (0.93750, col_seq[30][0], col_seq[30][0]),
                       (0.96875, col_seq[31][0], col_seq[31][0]),
                       (1.00000, col_seq[32][0], col_seq[32][0])),
             'green': ((0.00000, col_seq[0][1], col_seq[0][1]),
                       (0.03125, col_seq[1][1], col_seq[1][1]),
                       (0.06250, col_seq[2][1], col_seq[2][1]),
                       (0.09375, col_seq[3][1], col_seq[3][1]),
                       (0.12500, col_seq[4][1], col_seq[4][1]),
                       (0.15625, col_seq[5][1], col_seq[5][1]),
                       (0.18750, col_seq[6][1], col_seq[6][1]),
                       (0.21875, col_seq[7][1], col_seq[7][1]),
                       (0.25000, col_seq[8][1], col_seq[8][1]),
                       (0.28125, col_seq[9][1], col_seq[9][1]),
                       (0.31250, col_seq[10][1], col_seq[10][1]),
                       (0.34375, col_seq[11][1], col_seq[11][1]),
                       (0.37500, col_seq[12][1], col_seq[12][1]),
                       (0.40625, col_seq[13][1], col_seq[13][1]),
                       (0.43750, col_seq[14][1], col_seq[14][1]),
                       (0.46875, col_seq[15][1], col_seq[15][1]),
                       (0.50000, col_seq[16][1], col_seq[16][1]),
                       (0.53125, col_seq[17][1], col_seq[17][1]),
                       (0.56250, col_seq[18][1], col_seq[18][1]),
                       (0.59375, col_seq[19][1], col_seq[19][1]),
                       (0.62500, col_seq[20][1], col_seq[20][1]),
                       (0.65625, col_seq[21][1], col_seq[21][1]),
                       (0.68750, col_seq[22][1], col_seq[22][1]),
                       (0.71875, col_seq[23][1], col_seq[23][1]),
                       (0.75000, col_seq[24][1], col_seq[24][1]),
                       (0.78125, col_seq[25][1], col_seq[25][1]),
                       (0.81250, col_seq[26][1], col_seq[26][1]),
                       (0.84375, col_seq[27][1], col_seq[27][1]),
                       (0.87500, col_seq[28][1], col_seq[28][1]),
                       (0.90625, col_seq[29][1], col_seq[29][1]),
                       (0.93750, col_seq[30][1], col_seq[30][1]),
                       (0.96875, col_seq[31][1], col_seq[31][1]),
                       (1.00000, col_seq[32][1], col_seq[32][1])),
             'blue':  ((0.00000, col_seq[0][2], col_seq[0][2]),
                       (0.03125, col_seq[1][2], col_seq[1][2]),
                       (0.06250, col_seq[2][2], col_seq[2][2]),
                       (0.09375, col_seq[3][2], col_seq[3][2]),
                       (0.12500, col_seq[4][2], col_seq[4][2]),
                       (0.15625, col_seq[5][2], col_seq[5][2]),
                       (0.18750, col_seq[6][2], col_seq[6][2]),
                       (0.21875, col_seq[7][2], col_seq[7][2]),
                       (0.25000, col_seq[8][2], col_seq[8][2]),
                       (0.28125, col_seq[9][2], col_seq[9][2]),
                       (0.31250, col_seq[10][2], col_seq[10][2]),
                       (0.34375, col_seq[11][2], col_seq[11][2]),
                       (0.37500, col_seq[12][2], col_seq[12][2]),
                       (0.40625, col_seq[13][2], col_seq[13][2]),
                       (0.43750, col_seq[14][2], col_seq[14][2]),
                       (0.46875, col_seq[15][2], col_seq[15][2]),
                       (0.50000, col_seq[16][2], col_seq[16][2]),
                       (0.53125, col_seq[17][2], col_seq[17][2]),
                       (0.56250, col_seq[18][2], col_seq[18][2]),
                       (0.59375, col_seq[19][2], col_seq[19][2]),
                       (0.62500, col_seq[20][2], col_seq[20][2]),
                       (0.65625, col_seq[21][2], col_seq[21][2]),
                       (0.68750, col_seq[22][2], col_seq[22][2]),
                       (0.71875, col_seq[23][2], col_seq[23][2]),
                       (0.75000, col_seq[24][2], col_seq[24][2]),
                       (0.78125, col_seq[25][2], col_seq[25][2]),
                       (0.81250, col_seq[26][2], col_seq[26][2]),
                       (0.84375, col_seq[27][2], col_seq[27][2]),
                       (0.87500, col_seq[28][2], col_seq[28][2]),
                       (0.90625, col_seq[29][2], col_seq[29][2]),
                       (0.93750, col_seq[30][2], col_seq[30][2]),
                       (0.96875, col_seq[31][2], col_seq[31][2]),
                       (1.00000, col_seq[32][2], col_seq[32][2]))}
    cwm = _mpl.colors.LinearSegmentedColormap('moreland', cdict, N)
    return cwm

def _get_iron(N):
    """Helper function for generating cool warm "Iron" colormap used
    in thermographic imaging. It is a diverging colormap.

    Parameters
    ----------
    N : integer

    Returns
    -------
    colormap : colormap
      matplotlib colormap

    References
    ----------
    # Evernote: Infrared camera/ Thermographic camera pseudo colormap
    # Link: https://www.evernote.com/shard/s82/nl/9016199/b73314ad-af9d-442d-b324-d8a278ade985
    """
    # Palette with 7 special shades and 120 shades in temp scale
    col_seq = [ (0/255., 0/255., 0/255.),
                (0/255., 0/255., 45/255.),   # 0, 0, 36
                (5/255., 0/255., 60/255.),   # 0, 0, 51
                (8/255., 0/255., 70/255.),   # 0, 0, 66
                (11/255., 0/255., 85/255.),  # 0, 0, 81
                (14/255., 0/255., 92/255.),  # 2, 0, 90
                (17/255., 0/255., 99/255.),  # 4, 0, 99
                (20/255., 0/255., 106/255.), # 7, 0, 106
                (23/255., 0/255., 115/255.), # 11, 0, 115
                (27/255., 0/255., 119/255.), # 14
                (30/255., 0/255., 123/255.), # 20
                (33/255., 0/255., 128/255.), # 27
                (37/255., 0/255., 133/255.), # 33
                (43/255., 0/255., 137/255.), # 41
                (48/255., 0/255., 140/255.),
                (55/255., 0/255., 143/255.),
                (61/255., 0/255., 146/255.),
                (66/255., 0/255., 149/255.),
                (72/255., 0/255., 150/255.),
                (78/255., 0/255., 151/255.),
                (84/255., 0/255., 152/255.),
                (91/255., 0/255., 153/255.),
                (97/255., 0/255., 155/255.),
                (104/255., 0/255., 155/255.),
                (110/255., 0/255., 156/255.),
                (115/255., 0/255., 157/255.),
                (122/255., 0/255., 157/255.),
                (128/255., 0/255., 157/255.),
                (134/255., 0/255., 157/255.),
                (139/255., 0/255., 157/255.),
                (146/255., 0/255., 156/255.),
                (152/255., 0/255., 155/255.),
                (157/255., 0/255., 155/255.),
                (162/255., 0/255., 155/255.),
                (167/255., 0/255., 154/255.),
                (171/255., 0/255., 153/255.),
                (175/255., 1/255., 152/255.),
                (178/255., 1/255., 151/255.),
                (182/255., 2/255., 149/255.),
                (185/255., 4/255., 149/255.),
                (188/255., 5/255., 147/255.),
                (191/255., 6/255., 146/255.),
                (193/255., 8/255., 144/255.),
                (195/255., 11/255., 142/255.),
                (198/255., 13/255., 139/255.),
                (201/255., 17/255., 135/255.),
                (203/255., 20/255., 132/255.),
                (206/255., 23/255., 127/255.),
                (208/255., 26/255., 121/255.),
                (210/255., 29/255., 116/255.),
                (212/255., 33/255., 111/255.),
                (214/255., 37/255., 103/255.),
                (217/255., 41/255., 97/255.),
                (219/255., 46/255., 89/255.),
                (221/255., 49/255., 78/255.),
                (223/255., 53/255., 66/255.),
                (224/255., 56/255., 54/255.),
                (226/255., 60/255., 42/255.),
                (228/255., 64/255., 30/255.),
                (229/255., 68/255., 25/255.),
                (231/255., 72/255., 20/255.),
                (232/255., 76/255., 16/255.),
                (234/255., 78/255., 12/255.),
                (235/255., 82/255., 10/255.),
                (236/255., 86/255., 8/255.),
                (237/255., 90/255., 7/255.),
                (238/255., 93/255., 5/255.),
                (239/255., 96/255., 4/255.),
                (240/255., 100/255., 3/255.),
                (241/255., 103/255., 3/255.),
                (241/255., 106/255., 2/255.),
                (242/255., 109/255., 1/255.),
                (243/255., 113/255., 1/255.),
                (244/255., 116/255., 0/255.),
                (244/255., 120/255., 0/255.),
                (245/255., 125/255., 0/255.),
                (246/255., 129/255., 0/255.),
                (247/255., 133/255., 0/255.),
                (248/255., 136/255., 0/255.),
                (248/255., 139/255., 0/255.),
                (249/255., 142/255., 0/255.),
                (249/255., 145/255., 0/255.),
                (250/255., 149/255., 0/255.),
                (251/255., 154/255., 0/255.),
                (252/255., 159/255., 0/255.),
                (253/255., 163/255., 0/255.),
                (253/255., 168/255., 0/255.),
                (253/255., 172/255., 0/255.),
                (254/255., 176/255., 0/255.),
                (254/255., 179/255., 0/255.),
                (254/255., 184/255., 0/255.),
                (254/255., 187/255., 0/255.),
                (254/255., 191/255., 0/255.),
                (254/255., 195/255., 0/255.),
                (254/255., 199/255., 0/255.),
                (254/255., 202/255., 1/255.),
                (254/255., 205/255., 2/255.),
                (254/255., 208/255., 5/255.),
                (254/255., 212/255., 9/255.),
                (254/255., 216/255., 12/255.),
                (255/255., 219/255., 15/255.),
                (255/255., 221/255., 23/255.),
                (255/255., 224/255., 32/255.),
                (255/255., 227/255., 39/255.),
                (255/255., 229/255., 50/255.),
                (255/255., 232/255., 63/255.),
                (255/255., 235/255., 75/255.),
                (255/255., 238/255., 88/255.),
                (255/255., 239/255., 102/255.),
                (255/255., 241/255., 116/255.),
                (255/255., 242/255., 134/255.),
                (255/255., 244/255., 149/255.),
                (255/255., 245/255., 164/255.),
                (255/255., 247/255., 179/255.),
                (255/255., 248/255., 192/255.),
                (255/255., 249/255., 203/255.),
                (255/255., 251/255., 216/255.),
                (255/255., 253/255., 228/255.),
                (255/255., 254/255., 239/255.),
                (255/255., 255/255., 249/255.)]

    seqLen = len(col_seq)
    delta = 1.0/(seqLen - 1)
    r_tuple = ((i*delta, col_seq[i][0], col_seq[i][0]) for i in range(seqLen))
    g_tuple = ((i*delta, col_seq[i][1], col_seq[i][1]) for i in range(seqLen))
    b_tuple = ((i*delta, col_seq[i][2], col_seq[i][2]) for i in range(seqLen))
    cdict = {'red': tuple(r_tuple),
             'green': tuple(g_tuple),
             'blue': tuple(b_tuple)}
    cwm = _mpl.colors.LinearSegmentedColormap('Iron', cdict, N)
    return cwm

def _get_fire(N):
    """Helper function for generating Fire colormap.

    Parameters
    ----------
    N : integer

    Returns
    -------
    colormap : colormap
      matplotlib colormap

    References
    ----------
    # Fire colormap, ImageJ
    """
    col_seq = [ (  0/255.,   0/255.,   0/255.),
                (  0/255.,   0/255.,   7/255.),
                (  0/255.,   0/255.,  15/255.),
                (  0/255.,   0/255.,  22/255.),
                (  0/255.,   0/255.,  30/255.),
                (  0/255.,   0/255.,  38/255.),
                (  0/255.,   0/255.,  45/255.),
                (  0/255.,   0/255.,  53/255.),
                (  0/255.,   0/255.,  61/255.),
                (  0/255.,   0/255.,  65/255.),
                (  0/255.,   0/255.,  69/255.),
                (  0/255.,   0/255.,  74/255.),
                (  0/255.,   0/255.,  78/255.),
                (  0/255.,   0/255.,  82/255.),
                (  0/255.,   0/255.,  87/255.),
                (  0/255.,   0/255.,  91/255.),
                (  1/255.,   0/255.,  96/255.),
                (  4/255.,   0/255., 100/255.),
                (  7/255.,   0/255., 104/255.),
                ( 10/255.,   0/255., 108/255.),
                ( 13/255.,   0/255., 113/255.),
                ( 16/255.,   0/255., 117/255.),
                ( 19/255.,   0/255., 121/255.),
                ( 22/255.,   0/255., 125/255.),
                ( 25/255.,   0/255., 130/255.),
                ( 28/255.,   0/255., 134/255.),
                ( 31/255.,   0/255., 138/255.),
                ( 34/255.,   0/255., 143/255.),
                ( 37/255.,   0/255., 147/255.),
                ( 40/255.,   0/255., 151/255.),
                ( 43/255.,   0/255., 156/255.),
                ( 46/255.,   0/255., 160/255.),
                ( 49/255.,   0/255., 165/255.),
                ( 52/255.,   0/255., 168/255.),
                ( 55/255.,   0/255., 171/255.),
                ( 58/255.,   0/255., 175/255.),
                ( 61/255.,   0/255., 178/255.),
                ( 64/255.,   0/255., 181/255.),
                ( 67/255.,   0/255., 185/255.),
                ( 70/255.,   0/255., 188/255.),
                ( 73/255.,   0/255., 192/255.),
                ( 76/255.,   0/255., 195/255.),
                ( 79/255.,   0/255., 199/255.),
                ( 82/255.,   0/255., 202/255.),
                ( 85/255.,   0/255., 206/255.),
                ( 88/255.,   0/255., 209/255.),
                ( 91/255.,   0/255., 213/255.),
                ( 94/255.,   0/255., 216/255.),
                ( 98/255.,   0/255., 220/255.),
                (101/255.,   0/255., 220/255.),
                (104/255.,   0/255., 221/255.),
                (107/255.,   0/255., 222/255.),
                (110/255.,   0/255., 223/255.),
                (113/255.,   0/255., 224/255.),
                (116/255.,   0/255., 225/255.),
                (119/255.,   0/255., 226/255.),
                (122/255.,   0/255., 227/255.),
                (125/255.,   0/255., 224/255.),
                (128/255.,   0/255., 222/255.),
                (131/255.,   0/255., 220/255.),
                (134/255.,   0/255., 218/255.),
                (137/255.,   0/255., 216/255.),
                (140/255.,   0/255., 214/255.),
                (143/255.,   0/255., 212/255.),
                (146/255.,   0/255., 210/255.),
                (148/255.,   0/255., 206/255.),
                (150/255.,   0/255., 202/255.),
                (152/255.,   0/255., 199/255.),
                (154/255.,   0/255., 195/255.),
                (156/255.,   0/255., 191/255.),
                (158/255.,   0/255., 188/255.),
                (160/255.,   0/255., 184/255.),
                (162/255.,   0/255., 181/255.),
                (163/255.,   0/255., 177/255.),
                (164/255.,   0/255., 173/255.),
                (166/255.,   0/255., 169/255.),
                (167/255.,   0/255., 166/255.),
                (168/255.,   0/255., 162/255.),
                (170/255.,   0/255., 158/255.),
                (171/255.,   0/255., 154/255.),
                (173/255.,   0/255., 151/255.),
                (174/255.,   0/255., 147/255.),
                (175/255.,   0/255., 143/255.),
                (177/255.,   0/255., 140/255.),
                (178/255.,   0/255., 136/255.),
                (179/255.,   0/255., 132/255.),
                (181/255.,   0/255., 129/255.),
                (182/255.,   0/255., 125/255.),
                (184/255.,   0/255., 122/255.),
                (185/255.,   0/255., 118/255.),
                (186/255.,   0/255., 114/255.),
                (188/255.,   0/255., 111/255.),
                (189/255.,   0/255., 107/255.),
                (190/255.,   0/255., 103/255.),
                (192/255.,   0/255., 100/255.),
                (193/255.,   0/255.,  96/255.),
                (195/255.,   0/255.,  93/255.),
                (196/255.,   1/255.,  89/255.),
                (198/255.,   3/255.,  85/255.),
                (199/255.,   5/255.,  82/255.),
                (201/255.,   7/255.,  78/255.),
                (202/255.,   8/255.,  74/255.),
                (204/255.,  10/255.,  71/255.),
                (205/255.,  12/255.,  67/255.),
                (207/255.,  14/255.,  64/255.),
                (208/255.,  16/255.,  60/255.),
                (209/255.,  19/255.,  56/255.),
                (210/255.,  21/255.,  53/255.),
                (212/255.,  24/255.,  49/255.),
                (213/255.,  27/255.,  45/255.),
                (214/255.,  29/255.,  42/255.),
                (215/255.,  32/255.,  38/255.),
                (217/255.,  35/255.,  35/255.),
                (218/255.,  37/255.,  31/255.),
                (220/255.,  40/255.,  27/255.),
                (221/255.,  43/255.,  23/255.),
                (223/255.,  46/255.,  20/255.),
                (224/255.,  48/255.,  16/255.),
                (226/255.,  51/255.,  12/255.),
                (227/255.,  54/255.,   8/255.),
                (229/255.,  57/255.,   5/255.),
                (230/255.,  59/255.,   4/255.),
                (231/255.,  62/255.,   3/255.),
                (233/255.,  65/255.,   3/255.),
                (234/255.,  68/255.,   2/255.),
                (235/255.,  70/255.,   1/255.),
                (237/255.,  73/255.,   1/255.),
                (238/255.,  76/255.,   0/255.),
                (240/255.,  79/255.,   0/255.),
                (241/255.,  81/255.,   0/255.),
                (243/255.,  84/255.,   0/255.),
                (244/255.,  87/255.,   0/255.),
                (246/255.,  90/255.,   0/255.),
                (247/255.,  92/255.,   0/255.),
                (249/255.,  95/255.,   0/255.),
                (250/255.,  98/255.,   0/255.),
                (252/255., 101/255.,   0/255.),
                (252/255., 103/255.,   0/255.),
                (252/255., 105/255.,   0/255.),
                (253/255., 107/255.,   0/255.),
                (253/255., 109/255.,   0/255.),
                (253/255., 111/255.,   0/255.),
                (254/255., 113/255.,   0/255.),
                (254/255., 115/255.,   0/255.),
                (255/255., 117/255.,   0/255.),
                (255/255., 119/255.,   0/255.),
                (255/255., 121/255.,   0/255.),
                (255/255., 123/255.,   0/255.),
                (255/255., 125/255.,   0/255.),
                (255/255., 127/255.,   0/255.),
                (255/255., 129/255.,   0/255.),
                (255/255., 131/255.,   0/255.),
                (255/255., 133/255.,   0/255.),
                (255/255., 134/255.,   0/255.),
                (255/255., 136/255.,   0/255.),
                (255/255., 138/255.,   0/255.),
                (255/255., 140/255.,   0/255.),
                (255/255., 141/255.,   0/255.),
                (255/255., 143/255.,   0/255.),
                (255/255., 145/255.,   0/255.),
                (255/255., 147/255.,   0/255.),
                (255/255., 148/255.,   0/255.),
                (255/255., 150/255.,   0/255.),
                (255/255., 152/255.,   0/255.),
                (255/255., 154/255.,   0/255.),
                (255/255., 155/255.,   0/255.),
                (255/255., 157/255.,   0/255.),
                (255/255., 159/255.,   0/255.),
                (255/255., 161/255.,   0/255.),
                (255/255., 162/255.,   0/255.),
                (255/255., 164/255.,   0/255.),
                (255/255., 166/255.,   0/255.),
                (255/255., 168/255.,   0/255.),
                (255/255., 169/255.,   0/255.),
                (255/255., 171/255.,   0/255.),
                (255/255., 173/255.,   0/255.),
                (255/255., 175/255.,   0/255.),
                (255/255., 176/255.,   0/255.),
                (255/255., 178/255.,   0/255.),
                (255/255., 180/255.,   0/255.),
                (255/255., 182/255.,   0/255.),
                (255/255., 184/255.,   0/255.),
                (255/255., 186/255.,   0/255.),
                (255/255., 188/255.,   0/255.),
                (255/255., 190/255.,   0/255.),
                (255/255., 191/255.,   0/255.),
                (255/255., 193/255.,   0/255.),
                (255/255., 195/255.,   0/255.),
                (255/255., 197/255.,   0/255.),
                (255/255., 199/255.,   0/255.),
                (255/255., 201/255.,   0/255.),
                (255/255., 203/255.,   0/255.),
                (255/255., 205/255.,   0/255.),
                (255/255., 206/255.,   0/255.),
                (255/255., 208/255.,   0/255.),
                (255/255., 210/255.,   0/255.),
                (255/255., 212/255.,   0/255.),
                (255/255., 213/255.,   0/255.),
                (255/255., 215/255.,   0/255.),
                (255/255., 217/255.,   0/255.),
                (255/255., 219/255.,   0/255.),
                (255/255., 220/255.,   0/255.),
                (255/255., 222/255.,   0/255.),
                (255/255., 224/255.,   0/255.),
                (255/255., 226/255.,   0/255.),
                (255/255., 228/255.,   0/255.),
                (255/255., 230/255.,   0/255.),
                (255/255., 232/255.,   0/255.),
                (255/255., 234/255.,   0/255.),
                (255/255., 235/255.,   4/255.),
                (255/255., 237/255.,   8/255.),
                (255/255., 239/255.,  13/255.),
                (255/255., 241/255.,  17/255.),
                (255/255., 242/255.,  21/255.),
                (255/255., 244/255.,  26/255.),
                (255/255., 246/255.,  30/255.),
                (255/255., 248/255.,  35/255.),
                (255/255., 248/255.,  42/255.),
                (255/255., 249/255.,  50/255.),
                (255/255., 250/255.,  58/255.),
                (255/255., 251/255.,  66/255.),
                (255/255., 252/255.,  74/255.),
                (255/255., 253/255.,  82/255.),
                (255/255., 254/255.,  90/255.),
                (255/255., 255/255.,  98/255.),
                (255/255., 255/255., 105/255.),
                (255/255., 255/255., 113/255.),
                (255/255., 255/255., 121/255.),
                (255/255., 255/255., 129/255.),
                (255/255., 255/255., 136/255.),
                (255/255., 255/255., 144/255.),
                (255/255., 255/255., 152/255.),
                (255/255., 255/255., 160/255.),
                (255/255., 255/255., 167/255.),
                (255/255., 255/255., 175/255.),
                (255/255., 255/255., 183/255.),
                (255/255., 255/255., 191/255.),
                (255/255., 255/255., 199/255.),
                (255/255., 255/255., 207/255.),
                (255/255., 255/255., 215/255.),
                (255/255., 255/255., 223/255.),
                (255/255., 255/255., 227/255.),
                (255/255., 255/255., 231/255.),
                (255/255., 255/255., 235/255.),
                (255/255., 255/255., 239/255.),
                (255/255., 255/255., 243/255.),
                (255/255., 255/255., 245/255.),
                (255/255., 255/255., 246/255.),
                (255/255., 255/255., 247/255.),
                (255/255., 255/255., 251/255.),
                (255/255., 255/255., 255/255.),
                (255/255., 255/255., 255/255.),
                (255/255., 255/255., 255/255.),
                (255/255., 255/255., 255/255.),
                (255/255., 255/255., 255/255.),
                (255/255., 255/255., 255/255.) ]

    seqLen = len(col_seq)
    delta = 1.0/(seqLen - 1)
    r_tuple = ((i*delta, col_seq[i][0], col_seq[i][0]) for i in range(seqLen))
    g_tuple = ((i*delta, col_seq[i][1], col_seq[i][1]) for i in range(seqLen))
    b_tuple = ((i*delta, col_seq[i][2], col_seq[i][2]) for i in range(seqLen))
    cdict = {'red': tuple(r_tuple),
             'green': tuple(g_tuple),
             'blue': tuple(b_tuple)}
    firecm = _mpl.colors.LinearSegmentedColormap('Fire', cdict, N)
    return firecm

def _hue_scaling(args):
    """return scaled hue values as described in
    http://dlmf.nist.gov/help/vrml/aboutcolor

    args : ndarray of args / angle of complex numbers between in the open
           interval [0, 2*pi)
    q : scaled values returned in the interval [0, 1)
    """
    q = 4.0*_np.mod((args/(2*_np.pi) + 1), 1)
    mask1 = (q >= 0) * (q < 1)
    mask2 = (q >= 1) * (q < 2)
    mask3 = (q >= 2) * (q < 3)
    mask4 = (q >= 3) * (q < 4)
    q[mask1] = (60.0/360)*q[mask1]
    q[mask2] = (60.0/360)*(2.0*q[mask2] - 1)
    q[mask3] = (60.0/360)*(q[mask3] + 1)
    q[mask4] = (60.0/360)*2.0*(q[mask4] - 1)
    return q

def _get_complex_function_hsv(N, sat=0.9, linearPhaseMap=True):
    """Helper function for generating HSV based colormap (cplothsv) for
    rendering complex functions using domain mapping technique in matplotlib,
    especially by `mplutils.plot_complex_function()`.

    Parameters
    ----------
    N : integer
        length of the colormap
    sat : float
        saturation

    Returns
    -------
    colormap : colormap
      matplotlib colormap

    References
    ----------
    # `mplutils._hue()`
    """
    hLinear = _np.linspace(0, 2*_np.pi, num=N, endpoint=False)
    if linearPhaseMap:
        hScaled = hLinear/(2*_np.pi)
    else:
        hScaled = _hue_scaling(hLinear)
    s = sat*_np.ones_like(hScaled)
    v = _np.ones_like(hScaled)
    hsv = _np.dstack((hScaled, s, v))
    rgb = _mplc.hsv_to_rgb(hsv)
    # creation of the tuple structures for red, green, blue
    indexInColMap = _np.linspace(0, 1.0, num=N).reshape(N, 1)
    r = rgb[:,:,0].reshape(N, 1)
    g = rgb[:,:,1].reshape(N, 1)
    b = rgb[:,:,2].reshape(N, 1)
    rTupleStructure = _np.hstack((indexInColMap, r, r))
    gTupleStructure = _np.hstack((indexInColMap, g, g))
    bTupleStructure = _np.hstack((indexInColMap, b, b))
    cdict = {'red': tuple(tuple(elem) for elem in rTupleStructure),
             'green': tuple(tuple(elem) for elem in gTupleStructure),
             'blue': tuple(tuple(elem) for elem in bTupleStructure)}
    cplotHSVcm = _mpl.colors.LinearSegmentedColormap('cplothsv', cdict, N)
    return cplotHSVcm

# ***********************
# Test functions
# ***********************
def _test_get_colormap_description():
    desc = get_colormap_description()
    print(desc)

def _test_trace_colormap():
    """Visual test of the function trace_colormap()"""
    # Note that the figures will open in series after the preciding figure window
    # has been closed
    # Trace cm.hsv colormap (it contains 256 levels)
    trace_colormap(_mpl.cm.hsv)
    # Trace customized colormaps
    morelandCmap = get_colormap('moreland', N=256)
    trace_colormap(morelandCmap)
    ironCmap = get_colormap('iron', N=512)
    trace_colormap(ironCmap, show_vals=True)
    cplotHSVcm = get_colormap('cplothsv', N=256, sat=1.0)
    trace_colormap(cplotHSVcm, infoInTitle='sat=1.0, linear mapping of hsv')
    cplotHSVcm = get_colormap('cplothsv', N=256, sat=1.0, linearPhaseMap=False)
    trace_colormap(cplotHSVcm, infoInTitle='sat=1.0, scaled mapping of hsv')
    cplotHSVcm = get_colormap('cplothsv', N=256, sat=0.7)
    trace_colormap(cplotHSVcm, infoInTitle='sat=0.7, linear mapping of hsv')
    cplotHSVcm = get_colormap('cplothsv', N=256, sat=0.7, linearPhaseMap=False)
    trace_colormap(cplotHSVcm, infoInTitle='sat=0.7, scaled mapping of hsv')



if __name__ == '__main__':
    # Visual tests: These testing methods are meant to be manual tests which requires visual inspection.
    _test_get_colormap_description()
    _test_trace_colormap()