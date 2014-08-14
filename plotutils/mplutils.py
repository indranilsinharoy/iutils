# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          mplutils.py
# Purpose:       matplotlib Plotting utilities
#
# Author:        Indranil Sinharoy
#
# Created:       01/24/2013
# Last Modified: 08/14/2014
# Copyright:     (c) Indranil Sinharoy 2013
# Licence:       MIT License
#-------------------------------------------------------------------------------
from __future__ import division
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


class arrow(object):
    def __init__(self, start, end, a_col=(0.0,0.0,0.0), cone_scale=1.0,
                 fig_object=None, alpha=1.0):
        """Arrow class to draw arrow

        Parameters
        ----------
        start : 2-tuple
            2-d vector representing starting point of the arrow
        end : 2-tuple
            2-d vector representing end point of the arrow

        Returns
        ------- 
        None 
        """
        if fig_object != None:
            self.fig = None
        else:
            self.fig = fig_object
        arr_head2length_ratio = 0.1
        dx = (end[0]-start[0])
        dy = (end[1]-start[1])
        arr_length = np.sqrt(dx**2.0 + dy**2.0)

        alpha = alpha
        width = 0.01
        head_width = 0.15
        head_length = arr_head2length_ratio*arr_length

        self.twoDarrow = plt.arrow(start[0], start[1], dx, dy, color=a_col,
                                   alpha=alpha, width=width, head_width=head_width,
                                   head_length=head_length, length_includes_head=True)

def find_zero_crossings(f, a, b, func_args=(), n=100):
    """Retun a list of zero-crossings (roots) of the function within the
    interval (a,b)

    ``Usage: find_zero_crossings(f, a, b [,func_args, n]) -> zero_crossings``

    Parameters
    ----------
    f : object
        function-name whose zero-crossings (roots) are to be found
    a : float or int
        start of the interval
    b : float or int
        end of the interval
    func_args : tuple, optional
        a tuple of arguments that are to be passed to the function
        ``f`` in the expected order.
    n : integer, optional
        number of points on the real line where the function is evaluated
        in the process of finding the sign-changing intervals that are
        passed to the ``scipy.optimize.brentq`` function (Default==100).

    Returns
    -------
    zero_crossings : list
        zero crossings. If no zero-crossings are found, the returned list
        is empty.

    Examples
    --------
    (1) Zero crossings of a function that takes no arguments

    >>> mpu.find_zero_crossings(np.cos -2*np.pi, 2*np.pi)
    [-4.71238898038469, -1.5707963267948966, 1.5707963267948963, 4.71238898038469]

    (2) Zero crossing of a function that takes one argument

    >>> def func(x, a):
    >>>     return integrate.quad(lambda t: special.j1(t)/t, 0, x)[0] - a
    >>> mpu.find_zero_crossings(func_t2, 1e-10, 25, func_args=(1,))
    [2.65748482456961, 5.672547403169345, 8.759901449672629, 11.87224239501442, 14.99576753285061, 18.12516624215325, 21.258002755273516, 24.393014762783487]
    """
    # Evaluate the function at `n` points on the real line within the interval [a,b]
    real_line = np.linspace(a, b, n)
    fun_vals = [f(x, *func_args) for x in real_line]
    sign_change_arr = [a]   # initialize the first element
    for i in range(1, len(fun_vals)):
        if(fun_vals[i-1]*fun_vals[i] < 0):
            sign_change_arr.append(real_line[i])
    zero_crossings = []     # initialize empty list
    for j in range(1,len(sign_change_arr)):
        zero_crossings.append(optimize.brentq(f, sign_change_arr[j-1],
                              sign_change_arr[j], args=func_args))
    return zero_crossings


def set_spines(axes=None, remove=None, stype=None, soffset=None, zorder=3, 
               setinvisible=False):
    """Sets the spines of a matplotlib figure 

    Parameters
    ----------
    axes : list of axes
        list of axes for transforming their spines
    remove : list of strings
        a list of spines to remove, such as ['left', 'top'] to remove
        left and top spines.
    stype : string
        indicating type of spine as per the following options:
        * 'center_data' = create 2-axis spines at data center 
        * 'center_axes' = center 2-axis spines at axes center
    soffset : list of floats  
        list of spine offsets ``(left, right, top, bottom)``. The offset 
        is with respect to ``stype``. If ``stype`` is not provided, then 
        the offset is with respect to the axes. The offset values can 
        be negative or positive. If ``stype`` is ``center_axes`` or 
        ``None``, the offset values range is between -1.0 and 1.0 
    zorder : integer (between 1 and 20)
        set the zorder of the spines (Default = 3)
    setinvisible : bool
        if "True", set the visibility of the spines to "invisible". Then
        the spines will be "completely removed". This is useful if you
        are trying to print some target image with exact DPI values.
    """
    if axes is None:
        axes = [plt.gca(),]
    allSpines = ['left', 'right', 'top', 'bottom']

    # if set invisible
    if setinvisible:
        for ax in axes:
            for sp in allSpines:
                ax.spines[sp].set_visible(False)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
        return

    # Verify Remove list
    if remove:
        for r in remove:
            if r not in allSpines:
                raise ValueError, 'Invalid remove types given'
    else:
        remove = []
    
    # Verify stype
    if stype:
        if stype not in ['center_data', 'center_axes']: # there could be other variations in future
            raise ValueError, 'Invalid stype given'
    
    # Define what to do if stype is either 'center_data' or 'center_axes'
    if stype:
        if stype in ['center_data', 'center_axes']:
            if remove:
                for sp in ['right', 'top']:
                    if sp not in remove:
                        remove.append(sp)
            else:
                remove = ['right', 'top']
    # Set spine colors
    for ax in axes:
        for sp in allSpines:
            if sp in remove:
                ax.spines[sp].set_color('none')
                if sp == 'left':
                    ax.yaxis.set_tick_params(which='both', left='off', labeltop='off')
                    ax.yaxis.set_ticks_position('right')
                if sp == 'right':
                    ax.yaxis.set_tick_params(which='both', right='off', labeltop='off')
                    ax.yaxis.set_ticks_position('left')
                if sp == 'top':
                    ax.xaxis.set_tick_params(which='both', top='off', labeltop='off')
                    ax.xaxis.set_ticks_position('bottom')
                if sp == 'bottom':
                    ax.xaxis.set_tick_params(which='both', bottom='off', labeltop='off')
                    ax.xaxis.set_ticks_position('top')
            else:
                ax.spines[sp].set_color('#808080') # this is the value set in matplotlibrc for 'axes.edgecolor'
            # set zorder
            ax.spines[sp].zorder = zorder
        # Modify spine type
        if stype in ['center_data', 'center_axes']:
            ref = 'data' if stype == 'center_data' else 'axes'
            pos = 0 if ref == 'data' else 0.5
            ax.yaxis.set_ticks_position('left')
            ax.spines['left'].set_position((ref, pos)) 
            ax.xaxis.set_ticks_position('bottom')
            ax.spines['bottom'].set_position((ref, pos))

        # Set spine positions
        if soffset:
            if len(soffset) > 3:
                left, right, top, bottom = soffset
                ax.autoscale(enable=True, axis='both', tight=True) 
            elif len(soffset) > 2:
                [left, right, top], bottom = soffset, 0
                ax.autoscale(enable=True, axis='both', tight=True)
            elif len(soffset) > 1:
                [left, right], top, bottom = soffset, 0, 0
                ax.autoscale(enable=True, axis='x', tight=True)
            else:
                left, right, top, bottom = soffset[0], 0, 0, 0
                ax.autoscale(enable=True, axis='x', tight=True)
            ref = stype.split('_')[-1] if stype else 'axes'
            pos = 0 if ref == 'data' else 0.5
            for spine, offset in zip(allSpines, (left, right, top, bottom)):
                ax.spines[spine].set_position((ref, pos + offset))

def format_stem_plot(mline, stlines, bline, mecol='#222222', mfcol='#555555', 
                     mstyle='o', msize=5, mjoin='None', stcol='#f67088', 
                     slw=1.6, bcol='#BBBBBB', blw=1.1, bstyle='--'):
    """format matplotlib stem plot 

    Parameters
    ---------- 
    mline : markerline object 
        marker line returned by stem() function 
    stlines : stemlines object 
        stem lines returned by stem() function 
    bline : baseline object  
        base line returned by stem() function 
    mecol : string, optional  
        markerline edge color, default = '#222222'
    mfcol : string, optional
        markerline face color, default = '#555555'
    mstyle : string marker type, optional 
        marker pattern , default = 'o' 
    stcol : string, optional 
        stemlines color, default = '#f67088'
    slw : integer, optional 
        stemlines line width, default = 1.6
    bcol : string, optional 
        baseline color, default = '#BBBBBB'

    Returns
    ------- 
    None

    Examples
    -------- 
    >>> fig, ax = plt.subplots(1, 1)
    >>> x = np.linspace(-np.pi, np.pi)
    >>> y = np.sin(x)
    >>> mline, stlines, bline = ax.stem(x, y)
    >>> mpu.format_stem_plot(mline, stlines, bline)
    >>> plt.show()
    """
    # marker settings
    mline.set_marker(mstyle)
    mline.set_markersize(msize)
    mline.set_markeredgecolor(mecol)
    mline.set_markerfacecolor(mfcol)
    mline.set_linestyle(mjoin)
    # stem line settings
    for line in stlines:
        line.set_color(stcol)
        line.set_linewidth(slw)
    # base line settings
    bline.set_linestyle(bstyle)
    bline.set_linewidth(blw)
    bline.set_color(bcol)

# ------------------------------------------------------------------------
#           TESTING FUNCTIONS
# -------------------------------------------------------------------------

def _test_arrow():
    #test arrow with matplotlib figure
    fig = plt.figure("myfigure",facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    plt.grid()
    #Test drawing the arrows in 2D space
    arrow((0,0),(2,2),a_col=(0.0,0.0,0.0))
    arrow((0,0),(-2,2),a_col=(1.0,0.0,0.0))
    arrow((-2,2),(2,2),a_col=(0.0,0.0,1.0))
    a4 = arrow((0,0),(-3,-3),a_col=(0.0,1.0,1.0))
    a4.twoDarrow.set_linestyle('dashed')
    #passing numpy array vectors
    ori = np.array((0,0))
    v1 = np.array((3,-3))
    arrow(ori,v1,'c')
    plt.show()

def _test_find_zero_crossings():
    """test find_zero_crossings function"""
    print("\nTest for find_zero_crossings function")
    # Zero crossing test for function with no arguments
    def func_t1(x):
        """Computes Integrate [j1(t)/t, {t, 0, x}] - 1"""
        return integrate.quad(lambda t: special.j1(t)/t, 0, x)[0] - 1
    zero_cross = find_zero_crossings(func_t1, 1e-10, 25)
    exp_zc = [2.65748482456961, 5.672547403169345, 8.759901449672629, 11.87224239501442,
              14.99576753285061, 18.12516624215325, 21.258002755273516, 24.393014762783487]
    nt.assert_array_almost_equal(np.array(zero_cross), np.array(exp_zc), decimal=5)
    print("... find_zero_crossings OK for zero-argument function")
    # test for function with one argument
    def func_t2(x, a):
        """Computes Integrate [j1(t)/t, {t, 0, x}] - a"""
        return integrate.quad(lambda t: special.j1(t)/t, 0, x)[0] - a
    zero_cross = find_zero_crossings(func_t2, 1e-10, 25, func_args=(1,))
    nt.assert_array_almost_equal(np.array(zero_cross), np.array(exp_zc), decimal=5)
    print("... find_zero_crossings OK for one-argument function")
    # test for function with no arguments but no zero crossings
    def func_t3(x):
        return x**2.0 + 1.0
    zero_cross = find_zero_crossings(func_t3, 0, 25)
    nt.assert_equal(len(zero_cross),0)
    print("... find_zero_crossings OK for empty return list")
    print("All test for _test_find_zero_crossings() passed successfully")

if __name__ == '__main__':
    import numpy.testing as nt
    from numpy import set_printoptions
    from scipy import integrate, special
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    # Automatic tests
    _test_find_zero_crossings()
    # Visual tests: These testing methods are meant to be manual tests which requires visual inspection.
    _test_arrow()