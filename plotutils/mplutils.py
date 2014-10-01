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
from __future__ import division, print_function
import numpy as _np
from scipy import optimize as _optimize
import matplotlib.pyplot as _plt
import matplotlib.cm as _cm
from matplotlib.widgets import  RectangleSelector as _RectangleSelector

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
        arr_length = _np.sqrt(dx**2.0 + dy**2.0)

        alpha = alpha
        width = 0.01
        head_width = 0.15
        head_length = arr_head2length_ratio*arr_length

        self.twoDarrow = _plt.arrow(start[0], start[1], dx, dy, color=a_col,
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
    [-4.712388980, -1.57079632679, 1.57079632679, 4.712388980]

    (2) Zero crossing of a function that takes one argument

    >>> def func(x, a):
    >>>     return integrate.quad(lambda t: special.j1(t)/t, 0, x)[0] - a
    >>> mpu.find_zero_crossings(func_t2, 1e-10, 25, func_args=(1,))
    [2.65748, 5.67254, 8.75990, 11.87224, 14.99576, 18.12516, 21.25800, 24.39301]
    """
    # Evaluate the function at `n` points on the real line within the interval [a,b]
    real_line = _np.linspace(a, b, n)
    fun_vals = [f(x, *func_args) for x in real_line]
    sign_change_arr = [a]   # initialize the first element
    for i in range(1, len(fun_vals)):
        if(fun_vals[i-1]*fun_vals[i] < 0):
            sign_change_arr.append(real_line[i])
    zero_crossings = []     # initialize empty list
    for j in range(1,len(sign_change_arr)):
        zero_crossings.append(_optimize.brentq(f, sign_change_arr[j-1],
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
        left and top spines. Use ['all'] to remove all spines.
    stype : string
        indicating type of spine as per the following options:
        * 'center_data' = create 2-axes spines at data center
        * 'center_axes' = center 2-axes spines at axes center
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
        axes = [_plt.gca(),]
    elif not isinstance(axes, list):
        axes = [axes, ]
    allSpines = ['left', 'right', 'top', 'bottom']

    # if set invisible
    if setinvisible:
        for ax in axes:
            for sp in allSpines:
                ax.spines[sp].set_visible(False)
            ax.set_frame_on(False)
            ax.patch.set_visible(False)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
        return

    # Verify Remove list
    if remove:
        for r in remove:
            if r not in allSpines + ['all']:
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
        if 'all' in remove:
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            for sp in allSpines:
                ax.spines[sp].set_color('none')

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


def set_ticks(ax, numTicksX=5, numTicksY=5, xlim=None, ylim=None, which='major'):
    """convenience function to quickly set the ticks of a matplotlib plot

    Parameters
    ----------
    ax : axes object
        the axes object
    numTicksX : integer, optional
        number of ticks along the x-axis
    numTicksY : integer, optional
        number of ticks along the y-axis
    xlim : tuple, optional
        x-axis limits
    ylim : tuple, optional
        y-axis limits
    which : string, optional
        which ticks -- 'major' or 'minor'

    Returns
    -------
    None
    """
    xlim = ax.get_xlim() if xlim is None else xlim
    ylim = ax.get_ylim() if ylim is None else ylim

    def get_tick_locator_function(ax, axtype, which):
        """returns the appropriate tick locator function

        @ax : axis
        @axtype : 'x' or 'y'
        @which : 'major' or 'minor'
        """
        axType = 'yaxis' if axtype == 'y' else 'xaxis'
        locFunc = 'set_minor_locator' if which == 'minor' else 'set_major_locator'
        a = ax.__getattribute__(axType)
        aTLocFunc = a.__getattribute__(locFunc)
        return aTLocFunc

    #
    xticLocMult = abs(xlim[1] - xlim[0])/(numTicksX - 1)
    yticLocMult = abs(ylim[1] - ylim[0])/(numTicksY - 1)
    xticLabels = _plt.MultipleLocator(xticLocMult)
    yticLabels = _plt.MultipleLocator(yticLocMult)

    #
    if which == 'major':
        func = get_tick_locator_function(ax, 'x', 'major')
        func(xticLabels)
        func = get_tick_locator_function(ax, 'y', 'major')
        func(yticLabels)

    if which == 'minor':
        func = get_tick_locator_function(ax, 'x', 'minor')
        func(xticLabels)
        func = get_tick_locator_function(ax, 'y', 'minor')
        func(yticLabels)


def format_stem_plot(mline, stlines, bline, mecol='#222222', mfcol='#F52080',
                     mstyle='o', msize=6, mjoin='None', stcol='#0080FF',
                     slw=1.3, bcol='#BBBBBB', blw=1.1, bstyle='--'):
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
        markerline face color, default = '#F52080'
    mstyle : string marker type, optional
        marker pattern , default = 'o'
    msize : integer, optional
        marker size, default = 6
    stcol : string, optional
        stemlines color, default = '#0080FF'
    slw : float, optional
        stemlines line width, default = 1.3
    bcol : string, optional
        baseline color, default = '#BBBBBB'
    blw : float, optional
        baseline line width, default = 1.1
    bstyle : string, optional
        baseline style, defualt = '--'

    Returns
    -------
    None

    Examples
    --------
    >>> fig, ax = _plt.subplots(1, 1)
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

def imshow(image, bGray=False, fig=None, axes=None, subplot=None, interpol=None,
           xlabel=None, ylabel=None):
    """Rudimentary image display routine, for quick display of images without
    the axes
    """
    if (subplot == None):
        subplot = int(111)
    if(fig==None):
        fig = _plt.figure()
        axes = fig.add_subplot(subplot)
    elif(axes==None):
        axes = fig.add_subplot(subplot)
    if(bGray==True):
        _plt.gray()
        # print '\ngray'
    # plot the image
    imPtHandle = _plt.imshow(image, _cm.gray, interpolation=interpol)
    # get the image height and width to set the axis limits
    try:
        pix_height, pix_width = image.shape
    except:
        pix_height, pix_width, _ = image.shape
    # Set the xlim and ylim to constrain the plot
    axes.set_xlim(0, pix_width-1)
    axes.set_ylim(pix_height-1, 0)
    # Set the xlabel and ylable if provided
    if(xlabel != None):
        axes.set_xlabel(xlabel)
    if(ylabel != None):
        axes.set_ylabel(ylabel)
    # Make the ticks to empty list
    axes.xaxis.set_ticks([])
    axes.yaxis.set_ticks([])
    return imPtHandle, fig, axes

class ImageComparator(object):
    def __init__(self, numSubPlots, Hlist, fsize=None, dpi=None):
        """Image Comparator class for comparing sections of images plotted
        in adjacent subplots. The image coordinates are related through a
        specified homography

        Parameters
        ----------
        numSubPlots : integer
            number of sub-plots
        Hlist : list
            list of homography matrices that relates the master axis (subplot)
            to the slave subplots. The length of ``Hlist``, i.e. the number
            of H matrices should be 1 less than the total number of subplots.
        fsize : tuple, optional
            (width, height) in inches of the figure
        dpi : tuple, optional
            DPI resolution of the figure
        """
        if len(Hlist) != numSubPlots - 1:
            raise ValueError("The number of H matrices should be 1 less than "
                             "the number of subplots.")
        fig, axes = _plt.subplots(nrows=1, ncols=numSubPlots, figsize=fsize)
        self._fig = fig
        self._axlist = list(axes.flat)
        self._masterAx = self._axlist[0]
        self._Hlist = [_np.identity(3), ]  # dummy H for master to master
        self._Hlist = self._Hlist + Hlist
        # axis limit variables
        self._masterAxNativeY0 = None
        self._masterAxNativeY1 = None
        self._masterAxNativeX0 = None
        self._masterAxNativeX1 = None
        self._masterAxPrevY0 = None
        self._masterAxPrevY1 = None
        self._masterAxPrevX0 = None
        self._masterAxPrevX1 = None
        fig.tight_layout()
        # Rectangle selector and key press event
        self._rs = _RectangleSelector(self._masterAx, self._on_rec_draw, 'box')
        fig.canvas.mpl_connect('key_press_event', self._key_press)

    def imshow(self, image, subPlotNumber, title=None, fontsize=14, **kwargs):
        """method of ``ImageComparator`` to render a particular subplot

        Parameters
        ----------
        image : ndarray
            either a gray-level or a color RGB image to show in the particular
            subplot identified by ``subPlotNumber``
        subPlotNumber : integer
            the integer number representing the subplot STARTING WITH 0!!!
        colmap : matplotlib colormap, optional
            matplotlib-type colormap
        title : string, optional
            title of the subplot
        fontsize : integer, optional
            fontsize of the title

        Returns
        -------
        None

        Notes
        -----
        Just like plotting in Matplotlib, use the ``show()`` method of
        ``ImageComparator`` to display the figure.
        """
        ax = self._axlist[subPlotNumber]
        ax.imshow(image, interpolation='none', **kwargs)
        #if colmap:
        #    imp.set_cmap(colmap)
        # TODO!! How to handle data limits???
        ylim, xlim = image.shape[:2]
        ylim = ylim - 1
        xlim = xlim - 1
        ax.set_xlim(0, xlim)
        ax.set_ylim(ylim, 0)
        if title:
            ax.set_title(title, fontsize=fontsize)
        if ax == self._masterAx:
            self._masterAxNativeY0 = 0
            self._masterAxNativeY1 = ylim
            self._masterAxNativeX0 = 0
            self._masterAxNativeX1 = xlim
            self._masterAxPrevY0 = 0
            self._masterAxPrevY1 = ylim
            self._masterAxPrevX0 = 0
            self._masterAxPrevX1 = xlim
        _plt.draw()

    def _on_rec_draw(self, pos0, pos1):
        """internal callback function linked to RectangleSelector, called
        when the user selects a rectangular region in the image using the
        mouse
        """
        x0, x1 = pos0.xdata, pos1.xdata
        y0, y1 = pos0.ydata, pos1.ydata
        if (y1 > y0) and (x1 > x0):
            self._redraw_axes(x0, x1, y0, y1)

    def _redraw_axes(self, x0, x1, y0, y1):
        """internal function to re-draw the axes to only show the selected
        region of the image in the master subplot

        x0, x1, y0, y1 are the coordinates in the master axis/subplot
        """
        # redraw the axes of the master axis
        ax = self._masterAx
        self._masterAxPrevX0, self._masterAxPrevX1 = ax.get_xlim()
        self._masterAxPrevY1, self._masterAxPrevY0 = ax.get_ylim()
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        # propagate change to all other axis
        numSubplots = len(self._axlist)
        for i in range(1, numSubplots):
            axSlv = self._axlist[i]
            H = self._Hlist[i]
            x0t, x1t, y0t, y1t = self._get_xlim_and_ylim(H, x0, x1, y0, y1)
            #x0t, x1t, y0t, y1t = int(x0t), int(x1St), int(y0t), int(y1t)
            print("\nFrom x0, x1, y0, y1 :\n", x0, x1, y0, y1)
            print("\nTo  x0t, x1t, y0t, y1t:\n ", x0t, x1t, y0t, y1t)
            print("\nFrom row and col range:", y1 - y0, x1 - x0)
            print("\nTo row and col range:", y1t - y0t, x1t - x0t)
            axSlv.set_xlim(x0t, x1t)
            axSlv.set_ylim(y1t, y0t)
        _plt.draw()

    def _key_press(self, event):
        """internal call-back function that is called when a key is pressed
        """
        key = event.key
        backKeys = ['b', 'B', 'backspace', 'left']
        homeKeys = ['h', 'H', 'home']
        if key in backKeys:
            self._step_back()
        elif key in homeKeys:
            self._go_home()
        elif key == 'escape':
            _plt.close(self._fig)

    def _step_back(self):
        """internal function called when user selects `backspace`, `b`, `B`, or
        `left` arrow keys
        """
        self._redraw_axes(self._masterAxPrevX0, self._masterAxPrevX1,
                          self._masterAxPrevY0, self._masterAxPrevY1)

    def _go_home(self):
        """internal function called when user selects `h`, `H` or `home`
        keys
        """
        self._redraw_axes(self._masterAxNativeX0, self._masterAxNativeX1,
                          self._masterAxNativeY0, self._masterAxNativeY1)

    @staticmethod
    def _get_xlim_and_ylim(H, x0, x1, y0, y1):
        """internal function used to calculate the new axis limits of the
        subplots based on the homographies and axis limits of the master
        subplot.

        x0, x1, y0, y1 are the coordinates in the master axis/subplot
        """
        fp = _np.array([[x0, x1], [y0, y1], [1, 1]])
        tp = _np.dot(H, fp)
        tp = tp/tp[-1] # normalize
        print("Debugging printing")
        print("\nfp:\n", fp)
        print("\ntp:\n", tp)
        print("\nH:\n", H)
        x0t, x1t, y0t, y1t = tp[0,0], tp[0,1], tp[1,0], tp[1,1]
        return x0t, x1t, y0t, y1t

    @staticmethod
    def show():
        """method to render the figure window"""
        _plt.show()

# ------------------------------------------------------------------------
#           TESTING FUNCTIONS
# -------------------------------------------------------------------------

def _test_arrow():
    #test arrow with matplotlib figure
    fig = _plt.figure("myfigure",facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    _plt.grid()
    #Test drawing the arrows in 2D space
    arrow((0,0),(2,2),a_col=(0.0,0.0,0.0))
    arrow((0,0),(-2,2),a_col=(1.0,0.0,0.0))
    arrow((-2,2),(2,2),a_col=(0.0,0.0,1.0))
    a4 = arrow((0,0),(-3,-3),a_col=(0.0,1.0,1.0))
    a4.twoDarrow.set_linestyle('dashed')
    #passing numpy array vectors
    ori = _np.array((0,0))
    v1 = _np.array((3,-3))
    arrow(ori,v1,'c')
    _plt.show()

def _test_find_zero_crossings():
    """test find_zero_crossings function"""
    print("\nTest for find_zero_crossings function")
    # Zero crossing test for function with no arguments
    def func_t1(x):
        """Computes Integrate [j1(t)/t, {t, 0, x}] - 1"""
        return _integrate.quad(lambda t: _special.j1(t)/t, 0, x)[0] - 1
    zero_cross = find_zero_crossings(func_t1, 1e-10, 25)
    exp_zc = [2.65748482456961, 5.672547403169345, 8.759901449672629, 11.87224239501442,
              14.99576753285061, 18.12516624215325, 21.258002755273516, 24.393014762783487]
    _nt.assert_array_almost_equal(_np.array(zero_cross), _np.array(exp_zc), decimal=5)
    print("... find_zero_crossings OK for zero-argument function")
    # test for function with one argument
    def func_t2(x, a):
        """Computes Integrate [j1(t)/t, {t, 0, x}] - a"""
        return _integrate.quad(lambda t: _special.j1(t)/t, 0, x)[0] - a
    zero_cross = find_zero_crossings(func_t2, 1e-10, 25, func_args=(1,))
    _nt.assert_array_almost_equal(_np.array(zero_cross), _np.array(exp_zc), decimal=5)
    print("... find_zero_crossings OK for one-argument function")
    # test for function with no arguments but no zero crossings
    def func_t3(x):
        return x**2.0 + 1.0
    zero_cross = find_zero_crossings(func_t3, 0, 25)
    _nt.assert_equal(len(zero_cross),0)
    print("... find_zero_crossings OK for empty return list")
    print("All test for _test_find_zero_crossings() passed successfully")

def _test_ImageComparator():
    curFilePath = _os.path.realpath(__file__)
    testDataPath = curFilePath.rsplit('\\', 2)[0]
    imgPath = _os.path.join(testDataPath, 'testdata', 'mandrill.png')

    #H1 = _np.array([[ 9.43613054e-01,  -6.51706101e-02,   1.84603759e+02], # Need to use better homography
    #                [  2.58720236e-03,  6.57586485e-01,   1.00856861e+02],
    #                [ -3.72715493e-08,  -1.91863241e-05,  1.00000000e+00]])
    #H2 = _np.array([[  9.43613054e-01,  -6.51706101e-02,   1.84603759e+02], # need to use better homography
    #                [  2.58720236e-03,  6.57586485e-01,   1.00856861e+02],
    #                [ -3.72715493e-08,  -1.91863241e-05,  1.00000000e+00]])
    H1 =  _np.eye(3)
    H2 = _np.eye(3)
    H1[0,0] = 2
    H1[1,1] = 2
    ic = ImageComparator(numSubPlots=3, Hlist=[H1, H1], fsize=(16, 6))
    im0 = _imread(imgPath, flatten=True)
    im1 = im0.copy()
    im2 = im0.copy()
    ic.imshow(im0, 0, title='Master')
    ic.imshow(im1, 1, title='Slave1')
    ic.imshow(im2, 2, title='Slave2')
    ic.show()

if __name__ == '__main__':
    import numpy.testing as _nt
    import os as _os
    from numpy import set_printoptions
    from scipy import _integrate, _special
    from scipy.misc import _imread
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    # Automatic tests
    #_test_find_zero_crossings()
    # Visual tests: These testing methods are meant to be manual tests which requires visual inspection.
    #_test_arrow()
    _test_ImageComparator()