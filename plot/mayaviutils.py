# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          mayaviutils.py
# Purpose:       Mayavi plotting utilities
#
# Author:        Indranil Sinharoy
#
# Created:       01/24/2013
# Last Modified: 08/01/2014
#                 1. Added implicit_plot()
#                 2. Separated plottingUtils to mayaviutils and mplutils.
# Copyright:     (c) Indranil Sinharoy 2013 - 2017
# Licence:       MIT License
#-------------------------------------------------------------------------------
from __future__ import division
import numpy as np
from mayavi import mlab
from mayavi.modules.scalar_cut_plane import ScalarCutPlane
from mayavi.sources.builtin_surface import BuiltinSurface
from mayavi.modules.surface import Surface


class arrow(object):
    def __init__(self, start, end, a_col=(0.0,0.0,0.0), cone_scale=1.0,
                 fig_object=None, alpha=1.0):
        """Arrow class to draw arrow

        Parameters
        ----------
        start : 3-d vector
            starting point of the arrow.
        end : 3-d vector
            end point of the arrow.
        """

        if fig_object != None:
            self.fig = None
        else:
            self.fig = fig_object

        # plot using Mayavi
        # CALCULATE CONE SIZE/SCALE. The unscaled cone size is "1.0"
        r = (np.array(end)-np.array(start))
        r_len = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
        u = np.array(r[0]/r_len)
        v = np.array(r[1]/r_len)
        w = np.array(r[2]/r_len)

        #CALCULATE POSITION OF THE CONE
        #direction cosines
        alpha = u; beta = v; gamma = w;  #alpha = cos(a), beta = cos(b), gamma = cos(c)
        r_len_dash = 1.0*cone_scale #unscaled length of the cone is "1.0"
        x_dash = (r_len - r_len_dash)*alpha + start[0]
        y_dash = (r_len - r_len_dash)*beta  + start[1]
        z_dash = (r_len - r_len_dash)*gamma + start[2]

        # STICK
        x_stick = np.array([start[0],x_dash])
        y_stick = np.array([start[1],y_dash])
        z_stick = np.array([start[2],z_dash])
        self.cone = mlab.quiver3d(x_dash, y_dash, z_dash, u, v, w, mode='cone',
                    color=a_col,resolution=64,scale_factor=cone_scale)
        self.stick = mlab.plot3d(x_stick,y_stick,z_stick,color=a_col, tube_radius=0.05,name='stick')


def drawOriginAxes(plotExtents, displace=None, colAxes=True, cones=True,
                   xaxis=True, yaxis=True, zaxis=True, opacity=1.0,
                   scale_arrow_width=1.0, scale_label=0.5,
                   label_color=(0.1,0.1,0.1), visible=True, cone_scale_factor=1.0,
                   axis_tube_radius=0.05, axis_mono_col=(0.1,0.1,0.1)):
    """Function to draw crossed axes through the origin. The three axes have the
    option of adding a cone, and also may or may not have different colors.
    The axes are also labeled.


    Examples
    --------
    >>>plotExtents = (-10,10,-10,10,-10,10)
    >>>drawOriginAxes(plotExtents,colAxes=True,opacity=0.90)

    #also, if you need to scale the arrow & bar widths and the text label use:
    
    >>>drawOriginAxes(plotExtents, colAxes=True,scale_arrow_width=0.95, scale_label=0.5, opacity=0.95)

    Notes
    -----
    For simple, crossed axes without different colors, lables, and cones, it is
    better (simplicity) to use:

    >>>x = np.array([0]);y = np.array([0]);z = np.array([0]); # These may be standard Python scalars too
    >>>ss = some value (extent of the axes)
    >>>caxis = mlab.points3d(x, y, z, ss, mode='axes',color=(0,1,0), scale_factor=1)
    >>>caxis.actor.property.lighting = False
    """
    #even if the user doesn't want any axis through the origin, we need to draw
    #something before a Mayavi axes can be attached. So we draw only the z-axis
    #and set it's visibility to False
    if visible==False:
        cones=False          #don't draw any cones
        xaxis=False          #don't draw x-axis
        yaxis=False          #don't draw y-axis
        zaxis=True           #draw the z-axis
    ext2subtract = 0.0
    if cones==True:
        ext2subtract = 1.0*cone_scale_factor
    oa_xlim = np.array([plotExtents[0], plotExtents[1] - ext2subtract])
    oa_ylim = np.array([plotExtents[2], plotExtents[3] - ext2subtract])
    oa_zlim = np.array([plotExtents[4], plotExtents[5] - ext2subtract])

    center = np.array([0,0])
    oa_colork = label_color  # label color

    if colAxes:
        oa_colR = (0.9, 0, 0)
        oa_colG = (0, 0.9, 0)
        oa_colB = (0, 0, 0.9)
    else:
        oa_colR = axis_mono_col 
        oa_colG = axis_mono_col 
        oa_colB = axis_mono_col

    # x-axis
    if xaxis:
        x = np.array([oa_xlim[1]]);y = np.array([0]);z = np.array([0])
        u = np.array([1]); v = np.array([0]); w = np.array([0])
        if cones:
            x_cone = mlab.quiver3d(x, y, z, u, v, w, mode='cone', color=oa_colR, 
                                   scale_factor=cone_scale_factor)
            x_cone.actor.property.lighting = False
            x_cone.actor.property.opacity = opacity
            x_cone.actor.actor.scale = np.array((scale_arrow_width, scale_arrow_width, scale_arrow_width))
        x_axis = mlab.plot3d(oa_xlim, center, center, color=oa_colR, line_width=1.0,
                             tube_radius=axis_tube_radius, name='oax')
        x_axis.actor.actor.scale = np.array((1.0, scale_arrow_width, scale_arrow_width)) # don't scale along the x-axis
        x_axis.actor.property.lighting = False
        x_axis.actor.property.opacity = opacity
        # lately, text3d isn't working
        #xaxis_label = mlab.text3d(0.9*oa_xlim[1], 0, 0, 'x', scale=scale_label, color=oa_colork)
        xaxis_label = mlab.text(x=0.9*oa_xlim[1], y=0.0*oa_ylim[1], 
                                z=0, text='x', width=0.01*scale_label, color=oa_colork)
        
    # y-axis
    if yaxis:
        x = np.array([0]);y = np.array([oa_ylim[1]]);z = np.array([0])
        u = np.array([0]);v = np.array([1]);w = np.array([0])
        if cones:
            y_cone = mlab.quiver3d(x, y, z, u, v, w, mode='cone', color=oa_colG, 
                                   scale_factor=cone_scale_factor)
            y_cone.actor.property.lighting = False
            y_cone.actor.property.opacity = opacity
            y_cone.actor.actor.scale = np.array((scale_arrow_width, scale_arrow_width, scale_arrow_width))
        y_axis = mlab.plot3d(center,oa_ylim,center, color=oa_colG, line_width=1.0,
                             tube_radius=axis_tube_radius, name='oay')
        y_axis.actor.actor.scale = np.array((scale_arrow_width, 1.0, scale_arrow_width))  # don't scale along the y-axis
        y_axis.actor.property.lighting = False
        y_axis.actor.property.opacity = opacity
        #lately, text3d is not working        
        #yaxis_label = mlab.text3d(0,0.9*oa_ylim[1],0,'y',scale=scale_label,color=oa_colork)
        yaxis_label = mlab.text(x=0.015*oa_xlim[1], y=0.9*oa_ylim[1], 
                                z=0, text='y', width=0.01*scale_label, color=oa_colork)
    # z-axis
    if zaxis:
        x = np.array([0]);y = np.array([0]);z = np.array([oa_zlim[1]])
        u = np.array([0]);v = np.array([0]);w = np.array([1])
        if cones:
            z_cone = mlab.quiver3d(x, y, z, u, v, w, mode='cone', color=oa_colB, 
                                   scale_factor=cone_scale_factor)
            z_cone.actor.property.lighting = False
            z_cone.actor.property.opacity = opacity
            z_cone.actor.actor.scale = np.array((scale_arrow_width, scale_arrow_width, scale_arrow_width))
        z_axis = mlab.plot3d(center, center, oa_zlim, color=oa_colB, line_width=1.0,
                             tube_radius=axis_tube_radius, name='oaz')
        z_axis.actor.actor.scale = np.array((scale_arrow_width,scale_arrow_width, 1.0))  # don't scale along the z-axis
        z_axis.actor.property.lighting = False
        z_axis.actor.property.opacity = opacity
        #lately, text3d isn't working         
        #zaxis_label = mlab.text3d(0,0,0.9*oa_zlim[1],'z',scale=scale_label,color=oa_colork)
        zaxis_label = mlab.text(x=0.01*oa_xlim[1], y=0.0*oa_ylim[1], 
                                z=0.9*oa_zlim[1], text='z', width=0.01*scale_label, color=oa_colork)

    if visible==False:
        z_axis.actor.actor.visibility=False
        zaxis_label.actor.visibility=False

def mayaviFig(sceneName="Figure", plotExtents=[-10,10,-10,10,-10,10],
              crossAxes=True,oriAxes=True, colCrossAxes=True, fsize = (400, 350),
              bgcol=(0.97, 0.97, 0.97), fgcol=(0.07,0.07,0.07)):
    """Custom Mayavi figure. It returns the handle to the figure (scene in
    Mayavi pipeline language)

    Defaults:
    plotExtents = [-10,10,-10,10,-10,10]
    fsize = (400, 350)
    bgcol=(0.97, 0.97, 0.97)    #Color of the background
    fgcol=(0.07,0.07,0.07)      #color of all text annotation labels(axes, ori axes, scalar bar labels)
    Please use mlab.show() at the end to pop up the figure

    Also, note:
    You can set the anti-aliasing filter while displaying using the following command.
    However, this slows down the interaction speed quite a bit
    mfig.scene.render_window.aa_frames = 8
    [0=no anti-aliasing, ]
    """
    ##FIGURE
    mfig = mlab.figure(sceneName,bgcolor=bgcol,fgcolor=fgcol,size=fsize)
    # for parallel projection (you can also do it using the cam) do
    # mfig.scene.parallel_projection=True

    drawOriginAxes(plotExtents,colAxes=True)

    ###AXES
    maxes = mlab.axes(color=fgcol,extent=plotExtents,nb_labels=7)

    ##Change axes font size, font color, etc
    maxes.axes.font_factor=1

    ##Chage flymode. Default is 'outer_edges'
    #maxes.axes.fly_mode='none'
    #maxes.axes.fly_mode='closest_triad'
    #maxes.axes.fly_mode='outer_edges'

    ##Change x,y,z labels
    maxes.axes.x_label='x'
    maxes.axes.y_label='y'
    maxes.axes.z_label='z'
    #Change tick label format
    maxes.axes.label_format='%-#6.2g'
    #Opacity of the axes lines
    maxes.axes.property.opacity=1.0
    #Color of the axes lines
    maxes.axes.property.color=(0.7,0.7,0.7)
    ##Axes title/label text
    maxes.title_text_property.font_family='times'
    maxes.title_text_property.bold=False
    maxes.title_text_property.italic=True
    maxes.title_text_property.shadow=True
    ##Axes tick label text
    maxes.label_text_property.italic=False
    maxes.label_text_property.font_family='arial'
    maxes.label_text_property.bold=False

    ###OUTLINE
    #Create an outline for the data.
    outl = mlab.outline(color=(0.0,0.0,0.0),extent=plotExtents,opacity=0.25)

    return mfig


def implicit_plot(expr, ext_grid, fig_handle=None, Nx=101, Ny=101, Nz=101,
                 col_isurf=(50/255, 199/255, 152/255), col_osurf=(240/255,36/255,87/255),
                 opa_val=0.8, opaque=True, ori_axis=True, **kwargs):
    """Function to plot algebraic surfaces described by implicit equations
    in Mayavi

    Implicit functions are functions of the form

        ``F(x,y,z) = c``

    where ``c`` is an arbitrary constant.

    Parameters
    ----------
    expr : string
        the expression ``F(x,y,z) - c``; e.g. to plot a unit sphere, the
        ``expr`` will be ``x**2 + y**2 + z**2 - 1``
    ext_grid : 6-tuple
        tuple denoting the range of `x`, `y` and `z` for grid; it has the
        form - (xmin, xmax, ymin, ymax, zmin, zmax)
    fig_handle : figure handle object, optional
        if a mayavi figure object is passed, then the surface shall be
        added to the scene in the given figure. Then, it is the
        responsibility of the calling function to call ``mlab.show()``.
    Nx, Ny, Nz : integers, optional
        number of points along each axis. It is recommended to use odd
        numbers to ensure the calculation of the function at the origin.
    col_isurf : 3-tuple, optional
        color of inner surface, when double-layered surface is used.
        This is also the specified color for single-layered surface.
    col_osurf : 3-tuple, optional
        color of outer surface
    opa_val : float, optional
        opacity value (alpha) to use for surface
    opaque : boolean, optional
        flag to specify whether the surface should be opaque or not
    ori_axis : boolean
        Flag to specify whether a central axis to draw or not

    Notes
    -----
    1. Implementation note
       For opaque surfaces (i.e. if the argument ``opaque`` is True), a
       double layered surface is drawn in order to render two different
       surface colors for the inner and outer layers respectively. If a
       double-layered surface is not desired, make ``opaque`` = False
       and use a value of 1.0 for ``opa_val`` (the opacity of the surface).

    2. For using this function from within IPython
       To use this function within an IPython notebook the following
       wrapper method may help:

       ::

            def fig_embed_wrapper(func):
                def embed_figure(expr, ext_grid, Nx=101, Ny=101, Nz=101,
                             col_isurf=(1.0,1.0,0.14), col_osurf=(0.87,0.086,0.086),
                             opa_val=0.8, opaque=True, ori_axis=True,
                             fig_bg_col=(0.097, 0.097, 0.097), fig_size=(800, 800), zoom=1.0,
                             embed_fig_size=(6,6)):

                    figw = mlab.figure(1, bgcolor=fig_bg_col, fgcolor=(0, 0, 0), size=fig_size)
                    func(expr, ext_grid, figw, Nx, Ny, Nz, col_isurf, col_osurf,
                         opa_val, opaque, ori_axis)
                    cam = figw.scene.camera
                    cam.elevation(-20)
                    cam.zoom(zoom)
                    arr = mlab.screenshot()
                    mlab.show()
                    # matplotlib figure to embedd within IPython notebook
                    fig_embed = plt.figure(figsize=embed_fig_size)
                    ax = fig_embed.add_subplot(111)
                    ax.imshow(arr)
                    ax.axis('off')
                    plt.show()
                    return
                return embed_figure

    Following the above wrapper method, use the following construct.

        ``implicitplot = fig_embed_wrapper(implicit_plot)``

    References
    ----------
    "ImplicitFunction_and_AlgebraicSurface_Plotting_Using_Mayavi.ipynb"
    """
    if fig_handle==None:  # create a new figure
        fig = mlab.figure(1,bgcolor=(0.97, 0.97, 0.97), fgcolor=(0, 0, 0), size=(800, 800))
    else:
        fig = fig_handle
    xl, xr, yl, yr, zl, zr = ext_grid
    x, y, z = np.mgrid[xl:xr:eval('{}j'.format(Nx)),
                       yl:yr:eval('{}j'.format(Ny)),
                       zl:zr:eval('{}j'.format(Nz))]
    scalars = eval(expr)
    src = mlab.pipeline.scalar_field(x, y, z, scalars)
    if opaque:
        delta = 1.e-5
        opa_val=1.0
    else:
        delta = 0.0
        #col_isurf = col_osurf
    # In order to render different colors to the two sides of the algebraic surface,
    # the function plots two contour3d surfaces at a "distance" of delta from the value
    # of the solution.
    # the second surface (contour3d) is only drawn if the algebraic surface is specified
    # to be opaque.
    # TO DO: Can we be more intelligent here? i.e. render the second surface if and
    # only if the algebraic suface is not closed??
    cont1 = mlab.pipeline.iso_surface(src, color=col_isurf, contours=[0-delta],
                                      transparent=False, opacity=opa_val)
    cont1.compute_normals = False # for some reasons, setting this to true actually cause
                                  # more unevenness on the surface, instead of more smooth
    if opaque: # the outer surface is specular, the inner surface is not
        cont2 = mlab.pipeline.iso_surface(src, color=col_osurf, contours=[0+delta],
                                          transparent=False, opacity=opa_val)
        cont2.compute_normals = False
        cont1.actor.property.backface_culling = True
        cont2.actor.property.frontface_culling = True
        cont2.actor.property.specular = 0.2 #0.4 #0.8
        cont2.actor.property.specular_power = 55.0 #15.0
    else:  # make the surface (the only surface) specular
        cont1.actor.property.specular = 0.2 #0.4 #0.8
        cont1.actor.property.specular_power = 55.0 #15.0

    # Scene lights (4 lights are used)
    engine = mlab.get_engine()
    scene = engine.current_scene
    cam_light_azimuth = [78, -57, 0, 0]
    cam_light_elevation = [8, 8, 40, -60]
    cam_light_intensity = [0.72, 0.48, 0.60, 0.20]
    for i in range(4):
        camlight = scene.scene.light_manager.lights[i]
        camlight.activate = True
        camlight.azimuth = cam_light_azimuth[i]
        camlight.elevation = cam_light_elevation[i]
        camlight.intensity = cam_light_intensity[i]
    # axis through the origin
    if ori_axis:
        len_caxis = int(1.05*np.max(np.abs(np.array(ext_grid))))
        caxis = mlab.points3d(0.0, 0.0, 0.0, len_caxis, mode='axes',
                              color=(0.15,0.15,0.15), line_width=1.0,
                              scale_factor=1.,opacity=1.0)
        caxis.actor.property.lighting = False
    # if no figure is passed, the function will create a figure.
    if fig_handle==None:
        # Setting camera
        cam = fig.scene.camera
        cam.elevation(-20)
        cam.zoom(1.0) # zoom should always be in the end.
        mlab.show()

def drawScalarCutPlane(planeNorm=(0,0,0), planeOri=(0,0,0), filterNorm=None,
    tubing=False, viewControls=True, engine=None, scene=None):
    """Draws scalar cut plane in a Mayavi figure

    Parameters
    ----------
    planeNorm : 3-tuple
        Normal vector to the scalar cut plane
    planeOri : 3-tuple
        The coordinates in world coordinates where the plane should be placed
    filterNorm : 3-tuple
        ?? (default is `None`) if `None` then this is assigned to the `planeNorm`
    tubing : boolean
        Whether or not to put a tube surrounding the plane. Default is `False` i.e. no tubing
    viewControls: boolean
        Enable (if True, which is default behavior) or disable GUI based control of the cut-plane
    engine : Mayavi Engine
        Default = None
    scene : Mayavi scene object
        Default = None

    Returns
    -------
    scp : scalar cut plane object

    See also `drawScalarCutPlaneUsingPipeline`
    """
    if not engine:
        engine = mlab.get_engine()
    if not scene:
        scene = engine.scenes[0]
    scp = ScalarCutPlane()
    engine.add_module(scp)
    scp.implicit_plane.widget.origin = planeOri
    scp.implicit_plane.normal = planeNorm
    if filterNorm:
        scp.warp_scalar.filter.normal = filterNorm
    else:
        scp.warp_scalar.filter.normal =  planeNorm  # I'm currently not sure what it does ... it generally follows the normal
    scp.implicit_plane.widget.enabled = viewControls
    if viewControls:
        scp.implicit_plane.widget.tubing = tubing
    return scp

def drawScalarCutPlaneUsingPipeline(src, planeNorm=(0,0,0), planeOri=(0,0,0), filterNorm=None, tubing=False, viewControls=True):
    """Draws scalar cut plane using the Mayavi Pipeline

    Parameters
    ----------
    src : data source (Mayavi source, or VTK dataset)
        For example, `src` could be scalar field data - `src = mlab.pipeline.scalar_field(x,y,z,s)`
    planeNorm : 3-tuple
        Normal vector to the scalar cut plane
    planeOri : 3-tuple
        The coordinates in world coordinates where the plane should be placed
    filterNorm : 3-tuple
        ?? (default is `None`) if `None` then this is assigned to the `planeNorm`
    tubing : boolean
        Whether or not to put a tube surrounding the plane. Default is `False` i.e. no tubing
    viewControls: boolean
        Enable (if True, which is default behavior) or disable GUI based control of the cut-plane

    Returns
    -------
    scp : scalar cut plane object
    """
    scp = mlab.pipeline.scalar_cut_plane(src)
    scp.implicit_plane.widget.origin = planeOri
    scp.implicit_plane.normal = planeNorm
    if filterNorm:
        scp.warp_scalar.filter.normal = filterNorm
    else:
        scp.warp_scalar.filter.normal =  planeNorm  # I'm currently not sure what it does ... it generally follows the normal
    scp.implicit_plane.widget.enabled = viewControls
    if viewControls:
        scp.implicit_plane.widget.tubing = tubing
    return scp

def mayaviBuiltinPlane(len_x=1.0, len_y=1.0, loc=(0.0,0.0,0.0), normal=(0.0,0.0,1.0),
                       planeCol=(1.0,1.0,1.0), planeOpa=0.8, planeLighting=False,
                       drawNormal=True, normVecScale=1.0, engine=None, scene=None):
    """Function to draw Mayavi builtin plane surface

    The function assumes that a figure has already been drawn and a scene is present.

    Parameters
    ----------
    len_x : float
        side length of the plane along x axis (normally horizontal/width)
    len_y : float
        side length of the plane along y axis (normally vertical/height)
    loc : 3-tuple of floats
        (x,y,z) coordinates of the centroid of the plane in the world-coordinates
    normal : 3-tuple of floats
        (nx, ny, nz) normal vector of the plane surface
    planeCol : 3-tuple of floats (values between 0 and 1)
        Color of the plane
    planeOpa : float (between 0 and 1)
        opacity property of the plane
    planeLighting : boolean
        whether or not to use lighting on the plane (default is False)
    drawNormal : boolean
        whether or not to draw the plane normal
    normVecScale : float
        scale factor of the normal vector if drawn (default=1)
    engine : mayavi engine
        optional
    scene : mayavi scene
        optional

    Note: If the engine and the scene is not passed explicitly, it will grab the current engine and the first scene of that engine.

    Returns
    -------
    if `drawNormal` is `True` : 2-tuple (plane_surface, normal_vector)
    if `drawNormal` is `False` : plane_surface
    """
    if not engine:
        engine = mlab.get_engine()
    if not scene:
        scene = engine.scenes[0]
    # ensure the normal vector is of length 1
    nvl = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
    if nvl != 1.00:
        normal = normal[0]/nvl, normal[1]/nvl, normal[2]/nvl
    plane_surf = BuiltinSurface()
    engine.add_source(plane_surf)
    plane_surf.source = 'plane'
    #plane_surf.data_source.center = np.array(loc)
    origin = loc[0] - len_x/2.0, loc[1] - len_y/2.0, 0.0
    point1 = origin[0] + len_x, origin[1], 0.0
    point2 = origin[0], origin[1] + len_y, 0.0
    plane_surf.data_source.origin = np.array(origin)
    plane_surf.data_source.point1 = np.array(point1)
    plane_surf.data_source.point2 = np.array(point2)
    plane_surf.data_source.center = np.array(loc) # this needs to be set here
    plane_surf.data_source.normal = np.array(normal)
    surface = Surface()
    engine.add_filter(surface, plane_surf)
    # Add color and opacity and lighting properties
    surface.actor.property.color = planeCol
    surface.actor.property.opacity = planeOpa
    surface.actor.property.lighting = planeLighting
    if drawNormal:        # Draw plane normal
        norm_drawn = mlab.quiver3d(loc[0], loc[1], loc[2],
                                 [normal[0]], [normal[1]], [normal[2]],  # required to do this way for matching shape ... else Mayavi will throw error. This may also be related to the issue -- https://github.com/enthought/mayavi/issues/85
                                 mode='arrow', resolution=16, color=(1,0,0),
                                 scale_factor=normVecScale, opacity=1.0)
        return plane_surf, norm_drawn
    else:
        return plane_surf


# ------------------------------------------------------------------------
#           TESTING FUNCTIONS
# -------------------------------------------------------------------------


def _test_drawOriginAxes():
    ##FIGURE
    sceneName="myfigure"
    bgcol=(0.97, 0.97, 0.97)    #Color of the background
    fgcol=(0.2,0.2,0.2)      #color of all text annotation labels(axes, ori axes, scalar bar labels)
    fsize = (600, 650)
    mfig = mlab.figure(sceneName,bgcolor=bgcol,fgcolor=fgcol,size=fsize)
    # for parallel projection (you can also do it using the cam) do
    mfig.scene.parallel_projection=True
    mfig.scene.z_plus_view()
    plotExtents = (-10,10,-10,10,-10,10)

    #
    caxis = mlab.points3d(0, 0, 0, 10, mode='axes', color=(0,0,0), scale_factor=0.5, opacity=0.5)
    caxis.actor.property.lighting = False

    #different scenarios
    drawOriginAxes(plotExtents, scale_arrow_width=1.0, scale_label=0.5, 
                   opacity=0.95, visible=True)
    mlab.show()



def _test_mayaviFig():
    ##FIGURE
    sceneName="mayavifigure"
    bgcol=(0.97, 0.97, 0.97)    #Color of the background
    fgcol=(0.2,0.2,0.2)      #color of all text annotation labels(axes, ori axes, scalar bar labels)
    fsize = (400, 350)
    mfig = mlab.figure(sceneName,bgcolor=bgcol,fgcolor=fgcol,size=fsize)
    # for parallel projection (you can also do it using the cam) do
    # mfig.scene.parallel_projection=True
    plotExtents = [-10,10,-10,10,-10,10]

    #different scenarios
    drawOriginAxes(plotExtents,colAxes=True)
    #drawOriginAxes(plotExtents,colAxes=False)
    #drawOriginAxes(plotExtents,colAxes=True,cones=False)
    #drawOriginAxes(plotExtents,colAxes=True,cones=True,xaxis=False,yaxis=False)
    #drawOriginAxes(plotExtents,visible=False)

    ###AXES
    maxes = mlab.axes(color=fgcol,extent=plotExtents,nb_labels=7)

    ##Change axes font size, font color, etc
    maxes.axes.font_factor=1

    ##Chage flymode. Default is 'outer_edges'
    #maxes.axes.fly_mode='none'
    #maxes.axes.fly_mode='closest_triad'
    #maxes.axes.fly_mode='outer_edges'

    ##Change x,y,z labels
    maxes.axes.x_label='x'
    maxes.axes.y_label='y'
    maxes.axes.z_label='z'
    #Change tick label format
    maxes.axes.label_format='%-#6.2g'
    #Opacity of the axes lines
    maxes.axes.property.opacity=1.0
    #Color of the axes lines
    maxes.axes.property.color=(0.7,0.7,0.7)
    ##Axes title/label text
    maxes.title_text_property.font_family='times'
    maxes.title_text_property.bold=False
    maxes.title_text_property.italic=True
    maxes.title_text_property.shadow=True
    ##Axes tick label text
    maxes.label_text_property.italic=False
    maxes.label_text_property.font_family='arial'
    maxes.label_text_property.bold=False

    ###OUTLINE
    #Create an outline for the data.
    outl = mlab.outline(color=(0.0,0.0,0.0),extent=plotExtents,opacity=0.25)
    #Set the camera
    cam = mfig.scene.camera
    cam.parallel_projection = False
    cam.zoom(1.0)

    #The following camera position: +y is up, +z is towards left, +x is coming out
    #of the screen plane towards the viewer, the camera view center is 0,0,0.
    #Setting a non-zero roll will rotate the camera in the plane perpendicular
    #to the x-axis. azimuth controls the angle in the x-y plane (0-360),
    #elevation controls the angle in the z to x-y plane (0-180)
    #mlab.view(azimuth=0, elevation=90.0, reset_roll=True, roll = 00.0, distance= 30,focalpoint=(0,0,0))

    mlab.view(azimuth=20, elevation=55.0, reset_roll=True, roll = 00.0, distance= 30,focalpoint=(0,0,0))

    mlab.show()


def _test_arrow():
    #Test arrow with mayavi figure
    sceneName="myfigure"
    bgcol=(0.97, 0.97, 0.97)    #Color of the background
    fgcol=(0.07,0.07,0.07)      #color of all text annotation labels(axes, ori axes, scalar bar labels)
    fsize = (600, 600)
    mfig = mlab.figure(sceneName,bgcolor=bgcol,fgcolor=fgcol,size=fsize)
    # for parallel projection (you can also do it using the cam) do
    # mfig.scene.parallel_projection=True
    plotExtents = [-10,10,-10,10,-10,10]
    #draw the cross axis
    drawOriginAxes(plotExtents,colAxes=True,cones=True)
    #AXES
    maxes = mlab.axes(color=fgcol,extent=plotExtents,nb_labels=7)
    ##Change axes font size, font color, etc
    maxes.axes.font_factor=1
    ###OUTLINE
    #Create an outline for the data.
    outl = mlab.outline(color=(0.0,0.0,0.0),extent=plotExtents,opacity=0.25)


    #Test drawing the arrows in 3D space.
    a1 = arrow((0,0,0),(2,2,2),a_col=(0.0,0.0,0.0))
    a2 = arrow((0,0,0),(10,10,-10),a_col=(1.0,0.0,0.0))
    a3 = arrow((0,0,0),(-5,-5,-5),a_col=(0.0,0.0,1.0))
    a4 = arrow((0,0,0),(5,5,0),a_col=(0.0,1.0,0.0))
    a5 = arrow((5,5,0),(10,10,10),a_col=(0.0,1.0,1.0))

    #play of scale
    a6 = arrow((0,0,0),(-10,-5,0),a_col=(1.0,1.0,0.0),cone_scale=0.5)
    a7 = arrow((10,-10,-10),(0,0,0),a_col=(0.8,0.6,1.0),cone_scale=2.0)

    #Change the property of the cone of a6 to 1
    #print a6.cone   #the cone is a mayavi.modules.vectors.Vectors object

    a6.cone.glyph.glyph.scale_factor=1.0
    a7.cone.glyph.glyph.scale_factor=1.0
    #putting them back
    a6.cone.glyph.glyph.scale_factor=0.5
    a7.cone.glyph.glyph.scale_factor=2.0
    #change the color of the sick of a7
    a7.stick.actor.property.color=(1.0, 0.0, 1.0)
    #Set the camera
    cam = mfig.scene.camera
    cam.parallel_projection = True
    cam.zoom(1.12)
    #The following camera position: +y is up, +z is towards left, +x is coming out
    #of the screen plane towards the viewer, the camera view center is 0,0,0.
    #Setting a non-zero roll will rotate the camera in the plane perpendicular
    #to the x-axis. azimuth controls the angle in the x-y plane (0-360),
    #elevation controls the angle in the z to x-y plane (0-180)
    #mlab.view(azimuth=0, elevation=90.0, reset_roll=True, roll = 00.0, distance= 30,focalpoint=(0,0,0))

    mlab.view(azimuth=20, elevation=55.0, reset_roll=True, roll = 00.0, distance= 30,focalpoint=(0,0,0))
    #mlab.show_pipeline()
    mlab.show()

def _test_implicit_plot():
    """test implicit_plot() function
    Extensive testing already done in the IPython notebook on implicit function plotting
    """
    # Draw a unit sphere
    implicit_plot('x**2 + y**2 + z**2 - 1', (-2, 2, -2, 2, -2, 2))

if __name__ == '__main__':
    import numpy.testing as nt
    from numpy import set_printoptions
    from scipy import integrate, special
    #set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    # Automatic tests
    # Visual tests: These testing methods are meant to be manual tests which
    # requires visual inspection.
    _test_drawOriginAxes()
    #_test_mayaviFig()
    #_test_arrow
    #_test_implicit_plot()