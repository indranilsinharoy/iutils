# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 22:09:57 2013

TO USE THE UTILITIES IN THE PROGRAMS USE THE FOLLOWING LINE OF CODE:
    
    import sys
    sys.path.append('C:\\PROGRAMSANDEXPERIMENTS\\PYTHON\\PyUtils')
    import plottingUtils as pu

@author: Indranil
"""

import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt


class arrow(object):
    def __init__(self,start,end,a_col=(0.0,0.0,0.0),cone_scale=1.0,fig_object=None,alpha=1.0):
        """Arrow class to draw arrow
        start = starting point of the arrow. It is a 3-d vector for 3D plots (usually)
        plotted using Mayavi, or a 2-d vector for 2D plots, usually, using matplotlib
        end = end point of the arrow. It is a 3-d vector for 3D plots (usually)
        plotted using Mayavi, or a 2-d vector for 2D plots, usually, using matplotlib
        """
                
        if fig_object != None:        
            self.fig = None
        else:
            self.fig = fig_object
            
        if len(end) > 2:
            #plot using Mayavi
            #CALCULATE CONE SIZE/SCALE. The unscaled cone size is "1.0"    
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
            
            #STICK
            x_stick = np.array([start[0],x_dash])
            y_stick = np.array([start[1],y_dash])
            z_stick = np.array([start[2],z_dash])
            self.cone = mlab.quiver3d(x_dash, y_dash, z_dash, u, v, w, mode='cone',\
            color=a_col,resolution=64,scale_factor=cone_scale)
            self.stick = mlab.plot3d(x_stick,y_stick,z_stick,color=a_col,\
            tube_radius=0.05,name='stick')
        else:
            #plot using Matplotlib
            arr_head2length_ratio = 0.1
            dx = (end[0]-start[0])
            dy = (end[1]-start[1])
            arr_length = np.sqrt(dx**2.0 + dy**2.0)
            
            alpha = alpha
            width = 0.01
            head_width = 0.15
            head_length = arr_head2length_ratio*arr_length
            
            self.twoDarrow = plt.arrow(start[0],start[1],dx,dy,\
            color=a_col,alpha=alpha, width=width, head_width=head_width,\
            head_length=head_length,length_includes_head=True)
            



def drawOriginAxes(plotExtents,displace=None,colAxes=True,cones=True,\
xaxis=True,yaxis=True,zaxis=True,visible=True):
    """function to draw crossed axes through the origin. The three axes have the
    option of adding a cone, and also may or may not have different colors.
    The axes are also labeled.
    For simple, crossed axes through the center, without different colors, and
    without lables, it is more efficient to use:
        x = np.array([0]);y = np.array([0]);z = np.array([0]);
        ss = some value (extent of the axes)
        mlab.points3d(x, y, z, ss,mode='axes',color=(0,1,0), scale_factor=1)
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
        ext2subtract = 1.0
    oa_xlim = np.array([plotExtents[0],plotExtents[1]-ext2subtract])
    oa_ylim = np.array([plotExtents[2],plotExtents[3]-ext2subtract])
    oa_zlim = np.array([plotExtents[4],plotExtents[5]-ext2subtract])
    
    center = np.array([0,0])
    oa_colork = (0.1,0.1,0.1)
    
    if colAxes:
        oa_colR = (1.0,0,0)
        oa_colG = (0,1.0,0)
        oa_colB = (0,0,1.0)
    else:
        oa_colR,oa_colG,oa_colB = (0.1,0.1,0.1),(0.1,0.1,0.1),(0.1,0.1,0.1)
    

        
    
    # x-axis 
    if xaxis:
        x = np.array([oa_xlim[1]]);y = np.array([0]);z = np.array([0])
        u = np.array([1]);v = np.array([0]);w = np.array([0])
        if cones:
            x_cone = mlab.quiver3d(x, y, z, u, v, w, mode='cone',color=oa_colR, scale_factor=1.0)
        x_axis = mlab.plot3d(oa_xlim,center,center,color=oa_colR,line_width=1.0,\
        tube_radius=0.05,name='oax')
        xaxis_label = mlab.text3d(0.9*oa_xlim[1],0,0,'x',scale=0.5,color=oa_colork)
    
    # y-axis
    if yaxis:
        x = np.array([0]);y = np.array([oa_ylim[1]]);z = np.array([0])
        u = np.array([0]);v = np.array([1]);w = np.array([0])
        if cones:
            y_cone = mlab.quiver3d(x, y, z, u, v, w, mode='cone',color=oa_colG, scale_factor=1.0)
        y_axis = mlab.plot3d(center,oa_ylim,center,color=oa_colG,line_width=1.0,\
        tube_radius=0.05,name='oay')
        yaxis_label = mlab.text3d(0,0.9*oa_ylim[1],0,'y',scale=0.5,color=oa_colork)
    
    # z-axis
    if zaxis:
        x = np.array([0]);y = np.array([0]);z = np.array([oa_zlim[1]])
        u = np.array([0]);v = np.array([0]);w = np.array([1])
        if cones:
            z_cone = mlab.quiver3d(x, y, z, u, v, w, mode='cone',color=oa_colB, scale_factor=1.0)
        z_axis = mlab.plot3d(center,center,oa_zlim,color=oa_colB,line_width=1.0,\
        tube_radius=0.05,name='oaz')
        zaxis_label = mlab.text3d(0,0,0.9*oa_zlim[1],'z',scale=0.5,color=oa_colork)
    
    if visible==False:
        z_axis.actor.actor.visibility=False
        zaxis_label.actor.actor.visibility=False
    
def myMayaviFig(sceneName="Figure",plotExtents=[-10,10,-10,10,-10,10],crossAxes=True,oriAxes=True,\
colCrossAxes=True,fsize = (400, 350),bgcol=(0.97, 0.97, 0.97),\
fgcol=(0.07,0.07,0.07)):
    """my custom mayavi figure. It returns the handle to the figure (scene in 
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


def test_drawOriginAxes():
    pass




def test_myMayaviFig():
    ##FIGURE
    sceneName="myfigure"
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


def test_arrow_1():
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
    
def test_arrow_2():
    #test arrow with matplotlib figure
    fig = plt.figure("myfigure",facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    plt.grid()
    #Test drawing the arrows in 2D space
    a1 = arrow((0,0),(2,2),a_col=(0.0,0.0,0.0))
    a2 = arrow((0,0),(-2,2),a_col=(1.0,0.0,0.0))
    a3 = arrow((-2,2),(2,2),a_col=(0.0,0.0,1.0))
    a4 = arrow((0,0),(-3,-3),a_col=(0.0,1.0,1.0))
    a4.twoDarrow.set_linestyle('dashed')
    #passing numpy array vectors
    ori = np.array((0,0))
    v1 = np.array((3,-3))
    a5 = arrow(ori,v1,'c')

    
    
    plt.show()
 
if __name__ == '__main__':
    #test_drawOriginAxes()
    #test_myMayaviFig()
    #test_arrow_1()
    test_arrow_2()
    