#-------------------------------------------------------------------------------
# Name:        cgUtils.py
# Purpose:     Utility functions useful for computer graphics
#
# Author:      Indranil
#
# Created:     07/11/2012
# Copyright:   (c) Indranil 2012
# Licence:     MIT License
#-------------------------------------------------------------------------------
import numpy as np

def rotMat2D(angle,angleType='r'):
    """Return a 2D Rotation Matrix based on the input angle. The rotation is
    performed in Euclidean space.

    rotMat2D(angle [,angleType]) -> R

    args:
        angle: the angle of rotation
        angleType:
            r = Radians (Default)
            d = Degrees
    ret:
        R (rotation matrix)

    The rotation matrix R rotates points in the xy-Cartesian plane
    counterclockwise through an angle \theta about the origin of the Cartesian
    coordinate system. To perform the rotation using a rotation matrix R, the
    position of each point must be represented by a column vector v, containing
    the coordinates of the point. A rotated vector is obtained by using the matrix
    multiplication Rv.
    """
    if angleType=='d':
        angle = np.radians(angle)
    R = np.matrix(((np.cos(angle),-np.sin(angle)),
                   (np.sin(angle), np.cos(angle))))

    return R
    
def rotMat3D(axis,angle,angleType='r'):
    """Return the rotation matrix for 3D rotation by angle "angle" and aobut the
    axis "axis".
    """
    if angleType=='d':
        t = np.radians(angle)
    x = axis[0]; y = axis[1]; z = axis[2];
    R = (np.cos(t))*np.eye(3) +\
    (1-np.cos(t))*np.matrix(((x**2,x*y,x*z),(x*y,y**2,y*z),(z*x,z*y,z**2))) + \
    np.sin(t)*np.matrix(((0,-z,y),(z,0,-x),(-y,x,0)))
    return R


def test_rotMat2D():
    # Test the function rotMat2D
    angle = 3 # 3 degrees
    R = rotMat2D(angle,'d')
    print R

##    assert(R==np.array([[ 0.99862953, -0.05233596],
##        [ 0.05233596,  0.99862953]]))


if __name__ == '__main__':
    test_rotMat2D()
