#-------------------------------------------------------------------------------
# Name:        cgUtils.py
# Purpose:     Utility functions useful for computer graphics
#
# Author:      Indranil Sinharoy
#
# Created:     07/11/2012
# Copyright:   (c) Indranil Sinharoy, 2012 - 2013
# Licence:     MIT License
#-------------------------------------------------------------------------------
import numpy as np

def rotMat2D(angle, angleType='r'):
    """Return a 2D Rotation Matrix based on the input angle. The (in-plane) rotation is
    performed in Euclidean space.

    rotMat2D(angle [,angleType]) -> R

    Parameters
    ----------
    angle: the angle of rotation
    angleType:
            r = Radians (Default)
            d = Degrees

    Returns
    -------
        R : (numpy matrix) the rotation matrix

    Notes
    ------
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

def rotMat3D(axis, angle, angleType='r', tol=1e-12):
    """Return the rotation matrix for 3D rotation by angle `angle` and about an
    arbitrary axis `axis`.

    Parameters
    ----------
    axis : 3-tuple 
        (x, y, z) represent the arbitrary axis about which to rotate
    angle : float
        the rotation angle.
    angleType : string
        the unit of the rotation angle. `r` = radian (default), `d` = degree.
    tol : float (default=1e-12)
        set values below absolute of `tol` to zero

    Returns
    ------
    R: (numpy matrix) the 3x3 rotation matrix.

    Note
    ----
    The 3D rotation matrix is computed using the Rodrigues' rotation formula.
    See the following references:
    1. Axis-angle representation: https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
    2. Rodrigues' rotation formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    if angleType=='d':
        t = np.radians(angle)
    x, y, z = axis
    R = (np.cos(t))*np.eye(3) +\
    (1-np.cos(t))*np.matrix(((x**2,x*y,x*z),(x*y,y**2,y*z),(z*x,z*y,z**2))) + \
    np.sin(t)*np.matrix(((0,-z,y),(z,0,-x),(-y,x,0)))
    R[np.abs(R)<tol]=0.0
    return R


def _test_rotMat2D():
    # Test the function rotMat2D
    angle = 3 # 3 degrees
    R = rotMat2D(angle,'d')
    print R

def _test_rotMat3D():
    # Test the function rotMat3D
    R = rotMat3D((1,0,0),10,'d')
    print R

##    assert(R==np.array([[ 0.99862953, -0.05233596],
##        [ 0.05233596,  0.99862953]]))


if __name__ == '__main__':
    _test_rotMat2D()
    _test_rotMat3D()
