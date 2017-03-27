# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------------------
# Name:        rigidbody.py
# Purpose:     Utility functions useful for computer graphics, especially related to 
#              rigid body transformations
#
# Author:      Indranil Sinharoy
#
# Created:     07/11/2012
# Modified:    03/26/2017
# Copyright:   (c) Indranil Sinharoy, 2012 - 2017
# Licence:     MIT License
#-----------------------------------------------------------------------------------------
"""utility functions related to rigid body transformations for both computer vision and
computer graphics
"""
from __future__ import print_function, division
import numpy as _np

def dual_matrix(vec):
    """dual matrix (or the hat operator in skew theory), which is the skew-symmetric 
    matrix associated with the 3x1 vector

    Parameters
    ----------
    vec : ndarray 
        3-element numpy array in :math:`\mathbb{R}^3` space

    Returns
    -------
    vec_hat : numpy matrix
        the skew-symmetric matrix, a square matrix, associated with the vector ``vec``.

    Notes
    -----
    Given a vector :math:`v = [v_1, v_2, v_3]^T \in \mathbb{R}^3` the hat operator is

    .. math::

        \hat{v} =
        \\left[\\begin{array}{ccc}
          0  & -v_3 & v_2  \\\\
         v_3 &   0  & -v_1 \\\\
        -v_2 &  v_1 &  0
        \end{array}\\right]

    This isomorphism is represented mathematically as
    :math:`\\bigwedge : \mathbb{R}^3 \\rightarrow so(3); u \\mapsto \\hat{u}`

    Examples
    --------
    The following example also demonstrates a property of the dual matrix
    :math:`\\hat{u}`, that the column (and row) vectors span the subspace
    orthogonal to the vector :math:`u`

    >>> u = np.array([1, 2, 3])
    >>> uhat = cg.dual_matrix(u)
    >>> uhat
    matrix([[ 0., -3.,  2.],
            [ 3.,  0., -1.],
            [-2.,  1.,  0.]])
    >>> np.dot(uhat, u)
    matrix([[ 0.,  0.,  0.]])
    >>> np.dot(uhat.T, u)
    matrix([[ 0.,  0.,  0.]])
    """
    x, y, z = vec.reshape(-1)
    return _np.matrix(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))

def skew(vec):
    """return the skew-symmetric matrix from 3x1 vector ``vec``. 
    
    Parameters
    ----------
    vec : ndarray 
        3-element numpy array in :math:`\mathbb{R}^3` space

    Returns
    -------
    vec_hat : ndarray
        the skew-symmetric matrix, a square matrix, associated with the vector ``vec``.
        
    Notes
    -----
    This functions is same as ``dual_matrix()``, except that it returns ndarray.
    """
    return _np.asarray(dual_matrix(vec))
    
def rotMat2D(angle, atype='r'):
    """rotation matrix to rotate a vector/point in 2-D by `angle` in RHS
    
    Positive `angle` corresponds to rotation in counter-clockwise direction.

    Usage: ``rotMat2D(angle [,atype]) -> R``

    Parameters
    ----------
    angle : float
        the angle of rotation
    atype : string ('r' or 'd')
        r = radians (default)
        d = degrees

    Returns
    -------
    r : numpy matrix
        the rotation matrix

    Notes
    -----
    The rotation matrix, :math:`R \in SO(2)`, returned is the following form:

    .. math::

            R(\\theta) =
            \\left[\\begin{array}{lr}
            cos(\\theta) & - sin(\\theta) \\\\
            sin(\\theta) & cos(\\theta)
            \end{array}\\right]

    The rotation matrix :math:`R` rotates points/vectors in the xy-Cartesian plane 
    counter-clockwise by an angle :math:`\\theta` about the origin of the cartesian 
    coordinate system.

    To perform the rotation using the matrix :math:`R`, the position of each 
    point must be represented by a column vector :math:`v`, containing the coordinates 
    of the point. A rotated vector is obtained by using the matrix multiplication 
    :math:`Rv`.
    """
    if atype=='d':
        angle = _np.radians(angle)
    r = _np.matrix(((_np.cos(angle),-_np.sin(angle)),
                   (_np.sin(angle), _np.cos(angle))))

    return r

def rotMat3D(axis, angle, atype='r', tol=1e-12):
    """rotation matrix for rotating a vector/point about an arbitrary axis by an angle.

    Parameters
    ----------
    axis : 3-tuple
        (x, y, z) represent the arbitrary axis about which to rotate
    angle : float
        the rotation angle.
    atype : string
        the unit of the rotation angle. ``r`` = radian (default),
        ``d`` = degree.
    tol : float (default=1e-12)
        set values below absolute of ``tol`` to zero

    Returns
    -------
    r : numpy matrix
        the 3x3 rotation matrix.

    Notes
    -----
    the rotation matrix is computed using the Rodrigues' rotation formula [1]_, [2]_:

    .. math::

        R(\\theta) = I cos(\\theta) + sin(\\theta) \\hat{k} + (1 - cos(\\theta))kk^T

    where, :math:`\\theta` is the angle of rotation, and :math:`k` is the axis about 
    which the rotation is to be performed.

    References
    ----------
    .. [1] Axis-angle representation : https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation

    .. [2] Rodrigues' rotation formula : https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    t = _np.radians(angle) if atype=='d' else angle
    cos, sin, I = _np.cos, _np.sin, _np.identity
    k = _np.array(axis).reshape(3, 1)
    r = cos(t)*I(3) + sin(t)*dual_matrix(k) + (1 - cos(t))*k*k.T
    r[_np.abs(r) < tol] = 0.0
    return r

def rot2(theta, deg=True):
    """returns 2D rotation matrix :math:`R \in SO(2)`.

    Parameters
    ----------
    theta : float
        the angle of rotation
    deg : bool
        ``True`` = degree (default), ``False`` = radians

    Returns
    -------
    r : ndarray
        the rotation matrix

    Notes
    -----
    Same as the function ``rotMat2D()`` execpt for slight change in the input parameter
    specification, plus ``rot2()`` returns ``ndarray`` instead of numpy matrix. 
    See the function's docstring for details.
    
    See Also
    --------
    rotMat2D()
    """
    atype = 'd' if deg else 'r'
    return _np.asarray(rotMat2D(angle=theta, atype=atype))
    
def rotX(theta, deg=True):
    """3D matrix :math:`R \in SO(3)` for rotating a vector/point about the x-axis

    Parameters
    ----------
    theta : float
        the angle of rotation about x-axis 
    deg : bool
        ``True`` = degree (default), ``False`` = radians

    Returns
    -------
    r : numpy 3x3 matrix
        the rotation matrix
        
    Notes
    -----
    The rotation matrix, :math:`R \in SO(3)`, returned is the following form:

    .. math::

            R(\\theta) =
            \\left[\\begin{array}{ccc}
            1 & 0 & 0 \\\\
            0 & cos(\\theta) & - sin(\\theta)\\\\
            0 & sin(\\theta) & cos(\\theta)
            \end{array}\\right]

    See also: `rotMat3D()` for rotation about an arbitrary axis using axis angle formula
    """
    axis = (1, 0, 0)
    angle = _np.deg2rad(theta) if deg else theta
    return rotMat3D(axis, angle)

def rotY(theta, deg=True):
    """returns 3D matrix :math:`R \in SO(3)` for rotating a vector/point about the y-axis

    Parameters
    ----------
    theta : float
        the angle of rotation about y-axis 
    deg : bool
        ``True`` = degree (default), ``False`` = radians

    Returns
    -------
    r : numpy 3x3 matrix
        the rotation matrix
        
    Notes
    -----
    The rotation matrix, :math:`R \in SO(3)`, returned is the following form:

    .. math::

            R(\\theta) =
            \\left[\\begin{array}{ccc}
            cos(\\theta) & 0 & sin(\\theta)\\\\
            0 & 1 & 0 \\\\
            -sin(\\theta) & 0 & cos(\\theta)
            \end{array}\\right]
    
    See also: `rotMat3D()` for rotation about an arbitrary axis using axis angle formula
    """
    axis = (0, 1, 0)
    angle = _np.deg2rad(theta) if deg else theta
    return rotMat3D(axis, angle)
  
def rotZ(theta, deg=True):
    """returns 3D matrix :math:`R \in SO(3)` for rotating a vector/point about the z-axis

    Parameters
    ----------
    theta : float
        the angle of rotation about x-axis 
    deg : bool
        ``True`` = degree (default), ``False`` = radians

    Returns
    -------
    r : numpy 3x3 matrix
        the rotation matrix
        
    Notes
    -----
    The rotation matrix, :math:`R \in SO(3)`, returned is the following form:

    .. math::

            R(\\theta) =
            \\left[\\begin{array}{ccc}
            cos(\\theta) & -sin(\\theta) & 0 \\\\
            sin(\\theta) & cos(\\theta) & 0 \\\\
             0 & 0 & 1 
            \end{array}\\right]
            
    See also: `rotMat3D()` for rotation about an arbitrary axis using axis angle formula
    """
    axis = (0, 0, 1)
    angle = _np.deg2rad(theta) if deg else theta
    return rotMat3D(axis, angle)

def rotXYZ_intrinsic(phi, theta, psi, order='X-Y-Z', deg=True):
    """returns composed rotation matrix from Euler angles (xy'z''), s.t. the elementary
    rotations are intrinsic

    Parameters
    ----------
    phi : float
        angle of rotation about the x axis (x)  
    theta : float
        angle of rotation about the new y axis (y') 
    psi : float
        angle of rotation about the new z axis (z") 
    order : string 
        valid string sequence that specifies the order of rotation. For example, 
        'X-Y-Z' represents first rotation about x-axis, followed by second
        rotation about the y-axis, followed by third rotation about the z-axis.
    deg : bool
        `True` = degree (default), `False` = radians

    Returns
    -------
    r : ndarray
        the rotation matrix
    """
    X = rotX(phi, deg)
    Y = rotY(theta, deg)
    Z = rotZ(psi, deg)
    assert isinstance(X, _np.matrix) # in order to use the '*' operator to matrix multiply
    assert isinstance(Y, _np.matrix)
    assert isinstance(Z, _np.matrix)
    order = order.split('-')
    composition = '*'.join(order)
    return eval(composition)

# After complete movement to Python 3, the function parameters could be changed to:
#def rotXYZ_intrinsic(ϕ, θ, ψ, deg=True):
#    """returns composed rotation matrix from Euler angles (xy'z''), s.t. the elementary
#    rotations are intrinsic
#
#    Parameters
#    ----------
#    ϕ : float
#        angle of rotation about the x axis (x) 
#    θ : float
#        angle of rotation about the new y axis (y') 
#    ψ : float
#        angle of rotation about the new z axis (z")
#
#    Returns
#    -------
#    r : ndarray
#        the rotation matrix
#    """
#    rx = rotX(ϕ, deg)
#    ry = rotY(θ, deg)
#    rz = rotZ(ψ, deg)
#    return rx*ry*rz

def rotXYZ_extrinsic(phi, theta, psi, order='X-Y-Z', deg=True):
    """returns composed rotation matrix from Euler angles (xyz), s.t. the elementary
    rotations are extrinsic

    Parameters
    ----------
    phi : float
        angle of rotation about the x axis (x) 
    theta : float
        angle of rotation about the y axis (y) 
    psi : float
        angle of rotation about the z axis (z) 
    order : string 
        valid string sequence that specifies the order of rotation. For example, 
        'X-Y-Z' represents first rotation about x-axis, followed by second
        rotation about the y-axis, followed by third rotation about the z-axis.
    deg : bool
        `True` = degree (default), `False` = radians

    Returns
    -------
    r : ndarray
        the rotation matrix
    """
    X = rotX(phi, deg)
    Y = rotY(theta, deg)
    Z = rotZ(psi, deg)
    assert isinstance(X, _np.matrix) # in order to use the '*' operator to matrix multiply
    assert isinstance(Y, _np.matrix)
    assert isinstance(Z, _np.matrix)
    order = order.split('-')
    order.reverse()
    composition = '*'.join(order)
    return eval(composition)

def se2(x, y, theta=0, deg=True):
    """returns planar translation and rotation transformation matrix SE(2) as a 
    homogeneous (3x3) transformation matrix 
    
    Parameters
    ----------
    x : float
        translation along x-axis
    y : float
        translation along y-axis
    theta : float
        angle of rotation in the plane
    deg : bool
        ``True`` = degree (default), ``False`` = radians 
        
    Returns
    -------
    T : ndarray
        homogeneous 3x3 transformation matrix of the form:

    .. math::

            T(x, y, \\theta) =
            \\left[\\begin{array}{ccc}
            cos(\\theta) & - sin(\\theta)  & x \\\\
            sin(\\theta) &   cos(\\theta)  & y \\\\
                0        &     0           & 1
            \end{array}\\right]
    
    References
    ----------
    .. [1] Robotics, Vision and Control: Fundamental Algorithms in MATLAB, Peter Corke
    """
    r00, r01, r10, r11 = rot2(theta, deg).reshape(-1)
    T = _np.array([[r00, r01,   x],
                   [r10, r11,   y],
                   [0.0, 0.0, 1.0]])
    return T
    


#%%  TEST FUNCTIONS

def _test_dual_matrix():
    """basic test for dual_matrix() function
    """
    v = _np.array([1, 2, 3])
    dm = _np.matrix(([[ 0., -3.,  2.],
                      [ 3.,  0., -1.],
                      [-2.,  1.,  0.]]))
    _nt.assert_array_almost_equal(dm, dual_matrix(v), decimal=6)
    print("test dual_matrix() successful")

def _test_skew():
    """basic test for skew() function
    """
    v = _np.array([1, 2, 3])
    dm = dual_matrix(v)
    sk = skew(v)
    _nt.assert_almost_equal(dm, sk, decimal=6)
    print("test skew() successful")
    
def _test_rotMat2D():
    """test the function rotMat2D()
    """
    # simple test
    angle = 45 # 45 degrees
    r = rotMat2D(angle,'d')
    v = 1/_np.sqrt(2)
    rExp = _np.matrix([[v, -v],[v, v]])
    _nt.assert_array_almost_equal(r, rExp) # this is probably a good way to test for float values
    angRadians = _np.deg2rad(45)
    rr = rotMat2D(angRadians)
    _nt.assert_array_almost_equal(rr, rExp)
    # product of two rotation matrices
    randomAngle = lambda: _np.random.random_integers(0, 90)
    ra1, ra2 = randomAngle(), randomAngle()
    ra3 = ra1 + ra2
    r1 = rotMat2D(ra1, 'd')
    r2 = rotMat2D(ra2, 'd')
    r3 = rotMat2D(ra3, 'd')
    r2r1 = r1*r2
    r1r2 = r2*r1
    _nt.assert_array_almost_equal(r2r1, r1r2)
    _nt.assert_array_almost_equal(r2r1, r3)
    # rotation matrix properties
    _nt.assert_almost_equal(_lalg.det(r2), 1.0, decimal=8) # det() = +1
    _nt.assert_array_almost_equal(r2*r2.T, _np.identity(2)) # orthogonal matrix
    _nt.assert_array_almost_equal(r2.T, _lalg.inv(r2))    # inverse = transpose
    print("test rotMat2D() successful")

def _test_rotMat3D():
    """test the function rotMat3D
    """
    randomAngle = lambda: _np.random.random_integers(0, 90)
    angle = randomAngle()
    r = rotMat3D((1, 0, 0), angle, 'd')
    c, s = _np.cos(_np.deg2rad(angle)), _np.sin(_np.deg2rad(angle))
    rExp = _np.matrix([[ 1.0, 0.0, 0.0],
                       [ 0.0,   c,  -s],
                       [ 0.0,   s,   c]])
    _nt.assert_array_almost_equal(r, rExp, decimal=8)
    # rotation matrix properties
    _nt.assert_almost_equal(_lalg.det(r), 1.0, decimal=8) # det() = +1
    _nt.assert_array_almost_equal(r*r.T, _np.identity(3)) # orthogonal matrix
    _nt.assert_array_almost_equal(r.T, _lalg.inv(r))    # inverse = transpose
    print("test rotMat3D() successful")

def _test_rot2():
    theta = 15.0    
    r1 = rot2(theta)
    r2 = rot2(_np.deg2rad(theta), deg=False)    
    _nt.assert_array_almost_equal(r1, r2)
    print("test rot2() successful")

def _test_rotX():
    theta = 15.0
    r1 = rotX(theta)
    assert isinstance(r1, _np.matrix)
    r2 = rotX(_np.deg2rad(theta), deg=False)
    _nt.assert_array_almost_equal(r1, r2)
    print("test rotX() successful")   
    
def _test_rotY():
    theta = 15.0
    r1 = rotY(theta)
    assert isinstance(r1, _np.matrix)
    r2 = rotY(_np.deg2rad(theta), deg=False)
    _nt.assert_array_almost_equal(r1, r2)
    print("test rotY() successful")
    
def _test_rotZ():
    theta = 15.0
    r1 = rotZ(theta)
    assert isinstance(r1, _np.matrix)
    r2 = rotZ(_np.deg2rad(theta), deg=False)
    _nt.assert_array_almost_equal(r1, r2)
    print("test rotZ() successful")
    
def _test_rotXYZ_intrinsic():
    phi, theta, psi = 20, 30, 40
    r = rotXYZ_intrinsic(phi, theta, psi)
    assert isinstance(r, _np.matrix)
    re = rotX(phi)*rotY(theta)*rotZ(psi)
    _nt.assert_array_almost_equal(r, re)
    r = rotXYZ_intrinsic(phi, theta, psi, order='Z-Y-X')
    re = rotZ(psi)*rotY(theta)*rotX(phi)
    _nt.assert_array_almost_equal(r, re)
    print('test rotXYZ_intrinsic() successful')

def _test_rotXYZ_extrinsic():
    phi, theta, psi = 20, 30, 40
    r = rotXYZ_extrinsic(phi, theta, psi)
    assert isinstance(r, _np.matrix)
    re = rotZ(psi)*rotY(theta)*rotX(phi)
    _nt.assert_array_almost_equal(r, re)
    r = rotXYZ_extrinsic(phi, theta, psi, order='Z-Y-X')
    re = rotX(phi)*rotY(theta)*rotZ(psi) 
    _nt.assert_array_almost_equal(r, re)
    print('test rotXYZ_extrinsic() successful')

def _test_se2():
    T1 = se2(1, 2, 30)
    T1exp = _np.array([[ 0.866025,      -0.5,  1.0],
                       [      0.5,  0.866025,  2.0],
                       [      0.0,       0.0,  1.0]])
    _nt.assert_array_almost_equal(T1, T1exp, decimal=6)
    print("test se2() successful")

#%%
if __name__ == '__main__':
    import numpy.testing as _nt
    import numpy.linalg as _lalg
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    _test_dual_matrix() 
    _test_skew()
    _test_rotMat2D()
    _test_rotMat3D()
    _test_rot2()
    _test_rotX()
    _test_rotY()
    _test_rotZ()
    _test_rotXYZ_intrinsic()
    _test_rotXYZ_extrinsic()
    _test_se2()