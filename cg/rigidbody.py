# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------------------
# Name:        rigidbody.py
# Purpose:     Utility functions useful for computer graphics, especially related to 
#              rigid body transformations
#
# Author:      Indranil Sinharoy
#
# Created:     07/11/2012
# Modified:    03/30/2017
# Copyright:   (c) Indranil Sinharoy, 2012 - 2017
# Licence:     MIT License
#-----------------------------------------------------------------------------------------
"""utility functions related to rigid body transformations for both computer vision and
computer graphics
"""
from __future__ import print_function, division
import numpy as _np
import sympy as _sy

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
    """returns rotation matrix to rotate a vector/point in 2-D by `angle` about the
    origin in counter-clockwise direction
    
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
    """returns 3D rotation matrix for rotating a vector/point about an arbitrary 
    `axis` by `angle` in RHS.

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
    """returns 2D rotation matrix :math:`R \in SO(2)` to rotate a vector/point in a
    plane in counter-clockwise direction

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


def euler2rot(phi, theta, psi, order='X-Y-Z', ertype='extrinsic', deg=True):
    """returns rotation matrix from Euler angles

    Parameters
    ----------
    phi : float
        angle for first rotation   
    theta : float
        angle for second rotation   
    psi : float
        angle for third rotation  
    order : string 
        valid string sequence that specifies the order of rotation. Examples are 'X-Y-Z',
        'Z-Y-Z', 'z-x-z', 'z-x-y'. Furthermore, if `ertype` is "intrinsic", then 'X-Y-Z' 
        returns :math:`R=R_x(\phi)*R_y(\\theta)*R_z(\psi)` and if `ertype` is "extrinsic", 
        then the same sequence, 'X-Y-Z', returns :math:`R=R_z(\psi)*R_y(\\theta)*R_x(\phi)` 
    ertype : string ('extrinsic' or 'intrinsic')
        the type of elemental rotations. `extrinsic` represent rotations about the axes of
        the fixed origial coordinate system, `intrinsic` represent rotations about the axes
        of the rotating coordinate system attached to the rigid body
    deg : bool
        `True` = degree (default), `False` = radians

    Returns
    -------
    r : ndarray
        the rotation matrix
        
    Notes
    -----
    In the context of this function "Euler angles" constitutes both the Proper Euler angles
    and the Tait-Bryan angles.
    
    References
    ----------
    .. [1] Euler angles: https://en.wikipedia.org/wiki/Euler_angles
    """ 
    X = rotX 
    Y = rotY 
    Z = rotZ 
    order = order.upper()
    order = order.split('-')
    if not set(order).issubset({'X', 'Y', 'Z'}):
        raise ValueError('Incorrect order parameter ({}) specified.'.format(order))
    if ertype == 'extrinsic':
        order.reverse()
        composition = '{}(psi, deg)*{}(theta, deg)*{}(phi, deg)'.format(*order)
    elif ertype == 'intrinsic':
        composition = '{}(phi, deg)*{}(theta, deg)*{}(psi, deg)'.format(*order)
    else:
        raise ValueError('Incorrect elemental rotation parameter ({}) specified.'.format(ertype))
    #print(composition)
    return eval(composition)
    
 
def se2(x, y, theta=0, deg=True):
    """returns homogeneous transformation matrix SE(2) for planar translation and rotation  
    
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
    T : numpy matrix
        homogeneous 3x3 transformation matrix of the form:

    .. math::

            T(x, y, \\theta) =
            \\left[\\begin{array}{ccc}
            cos(\\theta) & - sin(\\theta)  & x \\\\
            sin(\\theta) &   cos(\\theta)  & y \\\\
                0        &     0           & 1
            \end{array}\\right]

    Example
    -------
    >>> T = rg.se2(1, 2, 30)
    matrix([[ 0.8660254, -0.5      ,  1.       ],
            [ 0.5      ,  0.8660254,  2.       ],
            [ 0.       ,  0.       ,  1.       ]])
    
    References
    ----------
    .. [1] Robotics, Vision and Control: Fundamental Algorithms in MATLAB, Peter Corke
    .. [2] Code: https://github.com/petercorke/robotics-toolbox-matlab
    """
    T = homo(rot2(theta, deg))
    T[:2, 2] = [x, y]
    return _np.matrix(T)
   
#%% Utility functions

def homo(m):
    """returns a homogeneous matrix constructed from the matrix `m`

    Parameters
    ----------
    m : ndarray or numpy matrix
        a 2x2 or 3x3 ndarray or matrix. Usually `m` is SO(2) or SO(3).

    Returns
    -------
    h : ndarray or numpy matrix
        a 3x3 or 4x4 homogeneous matrix
    """
    rows, cols = m.shape
    assert rows == cols
    h = _np.eye(rows + 1, cols + 1)
    h[:rows, :cols] = m[:, :]
    if isinstance(m, _np.matrix):
        return _np.matrix(h)
    else:
        return h

#%% Symbolic Computation functions (experimental, requires Sympy)

def rotX_symbolic(angle='ϕ'):
    t = _sy.symbols(angle, real=True)
    r = _sy.Matrix(((1,       0,              0 ), 
                    (0,  _sy.cos(t),  -_sy.sin(t)),
                    (0,  _sy.sin(t),   _sy.cos(t)),
                    ))
    return r

def rotY_symbolic(angle='θ'):
    t = _sy.symbols(angle, real=True)
    r = _sy.Matrix(((_sy.cos(t),  0, _sy.sin(t)), 
                    (     0,      1,     0     ),
                    (-_sy.sin(t), 0, _sy.cos(t)),
                    ))
    return r

def rotZ_symbolic(angle='ψ'):
    t = _sy.symbols(angle, real=True)
    r = _sy.Matrix(((_sy.cos(t), -_sy.sin(t), 0), 
                    (_sy.sin(t),  _sy.cos(t), 0),
                    (     0,          0,      1),
                    ))
    return r

def euler2rot_symbolic(angle1='ϕ', angle2='θ', angle3='ψ', order='X-Y-Z', ertype='extrinsic'):
    """returns symbolic expression for the composition of elementary rotation matrices
    
    Example
    -------
    >>> R = euler2rot_symbolic('1', '2', '3', 'X-Y-Z' , 'intrinsic')
    >>> c, s = sy.symbols('c, s', cls=sy.Function)
    >>> R.subs({sy.cos:c, sy.sin:s})
    Matrix([
           [                  c(2)*c(3),                 -c(2)*s(3),       s(2)],
           [ c(1)*s(3) + c(3)*s(1)*s(2), c(1)*c(3) - s(1)*s(2)*s(3), -c(2)*s(1)],
           [-c(1)*c(3)*s(2) + s(1)*s(3), c(1)*s(2)*s(3) + c(3)*s(1),  c(1)*c(2)]])
    """
    X = rotX_symbolic 
    Y = rotY_symbolic 
    Z = rotZ_symbolic
    order = order.split('-')
    if ertype == 'extrinsic':
        order.reverse()
        composition = '{}(angle3)*{}(angle2)*{}(angle1)'.format(*order)
    elif ertype == 'intrinsic':
        composition = '{}(angle1)*{}(angle2)*{}(angle3)'.format(*order)
    else:
        raise ValueError('Incorrect elemental rotation parameter.')
    #print(composition)
    return eval(composition)

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
      
def _test_euler2rot():
    # check with verified, known result 
    r = euler2rot(0.1, 0.2, 0.3, 'Z-Y-Z', 'intrinsic', False)
    assert isinstance(r, _np.matrix)
    rexp = _np.matrix([[ 0.902113  , -0.38355704,  0.19767681],
                       [ 0.3875172 ,  0.92164909,  0.01983384],
                       [-0.18979606,  0.0587108 ,  0.98006658]])
    _nt.assert_array_almost_equal(r, rexp, decimal=6)
    # validate intrinsic rotation
    r = euler2rot(20, 30, 40, 'X-Y-Z', 'intrinsic')
    rexp = rotX(20)*rotY(30)*rotZ(40)
    _nt.assert_array_almost_equal(r, rexp, decimal=6)
    r = euler2rot(20, 30, 40, 'Z-Y-Z', 'intrinsic')
    rexp = rotZ(20)*rotY(30)*rotZ(40)
    _nt.assert_array_almost_equal(r, rexp, decimal=6)
    # validate extrinsic rotations
    r = euler2rot(20, 30, 40, 'Z-Y-Z', 'extrinsic')
    rexp = rotZ(40)*rotY(30)*rotZ(20)
    _nt.assert_array_almost_equal(r, rexp, decimal=6)
    r = euler2rot(20, 30, 40, 'X-Y-Z', 'extrinsic')
    rexp = rotZ(40)*rotY(30)*rotX(20)
    _nt.assert_array_almost_equal(r, rexp, decimal=6)
    # test sequence specification in lower and upper hand
    r = euler2rot(20, 30, 40, 'x-Y-z', 'extrinsic')
    _nt.assert_array_almost_equal(r, rexp, decimal=6)
    # test exceptions
    try:
        r = euler2rot(20, 30, 40, 'X-Y-K', 'intrinsic')
    except Exception as err:
        _nt.assert_equal(isinstance(err, ValueError), True)
        print('\t...raised expected exception.')
    try:
        r = euler2rot(20, 30, 40, 'X-Y-Z', 'invalider')
    except Exception as err:
        _nt.assert_equal(isinstance(err, ValueError), True)
        print('\t...raised expected exception.')
    print('test euler2rot() successful')

def _test_se2():
    T1 = se2(1, 2, 30)
    T1exp = _np.array([[ 0.866025,      -0.5,  1.0],
                       [      0.5,  0.866025,  2.0],
                       [      0.0,       0.0,  1.0]])
    _nt.assert_array_almost_equal(T1, T1exp, decimal=6)
    print("test se2() successful")
    
def _test_homo():
    h1 = homo(rot2(30))
    _nt.assert_array_equal(_np.shape(h1), (3, 3))
    assert isinstance(h1, _np.ndarray)
    _nt.assert_array_equal(h1[2:], _np.array([[0, 0, 1]])) 
    _nt.assert_array_equal(h1[:,2], _np.array([0, 0, 1]))
    _nt.assert_array_almost_equal(h1[:2, :2], rot2(30))
    h2 = homo(rotMat2D(30))
    _nt.assert_array_equal(_np.shape(h2), (3, 3))
    assert isinstance(h2, _np.matrix)
    r = euler2rot(20, 30, 40)
    h3 = homo(r)
    _nt.assert_array_equal(_np.shape(h3), (4, 4))
    assert isinstance(h3, _np.matrix) 
    _nt.assert_array_almost_equal(h3[:3, :3], r)
    print('test homo() successful')

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
    _test_euler2rot()
    _test_se2()
    _test_homo()