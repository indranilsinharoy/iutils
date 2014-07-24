#-------------------------------------------------------------------------------
# Name:        cgUtils.py
# Purpose:     Utility functions useful for computer graphics
#
# Author:      Indranil Sinharoy
#
# Created:     07/11/2012
# Modified:    07/22/2014
# Copyright:   (c) Indranil Sinharoy, 2012 - 2014
# Licence:     MIT License
#-------------------------------------------------------------------------------
from __future__ import print_function, division
import numpy as _np

def dual_matrix(vec):
    """Returns the dual matrix, also known as the hat operator in skew
    theory. The dual matrix is the skew-symmetric matrix associated
    with the 3x1 vector

    Parameters
    ----------
    vec : 3-element numpy array or 3x1 numpy matrix object
        vector in :math:`\mathbb{R}^3` space

    Returns
    -------
    vec_hat : numpy matrix
        the skew-symmetric matrix, a square matrix, associated with the
        vector ``vec``.

    Notes
    -----
    The dual matrix, or the hat operator returns a 3x3 skew-symmetric
    matrix as shown below. Given a vector
    :math:`v = [v_1, v_2, v_3]^T \in \mathbb{R}^3` the hat operator
    returns:

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
    x, y, z = [elem.item() for elem in vec]
    return _np.matrix(((0.0, -z, y),(z, 0.0, -x), (-y, x, 0.0)))

def rotMat2D(angle, atype='r'):
    """Return a 2D rotation matrix, based on the input angle.

    The (in-plane) rotation is performed in Euclidean space.

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
    The rotation matrix, :math:`R \in SO(2)`, returned is
    the following form:

    .. math::

            R(\\theta) =
            \\left[\\begin{array}{lr}
            cos(\\theta) & - sin(\\theta) \\\\
            sin(\\theta) & cos(\\theta)
            \end{array}\\right]

    The rotation matrix :math:`R` rotates points/vectors in the
    xy-Cartesian plane counter-clockwise through an angle
    :math:`\\theta` about the origin of the cartesian coordinate
    system.

    To perform the rotation using the rotation matrix
    :math:`R`, the position of each point must be represented by
    a column vector :math:`v`, containing the coordinates of the
    point. A rotated vector is obtained by using the matrix
    multiplication :math:`Rv`.
    """
    if atype=='d':
        angle = _np.radians(angle)
    r = _np.matrix(((_np.cos(angle),-_np.sin(angle)),
                   (_np.sin(angle), _np.cos(angle))))

    return r

def rotMat3D(axis, angle, atype='r', tol=1e-12):
    """Return the rotation matrix for 3D rotation by angle ``angle``
    and about an arbitrary axis ``axis``.

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
    The 3D rotation matrix is computed using the Rodrigues' rotation
    formula which has the following form [1]_, [2]_:

    .. math::

        R(\\theta) = I cos(\\theta) + sin(\\theta) \\hat{k} + (1 - cos(\\theta))kk^T

    where, :math:`\\theta` is the angle of rotation, and :math:`k` is the
    axis about which the rotation is to be performed.

    References
    ----------
    .. [1] Axis-angle representation : https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation

    .. [2] Rodrigues' rotation formula : https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    t = _np.radians(angle) if atype=='d' else angle
    cos, sin, I = _np.cos, _np.sin, _np.identity
    k = _np.array(axis).reshape(3,1)
    r = cos(t)*I(3) + sin(t)*dual_matrix(k) + (1-cos(t))*k*k.T
    r[_np.abs(r)<tol]=0.0
    return r

def _test_rotMat2D():
    # Test the function rotMat2D
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
    # Test the function rotMat3D
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



if __name__ == '__main__':
    import numpy.testing as _nt
    import numpy.linalg as _lalg
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    _test_rotMat2D()
    _test_rotMat3D()
