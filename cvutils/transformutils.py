#-------------------------------------------------------------------------------
# Name:      transformutils.py
# Purpose:   Transformations for computer vision related applications
#
# Author:    Indranil Sinharoy
#
# Created:   25/09/2014
# Copyright: (c) Indranil Sinharoy, 2015
# License:   MIT License
#-------------------------------------------------------------------------------
from __future__ import print_function, division
import numpy as _np
import warnings as _warnings
import scipy.linalg as _linalg
try:
    import cv2 as _cv2
except ImportError:
    _OPENCV = False
else:
    _OPENCV = True


def normalize_2D_pts(p):
    """Function to normalize 2D homogeneous points
    
    This function, which is used for pre-conditioning 2D homogeneous points 
    before solving for homographies and fundamental matrices, translates and
    normalizes the set of points so that their centroid is at the origin, and
    their mean distance from the origin is sqrt(2)
    
    Parameters
    ----------
    p : ndarray
        ``p`` is a `3xN` array of for the set of `N` 2D homogeneous points
        
    Returns
    -------
    newPts : ndarray
        ``newPts`` has the same shape as ``p`` after normalization. Specifically,
        `newPts = np.dot(T, p)`
    T : ndarray
        the 3x3 similarity transformation matrix
    
    Notes
    -----
    1. If there are some points at infinity, the normalization is computed using
       just the finite points. The points at infinity are not affected by scaling
       and translation.
    
    References
    ----------
    1. Multi-view Geometry in Computer Vision, Richard Hartley and Andrew
       Zisserman 
    2. This code has been adapted from Peter Kovesi's MATLAB and Octave Functions
       for Computer Vision and Image Processing available at 
       http://www.csse.uwa.edu.au/~pk/research/matlabfns/       
    """
    eps = _np.spacing(1)
    finiteindex = _np.where(_np.abs(p[-1]) > eps)[0]
    if len(finiteindex) != p.shape[1]:
        _warnings.warn("Some points are at infinity")

    # enforce the scale to be 1 for all finite points    
    p[:, finiteindex] = p[:, finiteindex]/p[-1, finiteindex]
    
    c = _np.mean(p[:2, finiteindex], axis=1)      # centroid of finite points
    pNew = p[:2, finiteindex] - c[:, _np.newaxis] # shift origin to centroid
    dist = _np.sqrt(_np.sum(pNew**2, axis=0)) 
    scale = _np.sqrt(2)/_np.mean(dist)
    T = _np.diag([scale, scale, 1.0])
    T[:2, 2] = -c*scale
    return _np.dot(T, p), T

def get_homography2D(fp, tp, method='DLT', normbyh9=True):
    """Return the homography ``H``, such that ``fp`` is mapped to ``tp`` using 
    normalized DLT described in Algorithm (4.2) of Hartley and Zisserman. 
    
    Parameters
    ----------
    fp : ndarray
        ``fp`` can be a ``2xN`` or ``3xN`` ndarray of "from"-points. If ``fp`` is 
        ``3xN`` the scaling factors ``w_i`` may or may not be 1. i.e the structure 
        of ``fp = _np.array([[x0, x1, ...], [y0, y1, ...], [w0, w1, ...]])``. 
        If ``fp`` is 2xN, then it is assumed that ``w_i = 1`` in homogeneous
        coordinates. i.e. ``fp = _np.array([[x0, x1, ...], [y0, y1, ...]])``
    tp : ndarray
        a ``2xN`` or ``3xN`` ndarray of corresponding "to"-points. If ``tp`` is 
        ``3xN`` the scaling factors ``w_i'`` may or may not be 1. i.e the structure 
        of ``tp = _np.array([[x0', x1', ...], [y0', y1', ...], [w0', w1', ...]])``. 
        If ``tp`` is 2xN, then it is assumed that ``w_i' = 1`` in homogeneous 
        coordinates is 1. i.e. ``tp = _np.array([[x0', x1', ...], [y0', y1', ...]])``
    method : string, optional
        method to compute the 2D homography. Currently only normalized DLT has
        been implemented
    normbyh9 : bool, optional
        if ``True`` (default), the homography matrix ``H`` is normalized by 
        dividing all elements by ``H[-1,-1]``, so that ``H[-1,-1] = 1``. However, 
        this normalization will fail if ``H[-1,-1]`` is very small or zero (if
        the coordinate origin is mapped to a point at infinity by ``H``)
    
    Returns
    -------
    H : ndarray
        the 3x3 homography, ``H`` such that ``tp = np.dot(H, fp)``
       
    References
    ----------
    1. Multi-view Geometry in Computer Vision, Richard Hartley and Andrew
       Zisserman
       
    See Also
    --------
    get_affine2D()
    """
    if fp.shape != tp.shape:
        raise RuntimeError("The point arrays must have the same shape!")
    if (fp.shape[0] < 2) or (fp.shape[0] > 3):
        raise RuntimeError("The length of the input arrays in the first "
                           "dimension must be 3 or 2")
    numCorrespondences = fp.shape[1]
    if fp.shape[0] == 2:
        fp = _np.r_[fp, _np.ones((1, numCorrespondences))]
        tp = _np.r_[tp, _np.ones((1, numCorrespondences))]

    fp, T = normalize_2D_pts(fp)
    tp, Tdash = normalize_2D_pts(tp)

    # create matrix A of size 2*N by 9
    A = _np.zeros((2*numCorrespondences, 9))
    wdash = tp[2,:].tolist()
    for i in range(numCorrespondences):
        x = fp[:,i]
        xdash = tp[:2, i]
        A[2*i:2*(i+1), :] = _np.kron(_np.c_[_np.eye(2)*wdash[i], -xdash], x)
    
    # The solution is the unit singular vector corresponding to the smallest
    # singular value of A
    U, S, Vh = _linalg.svd(A)
    Htilde = Vh[8,:].reshape((3,3))
    
    # Denormalization H = T'^-1 H_tilde T
    H = _np.dot(_linalg.inv(Tdash), _np.dot(Htilde, T))    

    if normbyh9:
        H = H/H[2,2]
    return H

def get_affine2D(fp, tp, normbyha9=True):
    """Return the affine homography ``HA`` as described in Algorithm (4.7) of 
    Hartley and Zisserman.  
    
    Parameters
    ----------
    fp : ndarray
        ``fp`` can be a ``2xN`` or ``3xN`` ndarray of "from"-points. If ``fp`` is 
        ``3xN`` the scaling factors ``w_i`` may or may not be 1. i.e the structure 
        of ``fp = _np.array([[x0, x1, ...], [y0, y1, ...], [w0, w1, ...]])``. 
        If ``fp`` is 2xN, then it is assumed that ``w_i = 1`` in homogeneous
        coordinates. i.e. ``fp = _np.array([[x0, x1, ...], [y0, y1, ...]])``
    tp : ndarray
        a ``2xN`` or ``3xN`` ndarray of corresponding "to"-points. If ``tp`` is 
        ``3xN`` the scaling factors ``w_i'`` may or may not be 1. i.e the structure 
        of ``tp = _np.array([[x0', x1', ...], [y0', y1', ...], [w0', w1', ...]])``. 
        If ``tp`` is 2xN, then it is assumed that ``w_i' = 1`` in homogeneous 
        coordinates is 1. i.e. ``tp = _np.array([[x0', x1', ...], [y0', y1', ...]])``
    normbyha9 : bool, optional
        if ``True`` (default), the affine homography ``HA`` is normalized by 
        dividing all elements by ``HA[-1,-1]``, so that ``HA[-1,-1] = 1``. However, 
        this normalization will fail if ``HA[-1,-1]`` is very small or zero (if
        the coordinate origin is mapped to a point at infinity by ``HA``)
    
    Returns
    -------
    HA : ndarray
        the 3x3 affine homography, ``HA`` such that ``tp = np.dot(HA, fp)``
       
    References
    ----------
    1. Multi-view Geometry in Computer Vision, Richard Hartley and Andrew
       Zisserman
       
    See Also
    --------
    get_homography2D()
    """
    if fp.shape != tp.shape:
        raise RuntimeError("The point arrays must have the same shape!")
    if (fp.shape[0] < 2) or (fp.shape[0] > 3):
        raise RuntimeError("The length of the input arrays in the first "
                           "dimension must be 3 or 2")
    numCorrespondences = fp.shape[1]
    if fp.shape[0] == 2:
        fp = _np.r_[fp, _np.ones((1, numCorrespondences))]
        tp = _np.r_[tp, _np.ones((1, numCorrespondences))]

    fp, T = normalize_2D_pts(fp)
    tp, Tdash = normalize_2D_pts(tp)
    
    # the translation is zero as conditioned points have zero mean   
    A = _np.vstack((fp[:2], tp[:2]))
    U, S, Vh = _linalg.svd(A.T)
    B = Vh[:2, :2].T
    C = Vh[:2, 2:4].T
    zt = [[0.0], 
          [0.0]]
    H2x2 = _np.dot(C, _np.linalg.pinv(B))
    H = _np.vstack(( _np.hstack((   H2x2  ,   zt  )), 
                                 [0.0, 0.0,  1.0] ))
    # Denormalization H = T'^-1 H_tilde T   
    HA = _np.dot(_linalg.inv(Tdash), _np.dot(H, T))
    
    if normbyha9:
        HA = HA/HA[2,2]
    
    return HA

# ##########################################
# test functions to test the module methods
# ##########################################

def _test_get_homography2D():
    # Test homography estimation using the DLT method
    # create a random homography matrix
    randHomo = _np.random.rand(3,3)
    randHomo[:2, 2] = 100.0*_np.random.random(2)
    randHomo[2,2] = 1.0
    # create a set of "from" points
    numPts = 6
    fp = _np.random.randint(low=-50, high=50, size=(2, numPts))
    fp = _np.r_[fp, _np.ones((1, numPts))]
    tp = _np.dot(randHomo, fp) 
    # send fp and tp as 3xN arrays     
    homoEst = get_homography2D(fp, tp, method='DLT')
    _nt.assert_array_almost_equal(randHomo, homoEst)    
    
    #  Compare with OpenCV's function
    if _OPENCV:
        cvHomoEst, _ = _cv2.findHomography((fp[:2].T).astype(_np.float32), 
                                         (tp[:2].T).astype(_np.float32), method=0)
        # becoz we are using float32 (and not float64), we may not always get
        # a precision of above 5 decimal places when comparing the two matrices
        # therefore we use "decimal=4"                                        
        _nt.assert_array_almost_equal(homoEst, cvHomoEst, decimal=4)

    # When passing 2xN arrays, ensure that the points are homogenized
    homoEst = get_homography2D(fp[:2]/fp[-1], tp[:2]/tp[-1], method='DLT')
    _nt.assert_array_almost_equal(randHomo, homoEst)
    print("get_homography2D() test successful")
    
def _test_get_affine2D():
    # Test affine transform estimation function get_affine2D()
    # create a random affine transform matrix
    randHA2by2 = _np.random.rand(2, 2)
    randTxy = 100.0*_np.random.random(2).reshape(2,1)
    randAffine = _np.vstack((_np.hstack((randHA2by2, randTxy)),
                             _np.array([0.0, 0.0, 1.0])))
    # create a set of "from" points
    numPts = 6
    fp = _np.random.randint(low=-50, high=50, size=(2, numPts))
    fp = _np.r_[fp, _np.ones((1, numPts))]
    tp = _np.dot(randAffine, fp)
    affineEst = get_affine2D(fp, tp)
    _nt.assert_array_almost_equal(randAffine, affineEst) 
    
    # using OpenCV's getAffineTransform() function that accepts points as 2x3
    # ndarrays. i.e. it uses 3 correspondences between the "to" and "from" 
    # images. Therefore we shall use only the first 3 points from both "to" and
    # "from" set of points. Also, the dtype of the arrays is expected to be 
    # float32. The function returns a 2x3 matrix that represents the affine 
    # transformation
    if _OPENCV:
        cvAffineEst = _cv2.getAffineTransform((fp[:2, :3].T).astype(_np.float32), 
                                            (tp[:2, :3].T).astype(_np.float32))
        _nt.assert_array_almost_equal(affineEst[:2], cvAffineEst, decimal=4)
    print("get_affine2D() test successful")
                                     

if __name__ == '__main__':
    import numpy.testing as _nt 
    from numpy import set_printoptions
    
    set_printoptions(precision=5, suppress=False)
    # test functions
    _test_get_homography2D()
    _test_get_affine2D()
    

