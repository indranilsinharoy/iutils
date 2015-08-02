# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------------------
# Name:          geometry.py
# Purpose:       Geometry related Utilities
#
# Author:        Indranil Sinharoy
#
# Created:       06/30/2015
# Last Modified:
# Copyright:     (c) Indranil Sinharoy 2015
# License:       MIT License
#-----------------------------------------------------------------------------------------
"""Utility functions for geometry. All examples in docstrings assume that the module has 
   been imported as ``geo``
"""

from __future__ import division, print_function
import numpy as _np

class Point(object):
    """Point class represent a point in 3d space"""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z 

    def __repr__(self):
        return 'Point({}, {}, {})'.format(self.x, self.y, self.z)

    def __str__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.z)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_dist_from_origin(self):
        '''returns the Euclidean distance from origin'''
        return _np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def get_dist_from_point(self, other):
        '''returns the Euclidean distance from another point'''
        s = self - other
        return _np.sqrt(s.x**2 + s.y**2 + s.z**2)




#%% Test functions
def test_Point_class():
    p = Point()
    _nt.assert_array_equal((0.0, 0.0, 0.0), (p.x, p.y, p.z))
    p.x, p.y, p.z = 2, 3, 4
    _nt.assert_array_equal((2.0, 3.0, 4.0), (p.x, p.y, p.z))
    q = Point(10, 20, 30)
    s = p + q
    assert isinstance(s, Point)
    _nt.assert_array_equal((p.x + q.x, p.y + q.y, p.z + q.z), (s.x, s.y, s.z))
    # Test methods
    po = p.get_dist_from_origin()
    _nt.assert_almost_equal(po, 5.38516480713)
    pq = p.get_dist_from_point(q)
    qp = q.get_dist_from_point(p)
    assert pq == qp
    _nt.assert_almost_equal(pq, 32.0780298647)
    # Test equality 
    p = Point(3, 4, 5)
    q = Point(3, 4, 5)
    assert p is not q  # p and q are different objects
    assert p == q      # p and q are the same points
    l = Point()
    assert p != l
    print('Point class test successful')



if __name__ == '__main__':
    import numpy.testing as _nt
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    # test functions
    test_Point_class()
