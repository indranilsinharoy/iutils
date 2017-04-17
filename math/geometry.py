# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------------------
# Name:          geometry.py
# Purpose:       Geometry related Utilities
#
# Author:        Indranil Sinharoy
#
# Created:       06/30/2015
# Last Modified: 04/08/2017
# Copyright:     (c) Indranil Sinharoy 2015 - 2017
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
        return '{0.__name__}({1.x!r}, {1.y!r}, {1.z!r})'.format(type(self), self)

    def __str__(self):
        return 'Point({0.x!r}, {0.y!r}, {0.z!r})'.format(self)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def dist_from_origin(self):
        """returns the Euclidean distance from origin"""
        return _np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def get_dist_from_point(self, other):
        """returns the Euclidean distance from another point"""
        s = self - other
        return _np.sqrt(s.x**2 + s.y**2 + s.z**2)


class Vector:
    """Three dimensional vector class"""

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return '{0.__name__}({1.x!r}, {1.y!r}, {1.z!r})'.format(type(self), self)
        
    def __add__(self, other):
        """vector addition"""
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        """vector substraction"""
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __abs__(self):
        """absolute value"""
        return (self.x **2 + self.y **2 + self.z **2) **0.5
    
    def __mul__(self, scalar):
        """scalar multiplication"""
        return Vector(scalar*self.x, scalar*self.y, scalar*self.z)
    
    def __rmul__(self, scalar):
        return Vector(scalar*self.x, scalar*self.y, scalar*self.z)
    
    def __bool__(self):
        """returns true if vector magnitude is not zero"""
        return bool(self.x or self.y or self.z)
    
    def __getitem__(self, position):
        """to support indexing and functions such as list(Vector(1, 2, 3))"""
        return list((self.x, self.y, self.z))[position]


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
    po = p.dist_from_origin
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

def test_Vector_class():
    vec = Vector()
    _nt.assert_array_equal((0.0, 0.0, 0.0), (vec.x, vec.y, vec.z))
    _nt.assert_equal(True, isinstance(vec, Vector), 'Vector instance test failed')
    vec1 = Vector(1, 2, 3)
    _nt.assert_equal(True, isinstance(list(vec1), list), 'List instance test failed')
    _nt.assert_equal(2, vec1[1], 'Vector indexing test failed')
    vec2 = Vector(4, 5, 6)
    _nt.assert_array_equal(list((5, 7, 9)), list(vec1 + vec2), 'Vector addition test failed')
    _nt.assert_array_equal(list((-3, -3, -3)), list(vec1 - vec2), 'Vector substraction test failed')
    _nt.assert_array_equal(list((2, 4, 6)), list(2*vec1), 'Vector scalar pre-multiplication test failed')
    _nt.assert_array_equal(list((2, 4, 6)), list(vec1*2), 'Vector scalar post-multiplication test failed')
    vec3 = Vector(-1, 2, -3)
    _nt.assert_array_equal(_np.sqrt(vec3[0]**2 + vec3[1]**2 + vec3[2]**2), abs(vec3), 'Vector absolute test failed')
    _nt.assert_equal(True, bool(vec1), 'Vector bool test 1 failed')
    _nt.assert_equal(False, bool(vec), 'Vector bool test 2 failed')
    print('Vector class test successful')


if __name__ == '__main__':
    import numpy.testing as _nt
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    # test functions
    test_Point_class()
    test_Vector_class()