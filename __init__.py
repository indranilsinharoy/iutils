# iutils init file


# Try to import numba if available.
_hasNumba = False
try:
    from numba import double
except ImportError:
    pass
else:
    #from numba import jit
    _hasNumba = True
