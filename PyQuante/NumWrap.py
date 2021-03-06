"""
 NumWrap.py - Interface to Numeric and numpy

 An interface to the Numeric and numpy libraries that will (hopefully)
 make the transformation to numpy go as seamlessly as possible.

 Also interfaces the LinearAlgebra and numpy.linalg libraries, since
 these have to be done consistently with the Numeric/numpy choice
"""

# Todo
# - Migrate to numpy.linalg names
# - Change matrixmultiply to "matmul"
# - Remove NumWrap and Numeric support (maybe never do this)
#

use_numpy = True
import re
pat = re.compile('\D')

# As of 2/12/2007 we have updated PyQuante to the numpy convention,
# where eigenvectors are kept in columns.
if use_numpy:
    from numpy import array2string #added by Hatem Helal hhh23@cam.ac.uk
    from math import fabs #added by Hatem Helal hhh23@cam.ac.uk
    from numpy import array,zeros,concatenate,dot,ravel,arange
    from numpy import arcsinh,diagonal,identity,choose,transpose
    from numpy import reshape,take,trace
    from numpy import where,argsort
    from numpy import sum
    matrixmultiply = dot
    matmul = dot

    from numpy.linalg import det
    from numpy.linalg import eigh
    from numpy.linalg import solve
    from numpy.linalg import inv
    from numpy.linalg import cholesky
    
    # still need to kill these two, which are used by Optimize:
    import numpy.oldnumeric.mlab as MLab
    from numpy.oldnumeric import NewAxis
    import numpy as Numeric
else:
    from Numeric import array2string #added by Hatem Helal hhh23@cam.ac.uk
    from math import fabs #added by Hatem Helal hhh23@cam.ac.uk
    from Numeric import array,zeros,concatenate,dot,ravel,arange
    from Numeric import matrixmultiply
    from Numeric import arcsinh,diagonal,identity,choose,transpose
    from Numeric import reshape,take,trace
    from Numeric import where,argsort
    from Numeric import NewAxis
    from LinearAlgebra import Heigenvectors
    from LinearAlgebra import determinant as det
    from LinearAlgebra import solve_linear_equations as solve
    from LinearAlgebra import inverse as inv
    from LinearAlgebra import cholesky_decomposition as cholesky
    import Numeric
    import MLab
    matmul = matrixmultiply
    from Numeric import sum

    def eigh(A):
        val,vec = Heigenvectors(A)
        return val,transpose(vec)

