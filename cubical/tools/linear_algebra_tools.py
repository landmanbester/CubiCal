# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details

"""
13 July 2018

@author: landman

A collection of linear algebra subroutines specialised for dealing with data on incomplete grids

"""

import numpy as np

def pcg(x0, A, b, Flags=None, M=None, tol=1e-6, maxiter=None, return_valid=True):
    """
    This is a pre-conditioned conjugate gradient method for data on a possibly incomplete grid. 
    Incomplete data should be flagged by setting Flags to True at the relevant locations.
    :param x0: length N vector, initial guess for solution to Ax=b
    :param A: A can be any object which implements a matrix vector product via A()
    :param b: length N data vector in Ax=b
    :param Flags: specifies locations of missing data 
    :param M: Pre-conditioner which approximates the inverse of A i.e. x approx M(b)
    :param tol: tolerance (default is 1e-6)
    :param maxiter: maximum number of iterations (default is same size as x0)
    :param return_valid: if True will return the solution only at locations where we have valid data
    :return: the approximate solution to Ax=b
    """
    if Flags is not None:
        I = np.argwhere(Flags).squeeze()
    else:
        I = []
    x = x0
    x[I] = 0.0
    b[I] = 0.0
    k = 0
    r = b - A(x)
    r[I] = 0.0
    if M is not None:
        z = M(r)
        z[I] = 0.0
    else:
        z = r
    p = z
    r_norm = r.dot(z)
    b_norm = b.dot(b)
    if maxiter is None or maxiter > x0.size:
        maxiter = x0.size
    while np.sqrt(r_norm) > tol*b_norm and k < maxiter:
        k += 1
        Ap = A(p)
        Ap[I] = 0.0
        alpha = r_norm/(p.dot(Ap))
        x += alpha*p
        x[I] = 0.0
        r -= alpha*Ap
        r[I] = 0.0
        if M is not None:
            z = M(r)
            z[I] = 0.0
        else:
            z = r
        r_norm_next = r.dot(z)
        beta = r_norm_next/r_norm
        r_norm = r_norm_next.copy()
        p = z + beta*p
        p[I] = 0.0
    print "Iters = ", k
    if return_valid:
        if Flags is not None:
            I2 = np.argwhere(~Flags).squeeze()
        else:
            I2 = slice(None)
        return x[I2]
    else:
        return x