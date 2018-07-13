# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details

"""
15 Nov 2017

@author: landman

A collection of tools for matrices defined by multiple kronecker products (kronecker matrix for short) i.e.

A = K1 kron K2 kron K3 kron ...

These are useful for performing GPR when data are given on a (not necessarily uniform) Cartesian grid and when the 
covariance function is a tensor product kernel i.e. when it can be written as the product of 1D covariance functions.

"""

import numpy as np

def kron_N(x):
    """
    Computes N = N_1 x N_2 x ... x N_D i.e. the total number of points in x 
    :param x: an array of arrays holding matrices/vectors [x1, x2, ..., xD]
    :return: 
    """
    D = x.shape[0]
    dims = np.zeros(D)
    for i in xrange(D):
        dims[i] = x[i].shape[0]
    N = int(np.prod(dims))
    return N

def kron_collapse_shape(x):
    """
    Returns the dimensions of data from an array of diff_squares
    :param x: an array of arrays holding diff_squares i.e. x = [XX_1, XX_2, ..., XX_D]
    :return: 
    """
    D = x.shape[0]
    y = []
    for i in xrange(D):
        y.append(np.ones(x[i].shape[0]))
    return np.asarray(y)

def kron_kron(A):
    """
    Computes the Kronecker product of a series of matrices or vectors 
    :param A: an array of arrays holding matrices/vectors [A1, A2, ..., AD]
    :return: the Kronecker product of all elements of A
    """
    D = A.shape[0]
    # expand the kronecker product for A
    K = np.kron(A[0], A[1])
    if D > 2:
        for i in xrange(2, D):
            K = np.kron(K, A[i])
    return K

def kron_transpose(A):
    """
    Transposes the elements of a kronecker matrix
    :param A: an array of arrays holding matrices/vectors [A1, A2, ..., AD]
    :return: an array of arrays holding matrices/vectors [A1.T, A2.T, ..., AD.T]
    """
    D = A.shape[0]
    AT = np.empty((D), dtype=object)
    for i in xrange(D):
        AT[i] = A[i].T
    return AT

def kron_tensorvec(A, b):
    """
    Tensor product over non-square Knonecker matrices
    :param A: an array of arrays holding matrices [..., K3, K2, K1] where Ki is Mi x Gi
    :param b: the RHS vector of length prod(G1, G2, ..., GD)
    :return: the solution vector alpha = Ab of length prod(M1, M2, ..., MD)
    """
    D = A.shape[0]
    # get shape of sub-matrices
    G = np.zeros(D, dtype=np.int8)
    M = np.zeros(D, dtype=np.int8)
    for d in xrange(D):
        M[d], G[d] = A[d].shape
    x = b
    for d in xrange(D):
        Gd = G[d]
        rem = np.prod(np.delete(G, d))
        X = np.reshape(x, (Gd, rem))
        Z = np.einsum("ab,bc->ac", A[d], X)
        Z = np.einsum("ab -> ba", Z)
        x = Z.flatten()
        # replace with new dimension
        G[d] = M[d]
    return x

def kron_matvec(A, b, Flags=None, return_valid=True):
    """
    Computes matrix vector product of kronecker matrix in linear time. This is useful for data on a possibly
    incomplete Euclidean grid. Incomplete data should be flagged by setting Flags to True at the relevant locations.
    LB - Note it should be possible to do all of this in one single step because it is basically just a contraction over
    some inner indices. The problem is I can't figure out how to arrange the RHS vector for this to work. 
    :param A: an array of arrays holding matrices [..., K3, K2, K1] (note ordering)
    :param b: the RHS vector
    :param Flags: specifies locations of missing data 
    :return: A.dot(b)
    """
    D = A.shape[0]
    N = b.size
    if Flags is not None:
        x = np.where(Flags, 0.0, b)
    for d in xrange(D):
        Gd = A[d].shape[0]
        X = np.reshape(x,(Gd, N//Gd))
        Z = np.einsum("ab,bc->ca", A[d], X)  # matrix multiply and transpose
        x = Z.flatten()
    if return_valid:
        if Flags is not None:
            I = np.argwhere(~Flags).squeeze()
        else:
            I = slice(None)
        return x[I]
    else:
        return x

def kron_matmat(A, B):  # double check!!!!!!
    """
    Computes the product of two kronecker matrices
    :param A: 
    :param B: 
    :return: 
    """
    K = kron_kron(B)  # currently unavoidable
    M = K.shape[1]  # the product of Np_1 x Np_2 x ... x Np_3

    N = kron_N(A)
    C = np.zeros([N, M])
    for i in xrange(M):
        C[:,i] = kron_matvec(A, K[:, i])
    return C

def kron_tensormat(A, B):
    """
    Tensor product over non-square Knonecker matrices
    :param A: an array of arrays holding matrices [..., K3, K2, K1] where Ki is Mi x Gi
    :param B: the RHS matrix
    :return: the solution matrix
    """
    K = kron_kron(B)  # currently unavoidable
    M = K.shape[1]  # the product of Np_1 x Np_2 x ... x Np_3

    N = kron_N(A)
    C = np.zeros([N, M])
    for i in xrange(M):
        C[:,i] = kron_tensorvec(A, K[:, i])
    return C

def kron_trace(A):
    """
    Computes the trace of a kronecker matrix
    :param A: an array of arrays holding matrices [K1, K2, K3, ...]
    :return: trace(A)
    """
    D = A.shape[0]
    x = 1.0
    for i in xrange(D):
        x *= np.trace(A[i])
    return x

def kron_cholesky(A):
    """
    Computes the cholesky decomposition of a kronecker matrix
    :param A: an array of arrays holding matrices [K1, K2, K3, ...]
    :return: 
    """
    D = A.shape[0]
    L = np.zeros_like(A)
    for i in xrange(D):
        try:
            L[i] = np.linalg.cholesky(A[i])
        except: # add jitter
            L[i] = np.linalg.cholesky(A[i] + 1e-13*np.eye(A[i].shape[0]))
    return L

def kron_logdet(L):
    """
    Computes the determinant of kronecker matrix 
    :param L: an array of arrays holding matrices [L1, L2, L3, ...]
    :return: 
    """
    D = L.shape[0]
    # first get dimensions and determinants of individual matrices
    dims = np.zeros(D)
    dets = np.zeros(D)
    for i in xrange(D):
        dims[i] = L[i].shape[0]
        dets[i] = 2.0*np.sum(np.log(np.diag(L[i])))

    # now compute combined determinant
    x = 0.0
    n_prod = np.prod(dims)
    for i in xrange(D):
        x += n_prod*dets[i]/dims[i]

    return x

def kron_diag(A):
    """
    Computes the kronecker product of a series of diagonal matrices defined as vectors
    :param A: an array of arrays holding vectors [x1, x2, x3, ...] where each vector is the diagonal of a matrix
    :return: the diagonal matrix of the kronecker product
    """
    D = A.shape[0]
    Adiag = []
    for i in xrange(D):
        Adiag.append(A[i])
    Adiag = np.asarray(Adiag)
    return kron_kron(Adiag)

def kron_diag_diag(A):
    """
    Computes the kronecker product of a series of diagonal matrices defined as vectors
    :param A: an array of arrays holding vectors [x1, x2, x3, ...] where each vector is the diagonal of a matrix
    :return: the diagonal matrix of the kronecker product
    """
    D = A.shape[0]
    Adiag = []
    for i in xrange(D):
        Adiag.append(np.diag(A[i]))
    Adiag = np.asarray(Adiag)
    return kron_kron(Adiag)

def kron_inverse(A):
    """
    Computes the inverse of a kronecker matrix
    :param A: an array of arrays holding matrices [K1, K2, K3, ...]
    :return: 
    """
    D = A.shape[0]
    Ainv = np.zeros_like(A)
    for i in xrange(D):
        Ainv[i] = np.linalg.inv(A[i])
    return Ainv

def kron_eig(A, dtype='real'):
    """
    Computes the eigendecomposition of a kronecker matrix 
    :param A: an array of arrays holding matrices [K1, K2, K3, ...]
    :return: Eigenvectors Q and eigenvalues Lambda of a all submatrices
    """
    D = A.shape[0]
    # dims = np.zeros(D)
    # for i in xrange(D):
    #     dims[i] = A[i].shape[0]

    Qs = np.zeros_like(A)
    Lambdas = []

    if dtype=='real':
        for i in xrange(D):
            Lambda, Q = np.linalg.eigh(A[i])
            Qs[i] = Q
            Lambdas.append(Lambda)
        Lambdas = np.asarray(Lambdas)
    elif dtype=='complex':
        for i in xrange(D):
            Lambda, Q = np.linalg.eig(A[i])
            Qs[i] = Q
            Lambdas.append(Lambda)
        Lambdas = np.asarray(Lambdas)
    return Lambdas, Qs