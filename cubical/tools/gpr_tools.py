# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details


"""
13 July 2018

@author: landman

A collection of tools for performing Gaussian process regression

"""
import numpy as np
from cubical.tools import kronecker_tools as kt
from cubical.tools import linear_algebra_tools as lat


def abs_diff(x,xp):
    """
    Gets vectorised differences between x and xp
    :param x: NxD array of floats (inputs1)
    :param xp: NpxD array of floats (inputs2)
    """
    try:
        N, D = x.shape
        Np, D = xp.shape
    except:
        N = x.size
        D = 1
        Np = xp.size
        x = np.reshape(x, (N, D))
        xp = np.reshape(xp, (Np, D))
    xD = np.zeros([D, N, Np])
    xpD = np.zeros([D, N, Np])
    for i in xrange(D):
        xD[i] = np.tile(x[:, i], (Np, 1)).T
        xpD[i] = np.tile(xp[:, i], (N, 1))
    return np.linalg.norm(xD - xpD, axis=0)


def smp_kernel(x, ws, sigmas, mus):
    """
    This is the 1D spectral mixture product kernel with A=len(ws) components. The more components we have the 
    more expressive this kernel becomes. Can be extended to higher dimensions by assuming product structure.
    The hyper-parameters are easier to interpret in Fourier space. See discussion in spectral density below.
    :param x: length N array of inputs at which to evaluate kernel
    :param ws: length A array specifying power of components 
    :param sigmas: length A array specifying widths of components
    :param mus: length A array specifying the means of components
    :return: 
    """
    N = np.size(x)
    A = np.size(ws)
    # get matrix of differences
    rr = np.tile(x, (N, 1)).T - np.tile(x, (N, 1))
    # evaluate covariance matrix iteratively
    K = np.zeros([N, N], dtype=np.float64)
    for i in xrange(A):
        K += ws[i]**2*np.exp(-2.0*np.pi*rr**2*sigmas[i]**2)*np.cos(2.0*np.pi*rr*mus[i])
    return K


def smp_spectral_density(s, ws, sigmas, mus):
    """
    This is the spectral density of the 1D spectral mixture product kernel with A=len(ws) components. The spectral
    density is parametrised as a scale-location mixture of Gaussians as in https://arxiv.org/abs/1302.4245
    :param s: frequencies at which to evaluate spectral density
    :param ws: length A array specifying power of components 
    :param sigmas: length A array specifying widths of components
    :param mus: length A array specifying the means of components 
    :return: 
    """
    def gaussian_pdf(x, mu, sigma):
        return np.exp(-(x-mu)**2/(2*sigma**2))/np.sqrt(2.0*np.pi*sigma**2)
    N = np.size(s)
    A = np.size(ws)
    # iteratively evaluate spectral density
    S = np.zeros(N)
    for i in xrange(A):
        S += ws[i]**2*(gaussian_pdf(s, mus[i], sigmas[i]) + gaussian_pdf(-s, mus[i], sigmas[i]))
    return S


def rect_laplacian_eigenvector(x, j, L):
    """
    Evaluates eigen-functions of the Laplacian on 1D rectangular domain with Dirichlet boundary conditions. This 
    assumes that the function goes to zero at x = [-L,L]. 
    LB - Note it should be possible to generalise the boundary conditions
    :param x: length N vector of coordinates
    :param j: integer specifying the order of the basis function
    :param L: float specifying boundary of domain
    """
    return np.sin(j*np.pi*(x + L)/(2*L))/np.sqrt(L)


def rect_laplacian_eigenvalue(j, L):
    """
    Evaluates eigen-values of the Laplacian on 1D rectangular domain with Dirichlet boundary conditions. This 
    assumes that the function goes to zero at x = [-L,L]
    :param j: Dx1 array of integers (order of the basis function in the summation)
    :param L: Dx1 array of floats (boundary of domain)
    """
    return (j*np.pi/(2*L))**2


def reduced_rank_inverse_times(x, Sigmayinv, Phi, Sinv):
    """
    Assuming that our covariance matrix is low rank so that Ky = Phi Lambda Phi.T + Sigmay with Lambda only containing
    a few large eiegenvectors we can create an operator which approximates its inverse_times efficiently using the 
    Woodbury matrix identity.  
    :param x: 
    :param Sigmay: diagonal of Cramer-Rao bound
    :param Phi: kronecker matrix containing Phi_t and Phi_nu
    :param S: kronecker vector containing S_t and S_nu
    :return: 
    """
    # expand kronecker representation of diagonal S
    Sinv = kt.kron_kron(Sinv)

    rhs_vec1 = Sigmayinv[:, None] * x
    Z_op = lambda x2: Sinv[:, None] * x2 + kt.kron_tensorvec(kt.kron_transpose(Phi), Sigmayinv * kt.kron_tensorvec(Phi, x2))
    x0 = np.ones(x.size)
    rhs_vec2 = lat.pcg(x0, Z_op, rhs_vec1)
    rhs_vec2 *= Sigmayinv[:, None]
    return rhs_vec1 - rhs_vec2


def draw_time_frequency_samples(meanf, t, nu, ws_t, ws_nu, sigmas_t, sigmas_nu, mus_t, mus_nu, Nsamps):
    """
    Draws a 2D sample on a tnu grid from Gaussian process with smp kernel
    :param meanf: 
    :param t: 
    :param nu: 
    :param ws_t: 
    :param ws_nu: 
    :param sigmas_t: 
    :param sigmas_nu: 
    :param mus_t: 
    :param mus_nu: 
    :param Nsamps: 
    :return: 
    """
    Nt = t.size
    Nnu = nu.size  # assumes 1D
    Ntot = Nt*Nnu
    Knu = smp_kernel(nu, ws_nu, sigmas_nu, mus_nu) + 1e-13*np.eye(Nnu)  # jitter for numerical stability
    Kt = smp_kernel(t, ws_t, sigmas_t, mus_t) + 1e-13*np.eye(Nnu)  # jitter for numerical stability
    K = np.array([Kt, Knu])
    L = kt.kron_cholesky(K)
    samps = np.zeros([Nsamps, Nt, Nnu])
    for i in xrange(Nsamps):
        xi = np.random.randn(Ntot)
        samps[i] = meanf(t, nu) + kt.kron_matvec(L, xi).reshape(Nt, Nnu)
    return samps
