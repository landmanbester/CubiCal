import numpy as np
from numba import njit
import dask.array as da
from africanus.gps.kernels import exponential_squared as expsq
from africanus.linalg import kronecker_tools as kt
from africanus.linalg.pcg import pcg

def _interp_tnu(data, Sigma, Mask, Ky, M, Kp, tol, maxit):
    ndir, ntime, nfreq, nant, nc1, nc2 = y.shape
    nptime = Kp[0].shape[1]
    npfreq = Kp[1].shape[1]
    fp = np.zeros(ndir, nptime, npfreq, nant, nc1, nc2)
    for d in range(ndir):
        for ant in range(nant):
            for c1 in range(nc1):
                for c2 in range(nc2):
                    y = y[d, :, :, ant, c1, c2].flatten()
                    dy = Sigma[d, :, :, ant, c1, c2].flatten()
                    mask_ind = np.argwhere(Mask[d, :, :, ant, c1, c2].flatten())
                    unmask_ind = np.argwhere(~Mask[d, :, :, ant, c1, c2].flatten())

                    mbar = np.mean(y[unmask_ind])
                    x0 = np.zeros_like(y, dtype=y.dtype)
                    tmp = pcg(Ky, y - mbar, x0, M=M, mask_ind=mask_ind, tol=tol, maxit=maxit, verbose=0)
                    fp[d, :, :, ant, c1, c2] = mbar + kt.kron_tensorvec(Kp, tmp).reshape(nptime, npfreq)

    return fp


def interp(x, xp, y, Sigma):
    """
    Interpolate y = f(x) + eps  where x ~ N(0, Sigma) onto locations xp.

    x       - object array holding domain at which data is defined
    xp      - object array holding domain onto which we want to interpolate 
    y       - data values as masked array of shape (dir, time, freq, ant, corr1, corr2)
    Sigma   - Diagonal entries of covariance matrix, same shape as y

    It is assumed that the data are provided on a (possibly incomplete) grid
    defined by the outer product of the entries of x (and similarly for xp). 
    """
    if len(x) > 2:
        raise ValueError("Only interpolation over time and frequency currently supported")

    assert (y.shape == Sigma.shape).all()

    ndir, ntime, nfreq, nant, nc1, nc2 = y.shape  # is this always the case?

    # normalised time
    t = x[0]/np.mean(x[0])
    tp = xp[0]/np.mean(x[0])
    assert t.size == ntime

    # normalised freq
    nu = x[1]/np.mean(x[1])
    nup = xp[1]/np.mean(x[1])
    assert nu.size == nfreq

    # set time and freq covariance matrices
    tmean = np.mean(y, axis=(0, 2, 3, 4, 5))
    sigmat = (tmean.max() - tmean.min())/2.0
    lt = (t.max() - t.min())/4.0
    Kt = expsq(t, t, sigmat, lt)
    Kpt = expsq(t, tp, sigmat, lt)

    fmean = np.mean(y, axis=(0, 1, 3, 4, 5))
    sigmaf = (fmean.max() - fmean.min())/2.0
    lf = (nu.max() - nu.min())/4.0
    Kf = expsq(nu, nu, sigmaf, lf)
    Kpf = expsq(nu, nup, sigmaf, lf)

    # Kronecker matrices
    K = (Kt, Kf)
    Kp = (Kpt, Kpf)

    # construct masked operators
    @njit(fastmath=True, inline='always')
    def Ky(x):
        tmp = kt.kron_matvec(K, x) + sigma*x
        tmp[mask] = 0.0
        return tmp
    @njit(fastmath=True, inline='always')
    def Mop(x):
        tmp = kt.kron_matvec(K, x)
        tmp[mask] = 0.0
        return tmp


    fp = da.blockwise(_interp_tnu, ('dir', 'time', 'freq', 'ant', 'c1', 'c2'),
                      )



