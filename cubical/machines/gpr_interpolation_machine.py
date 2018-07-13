# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details

import numpy as np
from cubical import param_db
from cubical.tools import logger
from cubical.tools import gpr_tools as gpt
from cubical.tools import kronecker_tools as kt
from cubical.tools import linear_algebra_tools as lat
log = logger.getLogger("interp_machine")

class GPRInterpolationMachine(object):
    """
    Class to implement gpr interpolation of gain . These machines should implement and interface
    to a gain table/database.
    """
    def __init__(self, db_name, out_name='smoothed_gains'):
        # get basename
        last_slash_loc = [pos for pos, char in enumerate(db_name) if char == '/'][-1]
        self.base_name = db_name[0:last_slash_loc+1]

        # load in database
        self.db = param_db.load(db_name)

        # set location to write
        self.out_name = out_name

        #### for testing
        # set domain
        self.Nt = 100
        self.t = np.linspace(-1, 1, self.Nt)
        self.Nnu = 50
        self.nu = np.linspace(-1, 1, self.Nnu)
        # set number of mixture components for kernels
        self.A = 3
        self.ws_t = 0.1 + np.random.random(self.A)
        self.ws_nu = 0.1 + np.random.random(self.A)
        self.sigmas_t = 0.1 + np.random.random(self.A)
        self.sigmas_nu = 0.1 + np.random.random(self.A)
        self.mus_t = 0.1 + np.random.random(self.A)
        self.mus_nu = 0.1 + np.random.random(self.A)

        # draw some sample gain realisations
        self.Na = 1
        self.meanf = lambda t, nu: np.ones([np.size(t), np.size(nu)])
        self.gains = gpt.draw_time_frequency_samples(self.meanf, self.t, self.nu, self.ws_t, self.ws_nu, self.sigmas_t,
                                                     self.sigmas_nu, self.mus_t, self.mus_nu, self.Na)

        # set some flags
        flagtion = 0.25  # fraction of flagged data
        self.Flags = np.zeros([self.Na, self.Nt, self.Nnu], dtype=np.bool)
        for i in xrange(self.Na):
            I = np.np.unique(np.random.randint(0, self.Nt*self.Nnu, flagtion*self.Nt*self.Nnu))
            F = np.zeros(self.Nt*self.Nnu, dtype=np.bool)
            F[I] = 1
            self.Flags[i] = F.reshape(self.Nt, self.Nnu)

        # Simulate covariance matrix (the iid case is trivial and not realistic)
        self.Sigmays = np.zeros([self.Na, self.Nt, self.Nnu])
        for i in xrange(self.Na):
            self.Sigmays[i] = 0.1 + 0.05 * np.abs(np.random.randn(self.Nt, self.Nnu))

        # Add noise realisation to gains
        for i in xrange(self.Na):
            self.gains[i] = np.random.randn(self.Nt, self.Nnu) * np.sqrt(self.Sigmays[i])

    def smooth(self, antennas='all', t_out='Full', nu_out='Full', save_err=False):
        """
        This is the method that does the smoothing. 
        :param antennas: a list of antennas whose solutions should be smoothed. Default is to smooth all of them
        :param t_out: 1D array of times at which we want the smoothed solutions. Default is to fill the the grid
        :param nu_out: 1D array of frequencies at which we want the smoothed solutions. Default is to fill the the grid
        :param save_err: if true the approximate uncertainty in the gains will be saved to the same database as g_n_cov
        :return: writes smoothed solutions to .npz file next to where the database is kept
        """
        if antennas=='all':
            antennas = np.arange(self.Na)

        if t_out=='Full':
            t_out = self.t

        if nu_out=='Full':
            nu_out = self.nu

        # set basis funcs for pre-conditioner
        M = 12
        L = 2.0
        Phi_t = np.zeros([self.Nt, M])
        Phi_nu = np.zeros([self.Nnu, M])
        s_t = np.zeros(M)
        s_nu = np.zeros(M)
        for i in xrange(M):
            Phi_t[:, i] = gpt.rect_laplacian_eigenvector(self.t, i+1, L)
            s_t[i] = np.sqrt(gpt.rect_laplacian_eigenvalue(i+1, L))
            Phi_nu[:, i] = gpt.rect_laplacian_eigenvector(self.nu, i + 1, L)
            s_nu[i] = np.sqrt(gpt.rect_laplacian_eigenvalue(i + 1, L))

        # need to evaluate inside loop if hyper-parameters are different for different antennas
        S_t = gpt.smp_spectral_density(s_t, self.ws_t, self.sigmas_t, self.mus_t)
        S_nu = gpt.smp_spectral_density(s_nu, self.ws_nu, self.sigmas_nu, self.mus_nu)

        # sequential GPR
        for ant in antennas:
            # get prior covariance matrix for antenna
            Kt = gpt.smp_kernel(t_out, self.ws_t, self.sigmas_t, self.mus_t)
            Knu = gpt.smp_kernel(nu_out, self.ws_nu, self.sigmas_nu, self.mus_nu)
            K = np.array([Knu, Kt], dtype=object)
            # get Cramer_Rao bound for antenna
            Sigma = self.Sigmays[ant].flatten()
            # set the Ky operator
            Kyop = lambda x: kt.kron_matvec(K, x, self.Flags[ant], return_valid=False) + Sigma[:, None]*x
            # set preconditioner operator
            M_op = lambda x: 1
            # solve Kyinv.dot(y)
            y = self.gains[ant].flatten() - self.meanf.flatten()
            x0 = np.ones(self.Nt*self.Nnu)
            Kyinvy = lat.pcg(x0, Kyop, y, )
            post_mean = kt.kron_matvec(K, )



if __name__=="__main__":
    print "Hello"