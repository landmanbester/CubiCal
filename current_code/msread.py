from pyrap.tables import table
import numpy as np
from collections import Counter, OrderedDict
# from pprint import pprint


class DataHandler:
    """
    Handles measurement set interface and data container construction.

    Attributes:
        ms_name (str): Name of measurement set.
        data (table): Table containing measurement set data.
        nrows (int): Number of rows in the data table.
        ntime (int): Number of unique times in the measurement set.
        nfreq (int): Number of frequencies in the measurement set.
        ncorr (int): Number of correlations in the measurement set.
        nants (int): Number of antennas in the measurement set.
        tdict (dict): Dictionary of unique times and their occurrences.
        obvis: Observed visibility data.
        movis: Model visibility data.
        antea: First antenna column.
        anteb: Second antenna column.
        times: Time data.
        flags: Flag data.
        bflag: Bitflag data.
        chunk_tdim: Timeslots per chunk.
        chunk_fdim: Frequencies per chunk.
        chunk_tind: Indices for time axis.
        chunk_find: Indices for frequency axis.
        chunk_tkey: Unique times in the chunk.
        apply_flags (bool): Turns flagging functionality on or off.
        self.bitmask: Mask for use with the BITFLAG column.
    """

    def __init__(self, ms_name):
        """
        Initialisation method for DataHandler.

        Args:
            ms_name (str): Name of measurement set.
        """
        self.ms_name = ms_name
        self.data = table(self.ms_name)

        self.nrows = self.data.nrows()
        self.ntime = len(np.unique(self.fetch("TIME")))
        self.nfreq = self.data.getcoldesc("DATA")["shape"][0]
        self.ncorr = self.data.getcoldesc("DATA")["shape"][1]
        self.nants = table(self.ms_name + "/ANTENNA").nrows()

        self.obvis = None
        self.movis = None
        self.antea = None
        self.anteb = None
        self.times = None
        self.tdict = None
        self.flags = None
        self.bflag = None

        self.chunk_tdim = None
        self.chunk_fdim = None
        self.chunk_tind = None
        self.chunk_find = None
        self.chunk_tkey = None

        self.apply_flags = False
        self.bitmask = None

    def fetch(self, *args, **kwargs):
        """
        Convenience function which mimics pyrap.tables.table.getcol().

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            data.getcol(*args, **kwargs)
        """

        return self.data.getcol(*args, **kwargs)

    def index_t(self, times):
        """
        Converts times into indices.

        Args:
            times (np.array): Time data.

        Returns:
            times (np.array): Time data as indices.
        """

        for i, j in enumerate(np.unique(times)):
            times[times == j] = i

        return times.astype(np.int64)

    def fetch_all(self, *args, **kwargs):
        """
        Convenenience function for grabbing all the necessary data from the MS.
        Assigns values to initialised attributes.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        self.obvis = self.fetch("DATA", *args, **kwargs)
        self.movis = self.fetch("MODEL_DATA", *args, **kwargs)
        self.antea = self.fetch("ANTENNA1", *args, **kwargs)
        self.anteb = self.fetch("ANTENNA2", *args, **kwargs)
        self.times = self.index_t(self.fetch("TIME", *args, **kwargs))
        self.tdict = OrderedDict(sorted(Counter(self.times).items()))

        self.flags = self.fetch("FLAG", *args, **kwargs)
        try:
            self.bflag = self.fetch("BITFLAG", *args, **kwargs)
        except:
            print "No BITFLAG column in MS."

    def define_chunk(self, tdim=1, fdim=1):
        """
        Defines the chunk dimensions for the data.

        Args:
            tdim (int): Timeslots per chunk.
            fdim (int): Frequencies per chunk.
        """

        self.chunk_tdim = tdim
        self.chunk_fdim = fdim

        self.chunk_tind = [0]
        self.chunk_tind.extend(self.tdict.values())
        self.chunk_tind = list(np.cumsum(self.chunk_tind)[::self.chunk_tdim])
        if self.chunk_tind[-1] != self.nrows:
            self.chunk_tind.append(self.nrows)

        self.chunk_tkey = self.tdict.keys()[::self.chunk_tdim]
        self.chunk_tkey.append(self.ntime)

        self.chunk_find = range(0, self.nfreq, self.chunk_fdim)
        self.chunk_find.append(self.nfreq)

    def make_flag_array(self, f_t_row, l_t_row, f_f_col, l_f_col):
        """
        Combines flags into a flag array which can be applied to the data.

        Args:
            f_t_row (int): First time row to be accessed in data.
            l_t_row (int): Last time row to be accessed in data.
            f_f_col (int): First frequency column to be accessed in data.
            l_f_col (int): Last frequency column to be accessed in data.

        Returns:
            flags_arr (np.array): Array containing flags for the active data.
        """

        flags_arr = ~self.flags[f_t_row:l_t_row, f_f_col:l_f_col, :]

        if self.bitmask is not None:
            flags_arr &= ((self.bflag[f_t_row:l_t_row, f_f_col:l_f_col,
                           :] & self.bitmask) == 0)

        return flags_arr

    def vis_to_array(self, chunk_tdim, chunk_fdim, f_t_row, l_t_row, f_f_col,
                     l_f_col):
        """
        Converts input data into N-dimensional measurement matrices.

        Args:
            chunk_tdim (int):  Timeslots per chunk.
            chunk_fdim (int): Frequencies per chunk.
            f_t_row (int): First time row to be accessed in data.
            l_t_row (int): Last time row to be accessed in data.
            f_f_col (int): First frequency column to be accessed in data.
            l_f_col (int): Last frequency column to be accessed in data.

        Returns:
            flags_arr (np.array): Array containing flags for the active data.
        """

        # Creates empty 5D arrays into which the model and observed data can
        # be packed. TODO: 6D? dtype?

        obser_arr = np.empty([chunk_tdim, chunk_fdim, self.nants,
                              self.nants, self.ncorr], dtype=np.complex128)

        model_arr = np.empty([chunk_tdim, chunk_fdim, self.nants,
                              self.nants, self.ncorr], dtype=np.complex128)

        # Grabs the relevant time and antenna info.

        tchunk = self.times[f_t_row:l_t_row]
        tchunk -= np.min(tchunk)
        achunk = self.antea[f_t_row:l_t_row]
        bchunk = self.anteb[f_t_row:l_t_row]

        # The following takes the arbitrarily ordered data from the MS and
        # places it into a 5D data structure (measurement matrix).

        if self.apply_flags:
            flags_arr = self.make_flag_array(f_t_row, l_t_row, f_f_col, l_f_col)
            self.obvis[f_t_row:l_t_row, f_f_col:l_f_col, :] *= flags_arr
            self.movis[f_t_row:l_t_row, f_f_col:l_f_col, :] *= flags_arr

        obser_arr[tchunk, :, achunk, bchunk, :] \
            = self.obvis[f_t_row:l_t_row, f_f_col:l_f_col, :]
        obser_arr[tchunk, :, bchunk, achunk, :] \
            = self.obvis[f_t_row:l_t_row, f_f_col:l_f_col, :].conj()

        model_arr[tchunk, :, achunk, bchunk, :] \
            = self.movis[f_t_row:l_t_row, f_f_col:l_f_col, :]
        model_arr[tchunk, :, bchunk, achunk, :] \
            = self.movis[f_t_row:l_t_row, f_f_col:l_f_col, :].conj()

        # This zeros the diagonal elements in the "baseline" plane. This is
        # purely a precaution - we do not want autocorrelations on the
        # diagonal. TODO: For loop with fill_diagonal?

        obser_arr.reshape([-1, chunk_fdim, self.nants**2, self.ncorr])[
            :, :, ::self.nants + 1, :] = 0

        model_arr.reshape([-1, chunk_fdim, self.nants**2, self.ncorr])[
            :, :, ::self.nants + 1, :] = 0

        obser_arr = obser_arr.reshape([chunk_tdim, chunk_fdim,
                                       self.nants, self.nants, 2, 2])
        model_arr = model_arr.reshape([chunk_tdim, chunk_fdim,
                                       self.nants, self.nants, 2, 2])

        return obser_arr, model_arr

    def __iter__(self):
        return next(self)

    def next(self):
        """
        Generator for the DataHandler object.

        Yields:
            np.array: The next N-dimensional measurement matrix to be processed.
        """

        for i in xrange(len(self.chunk_tind[:-1])):
            for j in xrange(len(self.chunk_find[:-1])):

                first_t = self.chunk_tind[i]
                last_t = self.chunk_tind[i + 1]

                first_f = self.chunk_find[j]
                last_f = self.chunk_find[j + 1]

                t_dim = self.chunk_tkey[i + 1] - self.chunk_tkey[i]
                f_dim = last_f - first_f

                yield self.vis_to_array(t_dim, f_dim, first_t, last_t, first_f,
                                        last_f)
