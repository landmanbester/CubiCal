# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
import numpy as np
from collections import Counter, OrderedDict
import pyrap.tables as pt
import cPickle
import re
import traceback
import sys
from cubical.tools import shared_dict
import cubical.flagging as flagging
from cubical.flagging import FL
from pdb import set_trace as BREAK  # useful: can set static breakpoints by putting BREAK() in the code

# Try to import montblanc: if not successful, remember error for later.

try:
    import montblanc
except:
    montblanc = None
    montblanc_import_error = sys.exc_info()

if montblanc is not None:
    from cubical.MBTiggerSim import simulate, MSSourceProvider, ColumnSinkProvider
    from cubical.TiggerSourceProvider import TiggerSourceProvider
    from montblanc.impl.rime.tensorflow.sources import CachedSourceProvider, FitsBeamSourceProvider

from cubical.tools import logger, ModColor
log = logger.getLogger("data_handler")

def _parse_range(arg, nmax):
    """
    Helper function. Parses an argument into a list of numbers. Nmax is max number.
    Supports e.g. 5, "5", "5~7" (inclusive range), "5:8" (pythonic range), "5,6,7" (list).

    Args:
        arg (int or tuple or list or str):
            Raw range expression.
        nmax (int):
            Maximum possible range.

    Returns:
        list:
            Range of numbers.

    Raises:
        TypeError:
            If the type of arg is not understood. 
        ValueError:
            If the range cannot be parsed.
    """

    fullrange = range(nmax)

    if arg is None:
        return fullrange
    elif type(arg) is int:
        return [arg]
    elif type(arg) is tuple:
        return list(arg)
    elif type(arg) is list:
        return arg
    elif type(arg) is not str:
        raise TypeError("Cannot parse range of type '%s'."%type(arg))

    arg = arg.strip()

    if re.match("\d+$", arg):
        return [ int(arg) ]
    elif "," in arg:
        return map(int,','.split(arg))
    
    if re.match("(\d*)~(\d*)$", arg):
        i0, i1 = arg.split("~", 1)
        i0 = int(i0) if i0 else None
        i1 = int(i1)+1 if i1 else None
    elif re.match("(\d*):(\d*)$", arg):
        i0, i1 = arg.split(":", 1)
        i0 = int(i0) if i0 else None
        i1 = int(i1) if i1 else None
    else:
        raise ValueError("Cannot parse range '%s'."%arg)
    
    return fullrange[slice(i0,i1)]


## TERMINOLOGY:
## A "chunk" is data for one DDID, a range of timeslots (thus, a subset of the MS rows), and a 
## slice of channels. Chunks are the basic parallelization unit. Solver deals with a chunk of data.
##
## A "row chunk" is data for one DDID, a range of timeslots, and *all* channels. One can imagine a 
## row chunk as a "horizontal" vector of chunks across frequency.
##
## A "tile" is a collection of row chunks that are adjacent in time and/or DDID. One can imagine a 
## tile as a vertical stack of row chunks

class RowChunk(object):
    """ Very basic helper class. Encapsulates a row chunk. """

    def __init__(self, ddid, tchunk, rows):
        """
        Initialises a RowChunk.

        Args:
            ddid (int):
                DDID index for the RowChunk.
            tchunk (int):
                Time index for the RowChunk.
            rows (np.ndarray):
                An (nrows_in_chunk) size array of row indices.
        """

        self.ddid, self.tchunk, self.rows = ddid, tchunk, rows

class Tile(object):
    """
    Helper class which encapsulates a tile. A tile is a sequence of row chunks that's read and
    written as a unit.
    """
    
    # The tile list is effectively global. This is needed because worker subprocesses need to 
    # access the tiles.
    
    tile_list = None

    def __init__(self, handler, chunk):
        """
        Initialises a tile and sets the first row chunk.

        Args:
            handler (:obj:`~cubical.data_handler.ReadModelHandler`):
                Data hander object.
            chunk (:obj:`~cubical.data_handler.RowChunk`):
                Row chunk which is used to initialise the tile.
        """

        self.handler = handler
        self.rowchunks = [chunk]
        self.first_row = chunk.rows[0]
        self.last_row = chunk.rows[-1]
        self._rows_adjusted = False
        self._updated = False
        self.data = None

    def append(self, chunk):
        """
        Appends a row chunk to a tile.

        Args:
            chunk (:obj:`~cubical.data_handler.RowChunk`):
                Row chunk which will be appended to the assosciated tile.
        """

        self.rowchunks.append(chunk)
        self.first_row = min(self.first_row, chunk.rows[0])
        self.last_row = max(self.last_row, chunk.rows[-1])

    def merge(self, other):
        """
        Merges another tile into this one.

        Args:
            other (:obj:`~cubical.data_handler.Tile`):
                Tile which will be merged.
        """

        self.rowchunks += other.rowchunks
        self.first_row = min(self.first_row, other.first_row)
        self.last_row = max(self.last_row, other.last_row)

    def finalize(self):
        """
        Creates a list of chunks within the tile that can be iterated over and creates a list of
        chunk labels.

        This also adjusts the row indices of all row chunks so that they become relative to the 
        start of the tile.
        """

        self._data_dict_name = "DATA:{}:{}".format(self.first_row, self.last_row)

        # Adjust row indices so they become relative to the first row of the tile.

        if not self._rows_adjusted:
            for rowchunk in self.rowchunks:
                rowchunk.rows -= self.first_row
            self._rows_adjusted = True

        # Create a dict of { chunk_label: rows, chan0, chan1 } for all chunks in this tile.

        self._chunk_dict = OrderedDict()
        self._chunk_indices = {}
        num_freq_chunks = len(self.handler.chunk_find)-1
        for rowchunk in self.rowchunks:
            for ifreq in range(num_freq_chunks):
                key = "D{}T{}F{}".format(rowchunk.ddid, rowchunk.tchunk, ifreq)
                chan0, chan1 = self.handler.chunk_find[ifreq:ifreq + 2]
                self._chunk_dict[key] = rowchunk, chan0, chan1
                self._chunk_indices[key] = rowchunk.tchunk, rowchunk.ddid * num_freq_chunks + ifreq

        # Copy various useful info from handler and make a simple list of unique ddids.

        self.ddids = np.unique([rowchunk.ddid for rowchunk,_,_ in self._chunk_dict.itervalues()])
        self.ddid_col = self.handler.ddid_col[self.first_row:self.last_row+1]
        self.time_col = self.handler.time_col[self.first_row:self.last_row+1]
        self.antea = self.handler.antea[self.first_row:self.last_row+1]
        self.anteb = self.handler.anteb[self.first_row:self.last_row+1]
        self.times = self.handler.times[self.first_row:self.last_row+1]
        self.ctype = self.handler.ctype
        self.nants = self.handler.nants
        self.ncorr = self.handler.ncorr
        self.nchan = self.handler._nchans[0]

    def get_chunk_indices(self, key):
        """ Returns chunk indices based on the key value. """

        return self._chunk_indices[key]

    def get_chunk_keys(self):
        """ Returns all chunk keys. """

        return self._chunk_dict.iterkeys()

    def get_chunk_tfs(self, key):
        """
        Returns timestamps and freqs for the given chunk assosicated with key, as well as two slice 
        objects describing its position in the global time/freq space.

        Args:
            key (str):
                The label corresponding to the chunk of interest.

        Returns:
            tuple:
                Unique times, channel frequencies, time axis slice, frequency axis slice.
        """
        
        rowchunk, chan0, chan1 = self._chunk_dict[key]
        timeslice = slice(self.times[rowchunk.rows[0]], self.times[rowchunk.rows[-1]] + 1)
        
        return self.handler.uniq_times[timeslice], \
               self.handler._chanfr[rowchunk.ddid, chan0:chan1], \
               slice(self.times[rowchunk.rows[0]], self.times[rowchunk.rows[-1]] + 1), \
               slice(rowchunk.ddid * self.handler.nfreq + chan0, rowchunk.ddid * self.handler.nfreq + chan1)

    def load(self, load_model=True):
        """
        Fetches data from MS into tile data shared dict. This is meant to be called in the main 
        or I/O process.
        
        Args:
            load_model (bool, optional):
                If False, omits weights and model visibilities.

        Returns:
            :obj:`~cubical.tools.shared_dict.SharedDict`:
                Shared dictionary containing the MS data relevant to the tile.
        
        Raises:
            RuntimeError:
                If neither --model-lsm nor --model-column set (see [model] section in 
                DefaultParset.cfg).
        """
        
        # Create a shared dict for the data arrays.
        
        data = shared_dict.create(self._data_dict_name)

        # These flags indicate if the (corrected) data or flags have been updated
        # Gotcha for shared_dict users! The only truly shared objects are arrays.
        # Thus, we create an array for the flags.
        
        data['updated'] = np.array([False, False])

        print>>log,"reading tile for MS rows {}~{}".format(self.first_row, self.last_row)
        
        nrows = self.last_row - self.first_row + 1
        
        data['obvis'] = self.handler.fetch(
                         self.handler.data_column, self.first_row, nrows).astype(self.handler.ctype)
        print>> log(2), "  read " + self.handler.data_column

        data['uvwco'] = self.handler.fetch("UVW", self.first_row, nrows)
        print>> log(2), "  read UVW coordinates"

        # The following either reads model visibilities from the measurement set, or uses an lsm 
        # and Montblanc to simulate them. Data may need to be massaged to be compatible with 
        # Montblanc's strict requirements. 

        if load_model:
            if self.handler.sm_name:

                print>>log, "computing model visibilities"

                expected_nrows, sort_ind, row_identifiers = self.prep_data(data)

                srcs, snks = [], []

                measet_src = MSSourceProvider(self, data, sort_ind)
                tigger_src = TiggerSourceProvider(self)
                cached_src = CachedSourceProvider(tigger_src, clear_start=True, clear_stop=True)
                cached_ms_src = CachedSourceProvider(measet_src, 
                                                     cache_data_sources=["parallactic_angles"],
                                                     clear_start=False, clear_stop=False)

                srcs.append(cached_ms_src)
                srcs.append(cached_src)

                if self.handler.beam_pattern:
                    arbeam_src = FitsBeamSourceProvider(self.handler.beam_pattern,
                                                        self.handler.beam_l_axis,
                                                        self.handler.beam_m_axis)
                    srcs.append(arbeam_src)

                ndirs = tigger_src._nclus
                model_shape = np.array([ndirs, 1, expected_nrows, self.nchan, self.ncorr])

                data.addSharedArray('movis', model_shape, self.handler.ctype)

                column_snk = ColumnSinkProvider(self, data, sort_ind)

                snks.append(column_snk)

                for direction in xrange(ndirs):
                    print>>log(2), "simulating visbilities in direction {}.".format(direction)
                    simulate(srcs, snks, self.handler.mb_opts)
                    tigger_src.update_target()
                    column_snk._dir += 1

                self.unprep_data(data, nrows)

                data['movis'] = data['movis'][:,:,row_identifiers,:,:]

            elif self.handler.model_column:
                print>>log, "no LSM specified, reading {} only".format(self.handler.model_column)
                model_shape = [1,1] + list(data['obvis'].shape)
                data.addSharedArray('movis', model_shape, self.handler.ctype)

            else:
                raise RuntimeError("neither --model-lsm nor --model-column set")

            if self.handler.model_column:

                data['movis'][0,0,...] += self.handler.fetch(self.handler.model_column, 
                                                             self.first_row,
                                                             nrows).astype(self.handler.ctype)

                print>> log(2), "  read " + self.handler.model_column

            if self.handler.weight_column:
                weight = self.handler.fetch(self.handler.weight_column, self.first_row, nrows)
                # If weight_column is WEIGHT, expand along the freq axis (looks like WEIGHT SPECTRUM).
                if self.handler.weight_column == "WEIGHT":
                    data['weigh'] = weight[:, np.newaxis, :].repeat(self.handler.nfreq, 1)
                else:
                    data['weigh'] = weight
                print>> log(2), "  read weights from column {}".format(self.handler.weight_column)

        data.addSharedArray('covis', data['obvis'].shape, self.handler.ctype)

        # The following block of code deals with the various flagging operations and columns. The
        # aim is to correctly populate flag_arr from the various flag sources.

        # Make a flag array. This will contain FL.PRIOR for any points flagged in the MS.

        flag_arr = data.addSharedArray("flags", data['obvis'].shape, dtype=FL.dtype)

        # FLAG/FLAG_ROW only needed if applying them, or auto-filling BITLAG from them.

        flagcol = flagrow = None

        if self.handler._apply_flags or self.handler._auto_fill_bitflag:
            flagcol = self.handler.fetch("FLAG", self.first_row, nrows)
            flagrow = self.handler.fetch("FLAG_ROW", self.first_row, nrows)
            print>> log(2), "  read FLAG/FLAG_ROW"

        if self.handler._apply_flags:
            flag_arr[flagcol] = FL.PRIOR
            flag_arr[flagrow, :, :] = FL.PRIOR

        # Form up bitflag array, if needed.
        if self.handler._apply_bitflags or self.handler._save_bitflag or self.handler._auto_fill_bitflag:
            read_bitflags = False
            # If not explicitly re-initializing, try to read column.
            if not self.handler._reinit_bitflags:
                self.bflagrow = self.handler.fetch("BITFLAG_ROW", self.first_row, nrows)
                # If there's an error reading BITFLAG, it must be unfilled. This is a common 
                # occurrence so we may as well deal with it. In this case, if auto-fill is set, 
                # fill BITFLAG from FLAG/FLAG_ROW.
                try:
                    self.bflagcol = self.handler.fetch("BITFLAG", self.first_row, nrows)
                    print>> log(2), "  read BITFLAG/BITFLAG_ROW"
                    read_bitflags = True
                except Exception:
                    if not self.handler._auto_fill_bitflag:
                        print>> log, ModColor.Str(traceback.format_exc().strip())
                        print>> log, ModColor.Str("Error reading BITFLAG column, and --flags-auto-init is not set.")
                        raise
                    print>>log,"  error reading BITFLAG column: not fatal, since we'll auto-fill it from FLAG"
                    for line in traceback.format_exc().strip().split("\n"):
                        print>> log, "    "+line
            # If column wasn't read, create arrays.
            if not read_bitflags:
                self.bflagcol = np.zeros(flagcol.shape, np.int32)
                self.bflagrow = np.zeros(flagrow.shape, np.int32)
                if self.handler._auto_fill_bitflag:
                    self.bflagcol[flagcol] = self.handler._auto_fill_bitflag
                    self.bflagrow[flagrow] = self.handler._auto_fill_bitflag
                    # mark flags as updated: they will be saved below
                    data['updated'][1] = True
                    print>> log, "  auto-filled BITFLAG/BITFLAG_ROW of shape %s"%str(self.bflagcol.shape)
            if self.handler._apply_bitflags:
                flag_arr[(self.bflagcol & self.handler._apply_bitflags) != 0] = FL.PRIOR
                flag_arr[(self.bflagrow & self.handler._apply_bitflags) != 0, :, :] = FL.PRIOR

        # Create a placeholder for the gain solutions
        data.addSubdict("solutions")

        return data

    def prep_data(self, data):
        """
        Manipulates data to be consistent with Montblanc's requirements. Mainly adds elements 
        which are missing from the measurement set.

        Args:
            data (:obj:`~cubical.tools.shared_dict.SharedDict`):
                Shared dictionary containing data from the measurement set.

        Returns:
            tuple:
                The expected_nrows, sorted_ind and row_identifiers for the massaged data.

        Raises: 
            ValueError:
                If the number of rows remains inconsistent after removing auto-correlations.")
        """

        # Given data, we need to make sure that it looks the way MB wants it to.
        # First step - check the number of rows.

        n_bl = (self.nants*(self.nants - 1))/2
        ntime = len(np.unique(self.time_col))

        nrows = self.last_row - self.first_row + 1
        expected_nrows = n_bl*ntime*len(self.ddids)

        # The row identifiers determine which rows in the SORTED/ALL ROWS are required for the data
        # that is present in the MS. Essentially, they allow for the selection of an array of a size
        # matching that of the observed data. First term determines the offset by ddid, the second
        # is the offset by time, and the last turns antea and anteb into a unique offset per 
        # baseline.

        ddid_ind = self.ddid_col.copy()

        for ind, ddid in enumerate(self.ddids):
            ddid_ind[ddid_ind==ddid] = ind

        row_identifiers = ddid_ind*n_bl*ntime + (self.times - np.min(self.times))*n_bl + \
                          (-0.5*self.antea**2 + (self.nants - 1.5)*self.antea + self.anteb - 1).astype(np.int32)

        # Based on the number of rows versus the expected number of rows, we determine whether rows
        # must be added or removed. If rows must be removed, we assume they are all 
        # auto-correlations and raise an error if removing them still yields and inconsistent 
        # number of rows.

        if nrows == expected_nrows:
            logstr = (nrows, ntime, n_bl, len(self.ddids))
            print>> log, "  {} rows consistent with {} timeslots and {} baselines" \
                                                                "across {} bands".format(*logstr)
            
            sorted_ind = np.lexsort((self.anteb, self.antea, self.time_col, self.ddid_col))

        elif nrows < expected_nrows:
            logstr = (nrows, ntime, n_bl, len(self.ddids))
            print>> log, "  {} rows inconsistent with {} timeslots and {} baselines" \
                                                                "across {} bands".format(*logstr)
            print>> log, "  {} fewer rows than expected".format(expected_nrows - nrows)

            nmiss = expected_nrows - nrows

            baselines = [(a,b) for a in xrange(self.nants) for b in xrange(self.nants) if b>a]

            missing_bl = []
            missing_t = []
            missing_ddids = []

            for ddid in self.ddids:
                for t in np.unique(self.time_col):
                    t_sel = np.where((self.time_col==t)&(self.ddid_col==ddid))
                
                    missing_bl.extend(set(baselines) - set(zip(self.antea[t_sel], self.anteb[t_sel])))
                    missing_t.extend([t]*(n_bl - t_sel[0].size))
                    missing_ddids.extend([ddid]*(n_bl - t_sel[0].size))

            missing_uvw = [[0,0,0]]*nmiss 
            missing_antea = np.array([bl[0] for bl in missing_bl])
            missing_anteb = np.array([bl[1] for bl in missing_bl])
            missing_t = np.array(missing_t)
            missing_ddids = np.array(missing_ddids)

            data['uvwco'] = np.concatenate((data['uvwco'], missing_uvw))
            self.antea = np.concatenate((self.antea, missing_antea))
            self.anteb = np.concatenate((self.anteb, missing_anteb))
            self.time_col = np.concatenate((self.time_col, missing_t))
            self.ddid_col = np.concatenate((self.ddid_col, missing_ddids))

            sorted_ind = np.lexsort((self.anteb, self.antea, self.time_col, self.ddid_col))

        elif nrows > expected_nrows:
            logstr = (nrows, ntime, n_bl, len(self.ddids))
            print>> log, "  {} rows inconsistent with {} timeslots and {} baselines" \
                                                                "across {} bands".format(*logstr)
            print>> log, "  {} more rows than expected".format(nrows - expected_nrows)
            print>> log, "  assuming additional rows are auto-correlations - ignoring"

            sorted_ind = np.lexsort((self.anteb, self.antea, self.time_col, self.ddid_col))            
            sorted_ind = sorted_ind[np.where(self.antea!=self.anteb)]

            if np.shape(sorted_ind) != expected_nrows:
                raise ValueError("Number of rows inconsistent after removing auto-correlations.")

        return expected_nrows, sorted_ind, row_identifiers

    def unprep_data(self, data, nrows):
        """
        Reverts the changes made by prepdata. Makes data consistent with the measurement set.

        Args:
            data (:obj:`~cubical.tools.shared_dict.SharedDict`):
                Shared dictionary containing data from the measurement set.
            nrows (int):
                Number of rows which were present in the measurement set prior to prep_data.
        """

        data['uvwco'] = data['uvwco'][:nrows,...]
        self.antea = self.antea[:nrows]
        self.anteb = self.anteb[:nrows]
        self.time_col = self.time_col[:nrows]
        self.ddid_col = self.ddid_col[:nrows]

    def get_chunk_cubes(self, key):
        """
        Produces the CubiCal data cubes corresponding to the specified key.

        Args:
            key (str):
                The label corresponding to the chunk of interest.
    
        Returns:
            tuple:
                The data, model, flags and weights cubes for the given chunk key. 
                Shapes are as follows:
            
                - data (np.ndarray):    [n_mod, n_tim, n_fre, n_ant, n_ant, 2, 2]
                - model (np.ndarray):   [n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, 2, 2]
                - flags (np.ndarray):   [n_tim, n_fre, n_ant, n_ant]
                - weights (np.ndarray): [n_mod, n_tim, n_fre, n_ant, n_ant] or None for no weighting

                n_mod refers to number of models simultaneously fitted.
        """

        data = shared_dict.attach(self._data_dict_name)

        rowchunk, freq0, freq1 = self._chunk_dict[key]

        t_dim = self.handler.chunk_ntimes[rowchunk.tchunk]
        f_dim = freq1 - freq0
        freq_slice = slice(freq0, freq1)
        rows = rowchunk.rows
        nants = self.handler.nants

        flags = self._column_to_cube(data['flags'], t_dim, f_dim, rows, freq_slice, FL.dtype, FL.MISSING)
        flags = np.bitwise_or.reduce(flags, axis=-1) if self.ncorr==4 else np.bitwise_or.reduce(flags[...,::3], axis=-1)
        obs_arr = self._column_to_cube(data['obvis'], t_dim, f_dim, rows, freq_slice, self.handler.ctype, reqdims=6)
        obs_arr = obs_arr.reshape(list(obs_arr.shape[:-1]) + [2, 2])
        if 'movis' in data:
            mod_arr = self._column_to_cube(data['movis'], t_dim, f_dim, rows, freq_slice, self.handler.ctype, reqdims=7)
            mod_arr = mod_arr.reshape(list(mod_arr.shape[:-1]) + [2, 2])
            # flag invalid model visibilities
            flags[(~np.isfinite(mod_arr[0, 0, ...])).any(axis=(-2, -1))] |= FL.INVALID
        else:
            mod_arr = None

        # flag invalid data
        flags[(~np.isfinite(obs_arr[0, ])).any(axis=(-2, -1))] |= FL.INVALID
        flagged = flags != 0

        if 'weigh' in data:
            wgt_arr = self._column_to_cube(data['weigh'], t_dim, f_dim, rows, freq_slice, self.handler.ftype)
            wgt_arr = np.sqrt(np.sum(wgt_arr, axis=-1))  # take the square root of sum over correlations
            wgt_arr[flagged] = 0
            wgt_arr = wgt_arr.reshape([1, t_dim, f_dim, nants, nants])
        else:
            wgt_arr = None

        # zero flagged entries in data and model
        obs_arr[0, flagged, :, :] = 0
        if mod_arr is not None:
            mod_arr[0, 0, flagged, :, :] = 0

        return obs_arr, mod_arr, flags, wgt_arr

    def set_chunk_cubes(self, cube, flag_cube, key, column='covis'):
        """
        Copies a visibility cube, and an optional flag cube, back to tile column.

        Args:
            cube (np.ndarray):
                Cube containing visibilities.
            flag_cube (np.ndarray):
                Cube containing flags.
            key (str):
                The label corresponding to the chunk of interest.
            column (str, optional):
                The column to which the cube must be copied.
        """
        data = shared_dict.attach(self._data_dict_name)
        rowchunk, freq0, freq1 = self._chunk_dict[key]
        rows = rowchunk.rows
        freq_slice = slice(freq0, freq1)
        if cube is not None:
            data['updated'][0] = True
            self._cube_to_column(data[column], cube, rows, freq_slice)
        if flag_cube is not None:
            data['updated'][1] = True
            self._cube_to_column(data['flags'], flag_cube, rows, freq_slice, flags=True)

    def create_solutions_chunk_dict(self, key):
        """
        Creates a shared dict for the given chunk in which to store gain solutions.
        
        Args:
            key (str):
                The label corresponding to the chunk of interest.

        Returns:
            :obj:`~cubical.tools.shared_dict.SharedDict`:
                Shared dictionary containing gain solutions.
        """

        data = shared_dict.attach(self._data_dict_name)
        sd = data['solutions'].addSubdict(key)

        return sd

    def iterate_solution_chunks(self):
        """
        Iterates over per-chunk solution dictionaries. 

        Yields:
            tuple:
                A gain subdictionary and the time and frequency slices to which it corresponds:

                - Subdictionary (:obj:`~cubical.tools.shared_dict.SharedDict`))
                - Time slice (slice)
                - Frequency slice (slice)
        """
        
        data = shared_dict.attach(self._data_dict_name)
        soldict = data['solutions']
        for key in soldict.iterkeys():
            yield soldict[key]

    def save(self, unlock=False):
        """
        Saves 'corrected' column, and any updated flags, back to MS.

        Args:
            unlock (bool, optional):
                If True, calls the unlock method on the handler.
        """
        nrows = self.last_row - self.first_row + 1
        data = shared_dict.attach(self._data_dict_name)
        if self.handler.output_column and data['updated'][0]:
            print>> log, "saving {} for MS rows {}~{}".format(self.handler.output_column, self.first_row, self.last_row)
            if self.handler._add_column(self.handler.output_column):
                self.handler.reopen()
            self.handler.data.putcol(self.handler.output_column, data['covis'], self.first_row, nrows)

        if self.handler._save_bitflag and data['updated'][1]:
            print>> log, "saving flags for MS rows {}~{}".format(self.first_row, self.last_row)
            # clear bitflag column first
            self.bflagcol &= ~self.handler._save_bitflag
            # add bitflag to points where data wasn't flagged for prior reasons
            self.bflagcol[(data['flags']&~FL.PRIOR) != 0] |= self.handler._save_bitflag
            self.handler.data.putcol("BITFLAG", self.bflagcol, self.first_row, nrows)
            print>> log, "  updated BITFLAG column"
            self.bflagrow = np.bitwise_and.reduce(self.bflagcol,axis=(-1,-2))
            self.handler.data.putcol("BITFLAG_ROW", self.bflagrow, self.first_row, nrows)
            flag_col = self.bflagcol != 0
            self.handler.data.putcol("FLAG", flag_col, self.first_row, nrows)
            print>> log, "  updated FLAG column ({:.2%} visibilities flagged)".format(
                flag_col.sum() / float(flag_col.size))
            flag_row = flag_col.all(axis=(-1, -2))
            self.handler.data.putcol("FLAG_ROW", flag_row, self.first_row, nrows)
            print>> log, "  updated FLAG_ROW column ({:.2%} rows flagged)".format(
                flag_row.sum() / float(flag_row.size))

        if unlock:
            self.handler.unlock()

    def release(self):
        """ Releases the shared memory data dict. """

        data = shared_dict.attach(self._data_dict_name)
        data.delete()

    def _column_to_cube(self, column, chunk_tdim, chunk_fdim, rows, freqs, dtype, zeroval=0, reqdims=5):
        """
        Converts input data into N-dimensional measurement matrices.

        Args:
            column (np.ndarray):
                column array from which this will be filled
            chunk_tdim (int):  
                Timeslots per chunk.
            chunk_fdim (int): 
                Frequencies per chunk.
            rows (np.ndarray):
                Row slice (or set of indices).
            freqs (slice):       
                Frequency slice.
            dtype (various):       
                Data type of the resulting measurement matrix.
            zeroval (various, optional):
                Null value with which to fill missing array elements.
            reqdims (int):
                Required number of output dimensions.

        Returns:
            np.ndarray:
                Output cube of with reqdims axes.
        """

        # Start by establishing the possible dimensions and those actually present. Dimensions which
        # are not present are set to one, for flexibility reasons. Output shape is determined by
        # reqdims, which selects dimensions in reverse order from (ndir, nmod, nt, nf, na, na, nc). 
        # NOTE: The final dimension will be reshaped into 2x2 blocks outside this function.

        col_ndim = column.ndim

        possible_dims = ["dirs", "mods", "rows", "freqs", "cors"]

        dims = {possible_dims[-i] : column.shape[-i] for i in xrange(1, col_ndim + 1)}

        dims.setdefault("mods", 1)
        dims.setdefault("dirs", 1)

        out_shape = [dims["dirs"], dims["mods"], chunk_tdim, chunk_fdim, self.nants, self.nants, 4]
        out_shape = out_shape[-reqdims:]

        # Creates empty N-D array into which the column data can be packed.
        out_arr = np.full(out_shape, zeroval, dtype)

        # Grabs the relevant time and antenna info.

        achunk = self.antea[rows]
        bchunk = self.anteb[rows]
        tchunk = self.times[rows]
        tchunk -= np.min(tchunk)

        # Creates lists of selections to make subsequent selection from column and out_arr easier.

        corr_slice = slice(None) if self.ncorr==4 else slice(None, None, 3)

        col_selections = [[dirs, mods, rows, freqs, slice(None)][-col_ndim:] 
                            for dirs in xrange(dims["dirs"]) for mods in xrange(dims["mods"])]

        cub_selections = [[dirs, mods, tchunk, slice(None), achunk, bchunk, corr_slice][-reqdims:]
                            for dirs in xrange(dims["dirs"]) for mods in xrange(dims["mods"])]

        n_sel = len(col_selections)

        # The following takes the arbitrarily ordered data from the MS and places it into a N-D 
        # data structure (correlation matrix).

        for col_selection, cub_selection in zip(col_selections, cub_selections):

            if self.ncorr == 4:
                out_arr[cub_selection] = colsel = column[col_selection]
                cub_selection[-3], cub_selection[-2] = cub_selection[-2], cub_selection[-3]
                if dtype == self.ctype:
                    out_arr[cub_selection] = colsel.conj()[..., (0, 2, 1, 3)]
                else:
                    out_arr[cub_selection] = colsel[..., (0, 2, 1, 3)]
            
            elif self.ncorr == 2:
                out_arr[cub_selection] = colsel = column[col_selection]
                cub_selection[-3], cub_selection[-2] = cub_selection[-2], cub_selection[-3]
                if dtype == self.ctype:
                    out_arr[cub_selection] = colsel.conj()
                else:
                    out_arr[cub_selection] = colsel
            
            elif self.ncorr == 1:
                out_arr[cub_selection] = colsel = column[col_selection][..., (0,0)]
                cub_selection[-3], cub_selection[-2] = cub_selection[-2], cub_selection[-3]
                if dtype == self.ctype:
                    out_arr[cub_selection] = colsel.conj()
                else:
                    out_arr[cub_selection] = colsel

        # This zeros the diagonal elements in the "baseline" plane. This is purely a precaution - 
        # we do not want autocorrelations on the diagonal.
        
        out_arr[..., range(self.nants), range(self.nants), :] = zeroval

        return out_arr


    def _cube_to_column(self, column, in_arr, rows, freqs, flags=False):
        """
        Converts a measurement matrix back into an MS style column.

        Args:
            in_arr (np.ndarray):
                Input array which is to be made MS friendly.
            rows (np.ndarray): 
                Row indices or slice.
            freqs (slice): 
                Frequency slice.
            flags (bool, optional): 
                If True, input array is a flag cube (i.e. no correlation axes).
        """

        tchunk = self.times[rows]
        tchunk -= tchunk[0]  # is this correct -- does in_array start from beginning of chunk?
        achunk = self.antea[rows]
        bchunk = self.anteb[rows]

        # Flag cube has no correlation axis, so copy it into output column.
        
        if flags:
            column[rows, freqs, :] = in_arr[tchunk, :, achunk, bchunk, np.newaxis]
        
        # For other cubes, rehape the last two axes into one (faltten 2x2 correlation block).
        else:
            chunk = in_arr[tchunk, :, achunk, bchunk, :]
            newshape = list(chunk.shape[:-2]) + [chunk.shape[-2]*chunk.shape[-1]]
            chunk = chunk.reshape(newshape)
            if self.ncorr == 4:
                column[rows, freqs, :] = chunk
            elif self.ncorr == 2:                         # 2 corr -- take elements 0,3
                column[rows, freqs, :] = chunk[..., ::3]  
            elif self.ncorr == 1:                         # 1 corr -- take element 0
                column[rows, freqs, :] = chunk[..., :1]


class DataHandler:
    """ Main data handler. Interfaces with the measurement set. """

    def __init__(self, ms_name, data_column, sm_name, model_column, output_column=None,
                 taql=None, fid=None, ddid=None, flagopts={}, double_precision=False, ddes=False, 
                 weight_column=None, beam_pattern=None, beam_l_axis=None, beam_m_axis=None,
                 mb_opts=None):
        """
        Initialises a DataHandler object.

        Args:
            ms_name (str):
                Name of measeurement set.
            data_colum (str):
                Name of the input observed data column.
            sm_name (str):
                Name of sky model.
            model_column (str):
                Name of input model column.
            output_column (str or None, optional):
                Name of output column if specified, else None.
            taql (str):
                Additional TAQL query for data selection.
            fid (int or None, optional):
                Field identifier if specified, else None.
            ddid (int, list or None, optional):
                Data descriptor identifer/s if specified, else None.
            flagopts (dict, optional):
                Flagging options.
            double_precision (bool, optional):
                Use 64-bit precision if True, else 32-bit.
            ddes (bool, optional):
                If True, use direction dependent simulation.
            weight_column (str or None, optional):
                Name of input weight column if specified, else None.
            beam_pattern (str or None, optional):
                Pattern for reading beam files if specified, else None.
            beam_l_axis (str or None, optional):
                Corresponding axis in fits beam, else None.
            beam_m_axis (str or None, optional):
                Corresponding axis in fits beam, else None.
            mb_opts (dict or None):
                Dictionary of Montblanc options if specified, else None.

        Raises:
            RuntimeError:
                If Montblanc cannot be imported but simulation is required.
            ValueError:
                If selection from MS returns no rows.
        """

        if montblanc is None and sm_name:
            print>>log, ModColor.Str("Error importing Montblanc: ")
            for line in traceback.format_exception(*montblanc_import_error):
                print>>log, "  "+ModColor.Str(line)
            print>>log, ModColor.Str("Without Montblanc, LSM functionality is not available.")
            raise RuntimeError("Error importing Montblanc")

        self.ms_name = ms_name
        self.sm_name = sm_name
        self.mb_opts = mb_opts
        self.beam_pattern = beam_pattern
        self.beam_l_axis = beam_l_axis
        self.beam_m_axis = beam_m_axis

        self.fid = fid if fid is not None else 0

        self.ms = pt.table(self.ms_name, readonly=False, ack=False)

        print>>log, ModColor.Str("reading MS %s"%self.ms_name, col="green")

        _anttab = pt.table(self.ms_name + "::ANTENNA", ack=False)
        _fldtab = pt.table(self.ms_name + "::FIELD", ack=False)
        _spwtab = pt.table(self.ms_name + "::SPECTRAL_WINDOW", ack=False)
        _poltab = pt.table(self.ms_name + "::POLARIZATION", ack=False)
        _ddesctab = pt.table(self.ms_name + "::DATA_DESCRIPTION", ack=False)
        _feedtab = pt.table(self.ms_name + "::FEED", ack=False)

        self.ctype = np.complex128 if double_precision else np.complex64
        self.ftype = np.float64 if double_precision else np.float32
        self.nfreq = _spwtab.getcol("NUM_CHAN")[0]
        self.ncorr = _poltab.getcol("NUM_CORR")[0]
        self.nants = _anttab.nrows()

        self._nchans = _spwtab.getcol("NUM_CHAN")
        self._rfreqs = _spwtab.getcol("REF_FREQUENCY")
        self._chanfr = _spwtab.getcol("CHAN_FREQ")
        self._antpos = _anttab.getcol("POSITION")
        self._phadir = _fldtab.getcol("PHASE_DIR", startrow=self.fid, nrow=1)[0][0]
        self._poltype = np.unique(_feedtab.getcol('POLARIZATION_TYPE')['array'])
        
        if np.any([pol in self._poltype for pol in ['L','l','R','r']]):
            self._poltype = "circular"
            self.feeds = "rl"
        elif np.any([pol in self._poltype for pol in ['X','x','Y','y']]):
            self._poltype = "linear"
            self.feeds = "xy"
        else:
            print>>log,"  unsupported feed type. Terminating."
            sys.exit()

        # print some info on MS layout
        print>>log,"  detected {} ({}) feeds".format(self._poltype, self.feeds)
        print>>log,"  fields are "+", ".join(["{}{}: {}".format('*' if i==fid else "",i,name) for i, name in enumerate(_fldtab.getcol("NAME"))])
        self._spw_chanfreqs = _spwtab.getcol("CHAN_FREQ")  # nspw x nfreq array of frequencies
        print>>log,"  {} spectral windows of {} channels each ".format(*self._spw_chanfreqs.shape)

        # figure out DDID range
        self._ddids = _parse_range(ddid, _ddesctab.nrows())

        # use TaQL to select subset
        self.taql = self.build_taql(taql, fid, self._ddids)

        if self.taql:
            print>> log, "  applying TAQL query: %s" % self.taql
            self.data = self.ms.query(self.taql)
        else:
            self.data = self.ms

        self.nrows = self.data.nrows()

        self._datashape = (self.nrows, self.nfreq, self.ncorr)

        if not self.nrows:
            raise ValueError("MS selection returns no rows")

        self.time_col = self.fetch("TIME")
        self.uniq_times = np.unique(self.time_col)
        self.ntime = len(self.uniq_times)

        self._ddid_spw = _ddesctab.getcol("SPECTRAL_WINDOW_ID")
        # select frequencies corresponding to DDID range
        self._ddid_chanfreqs = np.array([self._spw_chanfreqs[self._ddid_spw[ddid]] for ddid in self._ddids ])

        self.all_freqs = self._ddid_chanfreqs.ravel()

        print>>log,"  %d antennas, %d rows, %d/%d DDIDs, %d timeslots, %d channels, %d corrs" % (self.nants,
                    self.nrows, len(self._ddids), _ddesctab.nrows(), self.ntime, self.nfreq, self.ncorr)
        print>>log,"  DDID central frequencies are at {} GHz".format(
                    " ".join(["%.2f"%(self._ddid_chanfreqs[i][self.nfreq/2]*1e-9) for i in range(len(self._ddids))]))
        self.nddid = len(self._ddids)


        self.data_column = data_column
        self.model_column = model_column
        self.weight_column = weight_column
        self.output_column = output_column
        self.simulate = bool(self.sm_name)
        self.use_ddes = ddes

        # figure out flagging situation
        if "BITFLAG" in self.ms.colnames():
            if flagopts["reinit-bitflags"]:
                self.ms.removecols("BITFLAG")
                if "BITFLAG_ROW" in self.ms.colnames():
                    self.ms.removecols("BITFLAG_ROW")
                print>> log, ModColor.Str("Removing BITFLAG column, since --flags-reinit-bitflags is set.")
                bitflags = None
            else:
                bitflags = flagging.Flagsets(self.ms)
        else:
            bitflags = None
        apply_flags  = flagopts.get("apply")
        save_bitflag = flagopts.get("save")
        auto_init    = flagopts.get("auto-init")

        self._reinit_bitflags = flagopts["reinit-bitflags"]
        self._apply_flags = self._apply_bitflags = self._save_bitflag = self._auto_fill_bitflag = None

        # no BITFLAG. Should we auto-init it?

        if auto_init:
            if not bitflags:
                self._add_column("BITFLAG", like_type='int')
                if "BITFLAG_ROW" not in self.ms.colnames():
                    self._add_column("BITFLAG_ROW", like_col="FLAG_ROW", like_type='int')
                self.reopen()
                bitflags = flagging.Flagsets(self.ms)
                self._auto_fill_bitflag = bitflags.flagmask(auto_init, create=True)
                print>> log, ModColor.Str("Will auto-fill new BITFLAG '{}' ({}) from FLAG/FLAG_ROW".format(auto_init, self._auto_fill_bitflag), col="green")
            else:
                self._auto_fill_bitflag = bitflags.flagmask(auto_init, create=True)
                print>> log, "BITFLAG column found. Will auto-fill with '{}' ({}) from FLAG/FLAG_ROW if not filled".format(auto_init, self._auto_fill_bitflag)

        # OK, we have BITFLAG somehow -- use these

        if bitflags:
            self._apply_flags = None
            self._apply_bitflags = 0
            if apply_flags:
                # --flags-apply specified as a bitmask, or a string, or a list of strings
                if type(apply_flags) is int:
                    self._apply_bitflags = apply_flags
                else:
                    if type(apply_flags) is str:
                        apply_flags = apply_flags.split(",")
                    for fset in apply_flags:
                        self._apply_bitflags |= bitflags.flagmask(fset)
            if self._apply_bitflags:
                print>> log, ModColor.Str("Applying BITFLAG {} ({}) to input data".format(apply_flags, self._apply_bitflags), col="green")
            else:
                print>> log, ModColor.Str("No flags will be read, since --flags-apply was not set.")
            if save_bitflag:
                self._save_bitflag = bitflags.flagmask(save_bitflag, create=True)
                print>> log, ModColor.Str("Will save new flags into BITFLAG '{}' ({}), and into FLAG/FLAG_ROW".format(save_bitflag, self._save_bitflag), col="green")

        # else no BITFLAG -- fall back to using FLAG/FLAG_ROW if asked, but definitely can'tr save

        else:
            if save_bitflag:
                raise RuntimeError("No BITFLAG column in this MS. Either use --flags-auto-init to insert one, or disable --flags-save.")
            self._apply_flags = bool(apply_flags)
            self._apply_bitflags = 0
            if self._apply_flags:
                print>> log, ModColor.Str("No BITFLAG column in this MS. Using FLAG/FLAG_ROW.")
            else:
                print>> log, ModColor.Str("No flags will be read, since --flags-apply was not set.")

        self.gain_dict = {}

    def build_taql(self, taql=None, fid=None, ddid=None):
        """
        Generate a combined TAQL query using possible options.

        Args:
            taql (str or None, optional):
                Additional TAQL query for data selection.
            fid (int or None, optional):
                Field identifier if specified, else None.
            ddid (int, list or None, optional):
                Data descriptor identifer/s if specified, else None.

        Returns:
            str:
                A TAQL query string. 
        """

        if taql:
            taqls = [ "(" + taql +")" ]
        else:
            taqls = []

        if fid is not None:
            taqls.append("FIELD_ID == %d" % fid)

        if ddid is not None:
            if isinstance(ddid,(tuple,list)):
                taqls.append("DATA_DESC_ID IN [%s]" % ",".join(map(str,ddid)))
            else:
                taqls.append("DATA_DESC_ID == %d" % ddid)

        return " && ".join(taqls)

    def fetch(self, *args, **kwargs):
        """
        Convenience function which mimics pyrap.tables.table.getcol().

        Args:
            args (tuple): 
                Variable length argument list.
            kwargs (dict): 
                Arbitrary keyword arguments.

        Returns:
            np.ndarray:
                Result of getcol(\*args, \*\*kwargs).
        """

        return self.data.getcol(*args, **kwargs)

    def define_chunk(self, tdim=1, fdim=1, chunk_by=None, chunk_by_jump=0, min_chunks_per_tile=4):
        """
        Fetches indexing columns (TIME, DDID, ANTENNA1/2) and defines the chunk dimensions for 
        the data.

        Args:
            tdim (int): 
                Timeslots per chunk.
            fdim (int): 
                Frequencies per chunk.
            chunk_by (str or None, optional):   
                If set, chunks will have boundaries imposed by jumps in the listed columns
            chunk_by_jump (int, optional): 
                The magnitude of a jump has to be over this value to force a chunk boundary.
            min_chunks_per_tile (int, optional): 
                The minimum number of chunks to be placed in a single tile.
            
        Attributes:
            antea (np.ndarray): ANTENNA1 column of MS subset.
            anteb (np.ndarray): 
                ANTENNA2 column of MS subset.
            ddid_col (np.ndarray): 
                DDID column of MS subset.
            time_col (np.ndarray): 
                TIME column of MS subset.
            times (np.ndarray):    
                Timeslot index number with same size as self.time_col.
            uniq_times (np.ndarray): 
                Unique timestamps in time_col.
        """

        self.antea = self.fetch("ANTENNA1")
        self.anteb = self.fetch("ANTENNA2")
        # read TIME and DDID columns, because those determine our chunking strategy
        self.time_col = self.fetch("TIME")
        self.ddid_col = self.fetch("DATA_DESC_ID")
        print>> log, "  read indexing columns"
        # list of unique times
        self.uniq_times = np.unique(self.time_col)
        # timeslot index (per row, each element gives index of timeslot)
        self.times = np.empty_like(self.time_col, dtype=np.int32)
        for i, t in enumerate(self.uniq_times):
            self.times[self.time_col == t] = i
        print>> log, "  built timeslot index ({} unique timestamps)".format(len(self.uniq_times))

        self.chunk_tdim = tdim
        self.chunk_fdim = fdim

        # TODO: this assumes each DDID has the same number of channels. I don't know of cases where it is not true,
        # but, technically, this is not precluded by the MS standard. Need to handle this one day
        self.chunk_find = range(0, self.nfreq, self.chunk_fdim)
        self.chunk_find.append(self.nfreq)
        num_freq_chunks = len(self.chunk_find) - 1

        print>> log, "  using %d freq chunks: %s" % (num_freq_chunks, " ".join(map(str, self.chunk_find)))

        # Constructs a list of timeslots at which we cut our time chunks. Use scans if specified, else
        # simply break up all timeslots

        if chunk_by:
            scan_chunks = self.check_contig(chunk_by, chunk_by_jump)
            timechunks = []
            for scan_num in xrange(len(scan_chunks) - 1):
                timechunks.extend(range(scan_chunks[scan_num], scan_chunks[scan_num+1], self.chunk_tdim))
        else:
            timechunks = range(0, self.times[-1], self.chunk_tdim)
        timechunks.append(self.times[-1]+1)        
        
        print>>log,"  found %d time chunks: %s"%(len(timechunks)-1, " ".join(map(str, timechunks)))

        # Number of timeslots per time chunk
        self.chunk_ntimes = []
        
        # Unique timestamps per time chunk
        self.chunk_timestamps = []
        
        # For each time chunk, create a mask for associated rows.
        
        timechunk_mask = {}
        
        for tchunk in range(len(timechunks) - 1):
            ts0, ts1 = timechunks[tchunk:tchunk + 2]
            timechunk_mask[tchunk] = (self.times>=ts0) & (self.times<ts1)
            self.chunk_ntimes.append(ts1-ts0)
            self.chunk_timestamps.append(np.unique(self.times[timechunk_mask[tchunk]]))

        # now make list of "row chunks": each element will be a tuple of (ddid, time_chunk_number, rowlist)

        chunklist = []

        for ddid in self._ddids:
            ddid_rowmask = self.ddid_col==ddid

            for tchunk in range(len(timechunks)-1):
                rows = np.where(ddid_rowmask & timechunk_mask[tchunk])[0]
                if rows.size:
                    chunklist.append(RowChunk(ddid, tchunk, rows))

        print>>log,"  generated {} row chunks based on time and DDID".format(len(chunklist))

        # init this, for compatibility with the chunk iterator below
        self.chunk_rind = OrderedDict([ ((chunk.ddid, chunk.tchunk), chunk.rows) for chunk in chunklist])

        # re-sort these row chunks into naturally increasing order (by first row of each chunk)
        def _compare_chunks(a, b):
            return cmp(a.rows[0], b.rows[0])
        chunklist.sort(cmp=_compare_chunks)

        # now, break the row chunks into tiles. Tiles are an "atom" of I/O. First, we try to define each tile as a
        # sequence of overlapping row chunks (i.e. chunks such that the first row of a subsequent chunk comes before
        # the last row of the next chunk). Effectively, if DDIDs are interleaved with timeslots, then all per-DDIDs
        # chunks will be grouped into a single tile.
        # It is also possible that we end up with one chunk = one tile (i.e. no chunks overlap).
        tile_list = []
        for chunk in chunklist:
            # if rows do not overlap, start new tile with this chunk
            if not tile_list or chunk.rows[0] > tile_list[-1].last_row:
                tile_list.append(Tile(self,chunk))
            # else extend previous tile
            else:
                tile_list[-1].append(chunk)

        print>> log, "  row chunks yield {} potential tiles".format(len(tile_list))

        # now, for effective I/O and parallelisation, we need to have a minimum amount of chunks per tile.
        # Coarsen our tiles to achieve this
        coarser_tile_list = []
        for tile in tile_list:
            # start new "coarse tile" if previous coarse tile already has the min number of chunks
            if not coarser_tile_list or len(coarser_tile_list[-1].rowchunks)*num_freq_chunks >= min_chunks_per_tile:
                coarser_tile_list.append(tile)
            else:
                coarser_tile_list[-1].merge(tile)

        Tile.tile_list = coarser_tile_list
        for tile in Tile.tile_list:
            tile.finalize()

        print>> log, "  coarsening this to {} tiles (min {} chunks per tile)".format(len(Tile.tile_list), min_chunks_per_tile)

    def check_contig(self, columns, jump_by=0):
        """
        Helper method, finds ranges of timeslots where the named columns do not change.

        Args:
            columns (list):
                Column names on which it base the check.
            jump_by (int, optional):
                Magnitude of a jump after which we force a chunk boundary.

        Returns:
            list:
                The chunk boundaries.
        """

        boundaries = {0, self.ntime}
        
        for column in columns:
            value = self.fetch(column)
            boundary_rows = np.where(abs(np.roll(value, 1) - value) > jump_by)[0]
            boundaries.update([self.times[i] for i in boundary_rows])

        return sorted(boundaries)

    def flag3_to_col(self, flag3):
        """
        Converts a 3D flag cube (ntime, nddid, nchan) back into the MS style.

        Args:
            flag3 (np.ndarray): 
                Input array which is to be made MS friendly.

        Returns:
            np.ndarray:
                Boolean array with same shape as self.obvis.
        """

        ntime, nddid, nchan = flag3.shape

        flagout = np.zeros(self._datashape, bool)

        for ddid in xrange(nddid):
            ddid_rows = self.ddid_col == ddid
            for ts in xrange(ntime):
                # find all rows associated with this DDID and timeslot
                rows = ddid_rows & (self.times == ts)
                if rows.any():
                    flagout[rows, :, :] = flag3[ts, ddid, :, np.newaxis]

        return flagout

    def add_to_gain_dict(self, gains, bounds, t_int=1, f_int=1):
        """
        Adds a gain array to the gain dictionary.

        Args:
            gains (np.ndarray):
                Gains for the current chunk.
            bounds (tuple):
                Tuple of (ddid, timechunk, first_f, last_f).
            t_int (int, optional):
                Number of timeslots per solution interval.
            f_int (int, optional):
                Number of frequencies per soultion interval.
        """

        n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = gains.shape

        ddid, timechunk, first_f, last_f = bounds

        timestamps = self.chunk_timestamps[timechunk]

        freqs = range(first_f,last_f)
        freq_indices = [[] for i in xrange(n_fre)]

        for f, freq in enumerate(freqs):
            freq_indices[f//f_int].append(freq)

        for d in xrange(n_dir):
            for t in xrange(n_tim):
                for f in xrange(n_fre):
                    comp_idx = (d,tuple(timestamps),tuple(freq_indices[f]))
                    self.gain_dict[comp_idx] = gains[d,t,f,:]

    def write_gain_dict(self, output_name=None):
        """
        Writes out a gain dictionary to disk.

        Args:
            output_name (str or None, optional):
                Name of output pickle file.
        """

        if output_name is None:
            output_name = self.ms_name + "/gains.p"

        cPickle.dump(self.gain_dict, open(output_name, "wb"), protocol=2)

    def _add_column (self, col_name, like_col="DATA", like_type=None):
        """
        Inserts a new column into the measurement set.

        Args:
            col_name (str): 
                Name of target column.
            like_col (str, optional): 
                Column will be patterned on the named column.
            like_type (str or None, optional): 
                If set, column type will be changed.

        Returns:
            bool:
                True if a new column was inserted, else False.
        """

        if col_name not in self.ms.colnames():
            # new column needs to be inserted -- get column description from column 'like_col'
            print>> log, "  inserting new column %s" % (col_name)
            desc = self.ms.getcoldesc(like_col)
            desc["name"] = col_name
            desc['comment'] = desc['comment'].replace(" ", "_")  # got this from Cyril, not sure why
            # if a different type is specified, insert that
            if like_type:
                desc['valueType'] = like_type
            self.ms.addcols(desc)
            return True
        return False

    def unlock(self):
        """ Unlocks the measurement set and shared memory dictionary. """

        if self.taql:
            self.data.unlock()
        self.ms.unlock()

    def lock(self):
        """ Locks the measurement set and shared memory dictionary. """

        self.ms.lock()
        if self.taql:
            self.data.lock()

    def close(self):
        """ Closes the measurement set and shared memory dictionary. """

        if self.taql:
            self.data.close()
        self.ms.close()

    def flush(self):
        """ Flushes the measurement set and shared memory dictionary. """

        if self.taql:
            self.data.flush()
        self.ms.flush()

    def reopen(self):
        """ Reopens the MS. Unfortunately, this is needed when new columns are added. """

        self.close()
        self.ms = self.data = pt.table(self.ms_name, readonly=False, ack=False)
        if self.taql:
            self.data = self.ms.query(self.taql)

    def save_flags(self, flags):
        """
        Saves flags to column in MS.

        Args:
            flags (np.ndarray): 
                Flag values to be written to column.
        """
        
        print>>log,"Writing out new flags"
        bflag_col = self.fetch("BITFLAG")
        # raise specified bitflag
        print>> log, "  updating BITFLAG column flagbit %d"%self._save_bitflag
        bflag_col &= ~self._save_bitflag         # clear the flagbit first
        bflag_col[flags] |= self._save_bitflag   # now set it where flagged
        self.data.putcol("BITFLAG", bflag_col)
        print>>log, "  updating BITFLAG_ROW column"
        self.data.putcol("BITFLAG_ROW", np.bitwise_and.reduce(bflag_col, axis=(-1,-2)))
        flag_col = bflag_col != 0
        print>> log, "  updating FLAG column ({:.2%} visibilities flagged)".format(
                                                                flag_col.sum()/float(flag_col.size))
        self.data.putcol("FLAG", flag_col)
        flag_row = flag_col.all(axis=(-1,-2))
        print>> log, "  updating FLAG_ROW column ({:.2%} rows flagged)".format(
                                                                flag_row.sum()/float(flag_row.size))
        self.data.putcol("FLAG_ROW", flag_row)

