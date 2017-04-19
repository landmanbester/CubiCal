import numpy as np
from collections import Counter, OrderedDict
import pyrap.tables as pt
import cPickle
import re
#import better_exceptions

from Tools import logger, ModColor
log = logger.getLogger("ReadModelHandler")

class FL(object):
    """Namespace for flag bits"""
    PRIOR    = 1       # prior flags (i.e. from MS)
    MISSING  = 1<<1    # missing data
    INVALID  = 1<<2    # invalid data or model (inf, nan)
    NOCONV   = 1<<4    # no convergence
    CHISQ    = 1<<5    # excessive chisq
    GOOB     = 1<<6    # gain solution out of bounds
    BOOM     = 1<<7    # gain solution exploded (i.e. went to inf/nan)
    GNULL    = 1<<8    # gain solution gone to zero

    @staticmethod
    def categories():
        """Returns dict of all flag categories defined above"""
        return OrderedDict([(attr, value) for attr, value in FL.__dict__.iteritems() if attr[0] != "_" and type(value) is int])


def _parse_range(arg, nmax):
    """Helper function. Parses an argument into a list of numbers. Nmax is max number.
    Supports e.g. 5, "5", "5~7" (inclusive range), "5:8" (pythonic range), "5,6,7" (list)
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
        raise TypeError("can't parse range of type '%s'"%type(arg))
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
        raise ValueError("can't parse range '%s'"%arg)
    return fullrange[slice(i0,i1)]


class ReadModelHandler:

    def __init__(self, ms_name, data_column, sm_name, model_column, taql=None, fid=None, ddid=None,
                 precision="32", ddes=False, weight_column=None):

        self.ms_name = ms_name
        self.sm_name = sm_name
        self.fid = fid if fid is not None else 0

        self.ms = pt.table(self.ms_name, readonly=False, ack=False)

        print>>log, ModColor.Str("reading MS %s"%self.ms_name, col="green")

        self._anttab = pt.table(self.ms_name + "::ANTENNA", ack=False)
        self._fldtab = pt.table(self.ms_name + "::FIELD", ack=False)
        self._spwtab = pt.table(self.ms_name + "::SPECTRAL_WINDOW", ack=False)
        self._poltab = pt.table(self.ms_name + "::POLARIZATION", ack=False)
        self._ddesctab = pt.table(self.ms_name + "::DATA_DESCRIPTION", ack=False)

        self.ctype = np.complex128 if precision=="64" else np.complex64
        self.ftype = np.float64 if precision=="64" else np.float32
        self.flagtype = np.int32
        self.nfreq = self._spwtab.getcol("NUM_CHAN")[0]
        self.ncorr = self._poltab.getcol("NUM_CORR")[0]
        self.nants = self._anttab.nrows()

        self._nchans = self._spwtab.getcol("NUM_CHAN")
        self._rfreqs = self._spwtab.getcol("REF_FREQUENCY")
        self._chanfr = self._spwtab.getcol("CHAN_FREQ")
        self._antpos = self._anttab.getcol("POSITION")
        self._phadir = self._fldtab.getcol("PHASE_DIR", startrow=self.fid,
                                           nrow=1)[0][0]

        # print some info on MS layout
        print>>log,"  fields are "+", ".join(["{}{}: {}".format('*' if i==fid else "",i,name) for i, name in enumerate(self._fldtab.getcol("NAME"))])
        self._spw_chanfreqs = self._spwtab.getcol("CHAN_FREQ")  # nspw x nfreq array of frequencies
        print>>log,"  {} spectral windows of {} channels each ".format(*self._spw_chanfreqs.shape)

        # figure out DDID range
        self._ddids = _parse_range(ddid, self._ddesctab.nrows())

        # use TaQL to select subset
        self.taql = self.build_taql(taql, fid, self._ddids)

        if self.taql:
            print>> log, "Applying TAQL query: %s" % self.taql
            self.data = self.ms.query(self.taql)
        else:
            self.data = self.ms

        self.nrows = self.data.nrows()

        if not self.nrows:
            raise ValueError("MS selection returns no rows")

        self.ntime = len(np.unique(self.fetch("TIME")))

        self._ddid_spw = self._ddesctab.getcol("SPECTRAL_WINDOW_ID")
        # select frequencies corresponding to DDID range
        self._ddid_chanfreqs = np.array([self._spw_chanfreqs[self._ddid_spw[ddid]] for ddid in self._ddids ])

        print>>log,"%d antennas, %d rows, %d/%d DDIDs, %d timeslots, %d channels, %d corrs" % (self.nants,
                    self.nrows, len(self._ddids), self._ddesctab.nrows(), self.ntime, self.nfreq, self.ncorr)
        print>>log,"DDID central frequencies are at {} GHz".format(
                    " ".join(["%.2f"%(self._ddid_chanfreqs[i][self.nfreq/2]*1e-9) for i in range(len(self._ddids))]))

        self.obvis = None
        self.movis = None
        self.covis = None
        self.antea = None
        self.anteb = None
        self.rtime = None
        self.times = None
        self.tdict = None
        self.flags = None
        self.bflag = None
        self.weigh = None
        self.uvwco = None
        self.nddid = len(self._ddids)

        self.chunk_tdim = None
        self.chunk_fdim = None
        self.chunk_tind = None
        self.chunk_find = None
        self.chunk_tkey = None
        self._first_f = None
        self._last_f = None

        self.data_column = data_column
        self.model_column = model_column
        self.weight_column = weight_column
        self.simulate = bool(self.sm_name)
        self.use_ddes = ddes
        self.apply_flags = False
        self.bitmask = None
        self.gain_dict = {}

    def build_taql(self, taql=None, fid=None, ddid=None):

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
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            data.getcol(*args, **kwargs)
        """

        return self.data.getcol(*args, **kwargs)

    def mass_fetch(self, *args, **kwargs):
        """
        Convenience function for grabbing all the necessary data from the MS.
        Assigns values to initialised attributes.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        self.obvis = self.fetch(self.data_column, *args, **kwargs).astype(self.ctype)
        print>>log,"  read "+self.data_column
        if self.model_column:
            self.movis = self.fetch(self.model_column, *args, **kwargs).astype(self.ctype)
            print>> log, "  read " + self.model_column
        else:
            self.movis = np.zeros_like(self.obvis)
        self.antea = self.fetch("ANTENNA1", *args, **kwargs)
        self.anteb = self.fetch("ANTENNA2", *args, **kwargs)
        # time & DDID columns
        self.time_col = self.fetch("TIME", *args, **kwargs)
        self.ddid_col = self.fetch("DATA_DESC_ID", *args, **kwargs)
        # list of unique times
        self.uniq_times = np.unique(self.time_col)
        # timeslot index (per row, each element gives index of timeslot)
        self.times = np.empty_like(self.time_col, dtype=np.int32)
        for i, t in enumerate(self.uniq_times):
            self.times[self.time_col == t] = i
        # not that timeslot does not need to be monotonic, but must be monotonic within each DDID
        # # check that times increase monotonically
        # if min(self.times[1:]-self.times[:-1]) < 0:
        #     print>>log, ModColor.Str("TIME column is not in increasing order. Cannot deal with this MS!")
        #     raise RuntimeError
        self.covis = np.empty_like(self.obvis)
        self.uvwco = self.fetch("UVW", *args, **kwargs)
        print>> log, "  read TIME, DATA_DESC_ID, UVW"

        if self.weight_column:
            self.weigh = self.fetch(self.weight_column, *args, **kwargs)
            # if column is WEIGHT, expand freq axis
            if self.weight_column == "WEIGHT":
                self.weigh = self.weigh[:,np.newaxis,:].repeat(self.nfreq, 1)
            print>> log, "  read weights from column {}".format(self.weight_column)

        # make a flag array. This will contain FL.PRIOR for any points flagged in the MS
        self.flags = np.zeros(self.obvis.shape, dtype=self.flagtype)
        if self.apply_flags:
            flags = self.fetch("FLAG", *args, **kwargs)
            print>> log, "  read FLAG"
            # apply FLAG_ROW on top
            flags[self.fetch("FLAG_ROW", *args, **kwargs),:,:] = True
            print>> log, "  read FLAG_ROW"
            # apply BITFLAG on top, if present
            if "BITFLAG" in self.ms.colnames():
                bflag = self.fetch("BITFLAG", *args, **kwargs)
                print>> log, "  read BITFLAG"
                bflag |= self.fetch("BITFLAG_ROW", *args, **kwargs)[:, np.newaxis, np.newaxis]
                print>> log, "  read BITFLAG_ROW"
                flags |= ((bflag&(self.bitmask or 0)) != 0)
            self.flags[flags] = FL.PRIOR

    def define_chunk(self, tdim=1, fdim=1, single_chunk_id=None):
        """
        Defines the chunk dimensions for the data.

        Args:
            tdim (int): Timeslots per chunk.
            fdim (int): Frequencies per chunk.
            single_chunk_id:   If set, iterator will yield only the one specified chunk. Useful for debugging.
        """

        self.chunk_tdim = tdim
        self.chunk_fdim = fdim
        self._single_chunk_id  = single_chunk_id

        # Constructs a list of timeslots at which we cut our time chunks.

        scan_chunks = self.check_contig()
        
        timechunks = []
        for scan_num in xrange(len(scan_chunks) - 1):
            timechunks.extend(range(scan_chunks[scan_num], scan_chunks[scan_num+1], self.chunk_tdim))
        timechunks.append(self.times[-1]+1)        
        
        print>>log,"found %d time chunks: %s"%(len(timechunks)-1, " ".join(map(str, timechunks)))

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

        # now, chunk_rind (chunk row index) will be an ordered dict, keyed by ddid,timechunk tuple.
        # Per each such tuple, it gives a _list of row indices_ corresponding to that chunk
        
        self.chunk_rind = OrderedDict()
        
        for ddid in self._ddids:
            ddid_rowmask = self.ddid_col==ddid
            for tchunk in range(len(timechunks)-1):
                self.chunk_rind[ddid,tchunk] = np.where(ddid_rowmask & timechunk_mask[tchunk])[0]

        print>>log,"will generate %d row chunks based on time and DDID"%(len(self.chunk_rind),)

        # TODO: this assumes each DDID has the same number of channels. I don't know of cases where it is not true,
        # but, technically, this is not precluded by the MS standard. Need to handle this...
        self.chunk_find = range(0, self.nfreq, self.chunk_fdim)
        self.chunk_find.append(self.nfreq)

        print>>log,"using %d freq chunks: %s"%(len(self.chunk_find)-1, " ".join(map(str, self.chunk_find)))

    def check_contig(self):

        scan = self.fetch("SCAN_NUMBER")

        if np.all(scan==scan[0]):
            scan_t = [0, self.ntime]
        else:
            scan_i = np.where(np.roll(scan,1)!=scan)[0]
            scan_t = list(self.times[scan_i])
            scan_t.append(self.ntime)

        return scan_t

    def __iter__(self):
        return next(self)

    def next(self):
        """
        Generator for the ReadModeHandler object.

        Yields:
            np.array: The next N-dimensional measurement matrix to be processed.
        """

        obs_arr = None
        mod_arr = None
        wgt_arr = None
        ichunk = 0

        for (ddid, tchunk), rows in self.chunk_rind.iteritems():
            nrows = len(rows)
            if not nrows:
                continue

            for j in xrange(len(self.chunk_find[:-1])):
                ichunk += 1
                self._chunk_label = "D%dT%dF%d" % (ddid, tchunk, j)
                if self._single_chunk_id and self._single_chunk_id != self._chunk_label:
                    continue

                self._chunk_ddid = ddid
                self._chunk_tchunk = tchunk
                self._chunk_fchunk = ddid*self.nfreq + j

                self._first_f = self.chunk_find[j]
                self._last_f = self.chunk_find[j + 1]

                t_dim = self.chunk_ntimes[tchunk]
                f_dim = self._last_f - self._first_f

                # validity array has the inverse meaning of flags (True if value is present and not flagged)
                flags = self.col_to_arr("flags", t_dim, f_dim, rows)
                obs_arr = self.col_to_arr("obser", t_dim, f_dim, rows)
                mod_arr = self.col_to_arr("model", t_dim, f_dim, rows)
                # flag invalid data or model
                flags[(~np.isfinite(obs_arr)).any(axis=(-2,-1))] |= FL.INVALID
                flags[(~np.isfinite(mod_arr[0,...])).any(axis=(-2,-1))] |= FL.INVALID
                # zero flagged entries in data and model
                obs_arr[flags!=0, :, :] = 0
                mod_arr[0, flags!=0, :, :] = 0

                if self.weight_column:
                    wgt_arr = self.col_to_arr("weigh", t_dim, f_dim, rows)
                    wgt_arr[flags!=0] = 0

                if self.simulate:
                    try:
                        import MBTiggerSim as mbt
                        import TiggerSourceProvider as tsp
                    except:
                        print>> log, ModColor.Str("Montblanc failed to initialize. Can't predict from an LSM.")
                        raise

                    mssrc = mbt.MSSourceProvider(self, t_dim, f_dim)
                    tgsrc = tsp.TiggerSourceProvider(self._phadir, self.sm_name,
                                                         use_ddes=self.use_ddes)
                    arsnk = mbt.ArraySinkProvider(self, t_dim, f_dim, tgsrc._nclus)

                    srcprov = [mssrc, tgsrc]
                    snkprov = [arsnk]

                    for clus in xrange(tgsrc._nclus):
                        mbt.simulate(srcprov, snkprov)
                        tgsrc.update_target()
                        arsnk._dir += 1

                    mod_shape = list(arsnk._sim_array.shape)[:-1] + [2,2]
                    mod_arr1 = arsnk._sim_array.reshape(mod_shape)
                    # add in MODEL_DATA column, if we had it
                    mod_arr1[0,...] += mod_arr[0,...]
                    mod_arr = mod_arr1

                yield obs_arr, mod_arr, flags, wgt_arr, (self._chunk_tchunk, self._chunk_fchunk), self._chunk_label


    def col_to_arr(self, target, chunk_tdim, chunk_fdim, rows):
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

        opts = {"model": (self.movis, self.ctype),
                "obser": (self.obvis, self.ctype),
                "weigh": (self.weigh, self.ftype),
                "flags": (self.flags, self.flagtype) }

        column, dtype = opts[target]

        # Creates empty 5D array into which the column data can be packed.

        out_arr = np.zeros([chunk_tdim, chunk_fdim, self.nants, self.nants, 4], dtype=dtype)
        # initial state of flags is FL.MISSING -- to be filled by actual flags where data is available
        if target is "flags":
            out_arr[:] = FL.MISSING

        # Grabs the relevant time and antenna info.

        achunk = self.antea[rows]
        bchunk = self.anteb[rows]
        tchunk = self.times[rows]
        tchunk -= np.min(tchunk)

        # Creates a tuple of slice objects to make subsequent slicing easier.

        selection = (rows,
                     slice(self._first_f, self._last_f),
                     slice(None))

        # The following takes the arbitrarily ordered data from the MS and
        # places it into tho 5D data structure (correlation matrix). 

        if self.ncorr==4:
            out_arr[tchunk, :, achunk, bchunk, :] = colsel = column[selection]
            if dtype == self.ctype:
                out_arr[tchunk, :, bchunk, achunk, :] = colsel.conj()[...,(0,2,1,3)]
            else:
                out_arr[tchunk, :, bchunk, achunk, :] = colsel[..., (0, 2, 1, 3)]
        elif self.ncorr==2:
            out_arr[tchunk, :, achunk, bchunk, ::3] = colsel = column[selection]
            out_arr[tchunk, :, achunk, bchunk, 1:3] = 0
            out_arr[tchunk, :, bchunk, achunk, 1:3] = 0
            if dtype == self.ctype:
                out_arr[tchunk, :, bchunk, achunk, ::3] = colsel.conj()
            else:
                out_arr[tchunk, :, bchunk, achunk, ::3] = colsel
        elif self.ncorr==1:
            out_arr[tchunk, :, achunk, bchunk, ::3] = colsel = column[selection][:,:,(0,0)]
            out_arr[tchunk, :, achunk, bchunk, 1:3] = 0
            out_arr[tchunk, :, bchunk, achunk, 1:3] = 0
            if dtype == self.ctype:
                out_arr[tchunk, :, bchunk, achunk, ::3] = colsel.conj()
            else:
                out_arr[tchunk, :, bchunk, achunk, ::3] = colsel

        # This zeros the diagonal elements in the "baseline" plane. This is
        # purely a precaution - we do not want autocorrelations on the
        # diagonal.
        if target is "flags":
            out_arr[:,:,range(self.nants),range(self.nants),:] = FL.MISSING
        else:
            out_arr[:,:,range(self.nants),range(self.nants),:] = 0

        # The final step here reshapes the output to ensure compatability
        # with code elsewhere. 

        if target is "obser":
            out_arr = out_arr.reshape([chunk_tdim, chunk_fdim, self.nants, self.nants, 2, 2])

        elif target is "flags":
            out_arr = np.bitwise_or.reduce(out_arr, axis=-1)

        elif target is "model":
            out_arr = out_arr.reshape([1, chunk_tdim, chunk_fdim, self.nants, self.nants, 2, 2])
        elif target is "weigh":
            out_arr = np.sqrt( np.sum(out_arr, axis=-1) )  # take the square root

        return out_arr


    def arr_to_col(self, in_arr, bounds):
        """
        Converts the calibrated measurement matrix back into the MS style.

        Args:
            in_arr (np.array): Input array which is to be made MS friendly.
            f_t_row (int): First time row in MS to which the data belongs.
            l_t_row (int): Last time row in MS to which the data belongs.
            f_f_col (int): First frequency in MS to which the data belongs.
            l_f_col (int): Last frequency in MS to which the data belongs.

        """

        ddid, timechunk, f_f_col, l_f_col = bounds

        rows = self.chunk_rind[ddid,timechunk]
        new_shape = [len(rows), l_f_col - f_f_col, 4]

        tchunk = self.times[rows]
        tchunk -= tchunk[0]        # is this correct -- does in_array start from beginning of chunk?
        achunk = self.antea[rows]
        bchunk = self.anteb[rows]

        if self.ncorr==4:
            self.covis[rows, f_f_col:l_f_col, :] = \
                in_arr[tchunk, :, achunk, bchunk, :].reshape(new_shape)
        elif self.ncorr==2:
            self.covis[rows, f_f_col:l_f_col, :] = \
                in_arr[tchunk, :, achunk, bchunk, :].reshape(new_shape)[:,:,::3]
        elif self.ncorr==1:
            self.covis[rows, f_f_col:l_f_col, :] = \
                in_arr[tchunk, :, achunk, bchunk, :].reshape(new_shape)[:,:,::4]


    def add_to_gain_dict(self, gains, bounds, t_int=1, f_int=1):

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

        if output_name is None:
            output_name = self.ms_name + "/gains.p"

        cPickle.dump(self.gain_dict, open(output_name, "wb"), protocol=2)


    def save(self, values, col_name, like_col="DATA"):
        """
        Saves values to column in MS.

        Args
        values (np.array): Values to be written to column.
        col_name (str): Name of target column.
        like_col (str): If new column needs to be inserted, it will be patterned on the named column.
        """
        if col_name not in self.ms.colnames():
            # new column needs to be inserted -- get column description from column 'like_col'
            print>>log, "  inserting new column %s" % (col_name)
            desc = self.ms.getcoldesc(like_col)
            desc["name"] = col_name
            desc['comment'] = desc['comment'].replace(" ","_")  # got this from Cyril, not sure why
            self.ms.addcols(desc)
            # close and re-open MS. Otherwise the putcol() call below bombs out. (The python table object Somehow
            # doesn't understand that the underlying column has been added.)
            self.ms.close()
            self.ms = pt.table(self.ms_name, readonly=False, ack=False)
            if self.data is not self.ms:
                self.data = self.ms.query(self.taql)

        self.data.putcol(col_name, values)
