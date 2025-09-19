"""
Implementation of the box-least squares periodogram [K2002]_
and variants.

.. [K2002] `Kovacs et al. 2002 <http://adsabs.harvard.edu/abs/2002A%26A...391..369K>`_

Modified from: https://github.com/johnh2o2/cuvarbase/ released under the GPL3.0 license.
"""
from __future__ import print_function, division

from builtins import zip
from builtins import range
import sys

import cupy as cp

import numpy as np

from pathlib import Path

_default_block_size = 256
_all_function_names = ['full_bls_no_sol',
                       'bin_and_phase_fold_custom',
                       'reduction_max',
                       'store_best_sols',
                       'store_best_sols_custom',
                       'bin_and_phase_fold_bst_multifreq',
                       'binned_bls_bst']


def _module_reader(fname, cpp_defs=None):
    with open(fname, 'r') as f:
        txt = f.read()

    if cpp_defs is None:
        return txt

    preamble = ['#define {key} {value}'.format(key=key,
                                               value=('' if value is None
                                                      else value))
                for key, value in cpp_defs.items()]
    txt = txt.replace('//{CPP_DEFS}', '\n'.join(preamble))

    return txt

    
def _reduction_max(max_func, arr, arr_args, nfreq, nbins,
                   stream, final_arr, final_argmax_arr,
                   final_index, block_size):
    """
    Perform max reduction over bins for all frequencies using CuPy kernels.

    Parameters:
    - max_func: CuPy RawKernel for max reduction
    - arr, arr_args: CuPy arrays, input arrays for reduction
    - nfreq: number of frequencies
    - nbins: number of bins initially
    - stream: CuPy stream to launch kernels on
    - final_arr, final_argmax_arr: CuPy arrays to store final max and argmax results
    - final_index: index offset for final result storage
    - block_size: CUDA block size (must be power of two)
    """
    # assert power of 2
    assert(block_size - 2 * (block_size / 2) == 0), "block_size must be power of two"

    block = (block_size, 1, 1)

    grid_size = int(np.ceil(nbins / block_size)) * nfreq
    nbins0 = nbins

    init = np.uint32(1)

    while grid_size > nfreq:
        max_func(grid=(grid_size, 1),
                   block=block,
                   stream=stream,
                   args=(arr, arr_args,
                         np.uint32(nfreq), np.uint32(nbins0), np.uint32(nbins),
                         arr, arr_args,
                         np.uint32(0), init))

        init = np.uint32(0)

        nbins0 = grid_size // nfreq
        grid_size = int(np.ceil(nbins0 / block_size)) * nfreq

    max_func(grid=(grid_size, 1),
               block=block,
               stream=stream,
               args=(arr, arr_args,
                     np.uint32(nfreq), np.uint32(nbins0), np.uint32(nbins),
                     final_arr, final_argmax_arr,
                     np.uint32(final_index), init))


def fmin_transit(t, rho=1., min_obs_per_transit=5, **kwargs):
    T = max(t) - min(t)
    qmin = float(min_obs_per_transit) / len(t)

    fmin1 = freq_transit(qmin, rho=rho)
    fmin2 = 2./(max(t) - min(t))
    return max([fmin1, fmin2])


def fmax_transit0(rho=1., **kwargs):
    return 8.6307 * np.sqrt(rho)


def q_transit(freq, rho=1., **kwargs):
    fmax0 = fmax_transit0(rho=rho)

    f23 = np.power(freq / fmax0, 2./3.)
    f23 = np.minimum(1., f23)
    return np.arcsin(f23) / np.pi


def freq_transit(q, rho=1., **kwargs):
    fmax0 = fmax_transit0(rho=rho)
    return fmax0 * (np.sin(np.pi * q) ** 1.5)


def fmax_transit(rho=1., qmax=0.5, **kwargs):
    fmax0 = fmax_transit0(rho=rho)
    return min([fmax0, freq_transit(qmax, rho=rho, **kwargs)])


def transit_autofreq(t, fmin=None, fmax=None, samples_per_peak=2,
                     rho=1., qmin_fac=0.2, qmax_fac=None, **kwargs):
    """
    Produce list of frequencies for a given frequency range
    suitable for performing Keplerian BLS.

    Parameters
    ----------
    t: array_like, float
        Observation times.
    fmin: float, optional (default: ``None``)
        Minimum frequency. By default this is determined by ``fmin_transit``.
    fmax: float, optional (default: ``None``)
        Maximum frequency. By default this is determined by ``fmax_transit``.
    samples_per_peak: float, optional (default: 2)
        Oversampling factor. Frequency spacing is multiplied by
        ``1/samples_per_peak``.
    rho: float, optional (default: 1)
        Mean stellar density of host star in solar units
        :math:`\\rho=\\rho_{\\star} / \\rho_{\\odot}`, where
        :math:`\\rho_{\\odot}`
        is the mean density of the sun
    qmin_fac: float, optional (default: 0.2)
        The minimum :math:`q` value to search in units of the Keplerian
        :math:`q` value
    qmax_fac: float, optional (default: None)
        The maximum :math:`q` value to search in units of the Keplerian
        :math:`q` value. If ``None``, this defaults to ``1/qmin_fac``.

    Returns
    -------
    freqs: array_like
        The frequency grid
    q0vals: array_like
        The list of Keplerian :math:`q` values.

    """
    if qmax_fac is None:
        qmax_fac = 1./qmin_fac

    if fmin is None:
        fmin = fmin_transit(t, rho=rho, samples_per_peak=samples_per_peak,
                            **kwargs)
    if fmax is None:
        fmax = fmax_transit(rho=rho, **kwargs)

    T = max(t) - min(t)
    freqs = [fmin]
    while freqs[-1] < fmax:
        df = qmin_fac * q_transit(freqs[-1], rho=rho) / (samples_per_peak * T)
        freqs.append(freqs[-1] + df)
    freqs = np.array(freqs)
    q0vals = q_transit(freqs, rho=rho)
    return freqs, q0vals

def compile_bls(block_size=_default_block_size,
                function_names=_all_function_names,
                **kwargs):
    """
    Compile BLS kernel using CuPy

    Parameters
    ----------
    block_size: int, optional
        CUDA threads per block (passed into the kernel via macro).
    function_names: list, optional
        Kernel function names to extract.

    Returns
    -------
    functions: dict
        Dictionary mapping function name to CuPy RawKernel object.
    """

    # Load kernel source
    cppd = dict(BLOCK_SIZE=block_size)
    kernel_path = Path(__file__).parent / 'gpu_bls.cu'
    kernel_txt = _module_reader(kernel_path, cpp_defs=cppd)

    # Compile the kernel source using CuPy's RawModule
    module = cp.RawModule(code=kernel_txt, options=('--use_fast_math',), name_expressions=function_names)

    # Extract function objects
    functions = {name: module.get_function(name) for name in function_names}

    return functions


class BLSMemory(object):
    def __init__(self, max_ndata, max_nfreqs, stream=None, **kwargs):
        self.max_ndata = max_ndata
        self.max_nfreqs = max_nfreqs
        self.t = None
        self.yw = None
        self.w = None

        self.t_g = None
        self.yw_g = None
        self.w_g = None

        self.freqs = None
        self.freqs_g = None

        self.qmin = None
        self.nbins0_g = None
        self.qmax = None
        self.nbinsf_g = None

        self.bls = None
        self.bls_g = None

        self.rtype = cp.float32
        self.stream = stream

        self.allocate_pinned_arrays(nfreqs=max_nfreqs, ndata=max_ndata)

    def allocate_pinned_arrays(self, nfreqs=None, ndata=None):
        if nfreqs is None:
            nfreqs = int(self.max_nfreqs)
        if ndata is None:
            ndata = int(self.max_ndata)

        # Host-pinned memory using CuPy
        self.bls = cp.cuda.alloc_pinned_memory(nfreqs * cp.dtype(self.rtype).itemsize)
        self.bls = np.frombuffer(self.bls, dtype=self.rtype, count=nfreqs)
        
        self.nbins0 = cp.cuda.alloc_pinned_memory(nfreqs * np.dtype(np.int32).itemsize)
        self.nbins0 = np.frombuffer(self.nbins0, dtype=np.int32, count=nfreqs)

        self.nbinsf = cp.cuda.alloc_pinned_memory(nfreqs * np.dtype(np.int32).itemsize)
        self.nbinsf = np.frombuffer(self.nbinsf, dtype=np.int32, count=nfreqs)

        self.t = cp.cuda.alloc_pinned_memory(ndata * cp.dtype(self.rtype).itemsize)
        self.t = np.frombuffer(self.t, dtype=self.rtype, count=ndata)

        self.yw = cp.cuda.alloc_pinned_memory(ndata * cp.dtype(self.rtype).itemsize)
        self.yw = np.frombuffer(self.yw, dtype=self.rtype, count=ndata)

        self.w = cp.cuda.alloc_pinned_memory(ndata * cp.dtype(self.rtype).itemsize)
        self.w = np.frombuffer(self.w, dtype=self.rtype, count=ndata)

    def allocate_freqs(self, nfreqs=None):
        if nfreqs is None:
            nfreqs = self.max_nfreqs

        self.freqs_g = cp.zeros(nfreqs, dtype=self.rtype)
        self.bls_g = cp.zeros(nfreqs, dtype=self.rtype)
        self.nbins0_g = cp.zeros(nfreqs, dtype=cp.uint32)
        self.nbinsf_g = cp.zeros(nfreqs, dtype=cp.uint32)

    def allocate_data(self, ndata=None):
        if ndata is None:
            ndata = len(self.t)
        self.t_g = cp.zeros(ndata, dtype=self.rtype)
        self.yw_g = cp.zeros(ndata, dtype=self.rtype)
        self.w_g = cp.zeros(ndata, dtype=self.rtype)

    def transfer_data_to_gpu(self, transfer_freqs=True):
        if self.stream is None:
            self.t_g.set(self.t)
            self.yw_g.set(self.yw)
            self.w_g.set(self.w)

            if transfer_freqs:
                self.freqs_g.set(self.freqs)
                self.nbins0_g.set(self.nbins0)
                self.nbinsf_g.set(self.nbinsf)

        else:
            with self.stream:
                self.t_g.set(self.t)
                self.yw_g.set(self.yw)
                self.w_g.set(self.w)

                if transfer_freqs:
                    self.freqs_g.set(self.freqs)
                    self.nbins0_g.set(self.nbins0)
                    self.nbinsf_g.set(self.nbinsf)

    def transfer_data_to_cpu(self):
        if self.stream is None:
            self.bls_g.get(out=self.bls)
        else:
            with self.stream:
                self.bls_g.get(out=self.bls)
            self.stream.synchronize()
            
        self.bls /= self.yy

    def setdata(self, t, y, dy, qmin=None, qmax=None,
                freqs=None, nf=None, transfer=True,
                **kwargs):

        if freqs is not None:
            self.freqs = np.asarray(freqs).astype(self.rtype)
            self.nbinsf = (np.ones_like(self.freqs) / qmin).astype(np.uint32)
            self.nbins0 = (np.ones_like(self.freqs) / qmax).astype(np.uint32)

        self.t[:len(t)] = np.asarray(t).astype(self.rtype)

        w = np.power(dy, -2)
        w /= sum(w)
        self.w[:len(t)] = w.astype(self.rtype)

        self.ybar = sum(y * w)
        self.yy = np.dot(w, np.power(y - self.ybar, 2))

        u = (y - self.ybar) * w
        self.yw[:len(t)] = u.astype(self.rtype)

        if any(x is None for x in [self.t_g, self.yw_g, self.w_g]):
            self.allocate_data()

        if self.freqs_g is None:
            if nf is None:
                nf = len(freqs)
            self.allocate_freqs(nfreqs=nf)

        if transfer:
            self.transfer_data_to_gpu(transfer_freqs=(freqs is not None))

        return self

    @classmethod
    def fromdata(cls, t, y, dy, qmin=None, qmax=None,
                 freqs=None, nf=None, transfer=True,
                 **kwargs):
        max_ndata = kwargs.get('max_ndata', len(t))
        max_nfreqs = kwargs.get('max_nfreqs', nf if freqs is None
                                else len(freqs))
        c = cls(max_ndata, max_nfreqs, **kwargs)

        return c.setdata(t, y, dy, qmin=qmin, qmax=qmax,
                         freqs=freqs, nf=nf, transfer=transfer,
                         **kwargs)

def eebls_gpu_fast(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
                   ignore_negative_delta_sols=False,
                   functions=None, stream=None, dlogq=0.3,
                   memory=None, noverlap=2, max_nblocks=5000,
                   force_nblocks=None, dphi=0.0,
                   shmem_lim=None, freq_batch_size=None,
                   transfer_to_device=True,
                   transfer_to_host=True, **kwargs):
    """
    Box-Least Squares with PyCUDA but about 2-3 orders of magnitude
    faster than eebls_gpu. Uses shared memory for the binned data,
    which means that there is a lower limit on the q values that
    this function can handle.

    To save memory and improve speed, the best solution is not
    kept. To get the best solution, run ``eebls_gpu`` at the
    optimal frequency.

    .. warning::

        If you are running on a single-GPU machine, there may be a
        kernel time limit set by your OS. If running this function
        produces a timeout error, try setting ``freq_batch_size`` to a
        reasonable number (~10). That will split up the computations by
        frequency.

    .. note::

        No extra global memory is needed, meaning you likely do *not* need
        to use ``large_run`` with this function.

    .. note::

        There is no ``noverlap`` parameter here yet. This is only a problem
        if the optimal ``q`` value is close to ``qmin``. To alleviate this,
        you can run this function ``noverlap`` times with
        ``dphi = i/noverlap`` for the ``i``-th run. Then take the best solution
        of all runs.

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    freqs: array_like, float
        Frequencies
    qmin: float or array_like, optional (default: 1e-2)
        minimum q values to search at each frequency
    qmax: float or array_like (default: 0.5)
        maximum q values to search at each frequency
    ignore_negative_delta_sols: bool
        Whether or not to ignore solutions with a negative delta (i.e. an inverted dip)
    dphi: float, optional (default: 0.)
        Phase offset (in units of the finest grid spacing). If you
        want ``noverlap`` bins at the smallest ``q`` value, run this
        function ``noverlap`` times, with ``dphi = i / noverlap``
        for the ``i``-th run and take the best solution for all the runs.
    dlogq: float
        The logarithmic spacing of the q values to use. If negative,
        the q values increase by ``dq = qmin``.
    functions: dict
        Dictionary of compiled functions (see :func:`compile_bls`)
    freq_batch_size: int, optional (default: None)
        Number of frequencies to compute in a single batch; if
        ``None`` this will run a single batch for all frequencies
        simultaneously
    shmem_lim: int, optional (default: None)
        Maximum amount of shared memory to use per block in bytes.
        This is GPU-dependent but usually around 48KB. If ``None``,
        uses device information provided by PyCUDA (recommended).
    max_nblocks: int, optional (default: 200)
        Maximum grid size to use
    force_nblocks: int, optional (default: None)
        If this is set the gridsize is forced to be this value
    memory: :class:`BLSMemory` instance, optional (default: None)
        See :class:`BLSMemory`.
    transfer_to_host: bool, optional (default: True)
        Transfer BLS back to CPU.
    transfer_to_device: bool, optional (default: True)
        Transfer data to GPU
    **kwargs:
        passed to `compile_bls`

    Returns
    -------
    bls: array_like, float
        BLS periodogram, normalized to
        :math:`1 - \chi_2(\omega) / \chi_2(constant)`

    """

    fname = 'full_bls_no_sol'

    if functions is None:
        functions = compile_bls(function_names=[fname], **kwargs)
    
    func = functions[fname]

    if shmem_lim is None:
        shmem_lim = cp.cuda.Device().attributes["MaxSharedMemoryPerBlock"]

    if memory is None:
        memory = BLSMemory.fromdata(t, y, dy, qmin=qmin, qmax=qmax,
                                    freqs=freqs, stream=stream,
                                    transfer=True, **kwargs)
    elif transfer_to_device:
        memory.setdata(t, y, dy, qmin=qmin, qmax=qmax,
                       freqs=freqs, transfer=True, **kwargs)

    float_size = np.float32(1).nbytes
    block_size = kwargs.get('block_size', _default_block_size)
    
    if freq_batch_size is None:
        freq_batch_size = len(freqs)
        
    block = (block_size, 1, 1)

    # minimum q value that we can handle with the shared memory limit
    qmin_min = 2 * float_size / (shmem_lim - float_size * block_size)
    i_freq = 0

    while i_freq < len(freqs):
        j_freq = min(i_freq + freq_batch_size, len(freqs))
        nfreqs = j_freq - i_freq
        max_nbins = max(memory.nbinsf[i_freq:j_freq])
        mem_req = (block_size + 2 * max_nbins) * float_size

        if mem_req > shmem_lim:
            raise RuntimeError(
                f"qmin = {1./max_nbins:.2e} requires too much shared memory.\n"
                f"Either try a larger value of qmin (> {qmin_min:.2e}) "
                "or avoid using eebls_gpu_fast."
            )

        nblocks = min([nfreqs, max_nblocks])
        if force_nblocks is not None:
            nblocks = force_nblocks
            
        grid = (nblocks, 1)

        args = (
            memory.t_g, memory.yw_g, memory.w_g,
            memory.bls_g, memory.freqs_g,
            memory.nbins0_g, memory.nbinsf_g,
            np.uint32(len(t)), np.uint32(nfreqs), np.uint32(i_freq),
            np.uint32(max_nbins), np.uint32(noverlap),
            np.float32(dlogq), np.float32(dphi),
            np.uint32(ignore_negative_delta_sols)
        )

        func(grid, block, args, stream=stream, shared_mem=int(mem_req))
        i_freq = j_freq

    if transfer_to_host:
        memory.transfer_data_to_cpu()
        if stream is not None:
            stream.synchronize()

    return memory.bls


def dnbins(nbins, dlogq):
    if (dlogq < 0):
        return 1

    n = int(np.floor(dlogq * nbins))

    return n if n > 0 else 1


def nbins_iter(i, nb0, dlogq):
    nb = nb0
    for j in range(i):
        nb += dnbins(nb, dlogq)

    return nb


def count_tot_nbins(nbins0, nbinsf, dlogq):
    ntot = 0

    i = 0
    while nbins_iter(i, nbins0, dlogq) <= nbinsf:
        ntot += nbins_iter(i, nbins0, dlogq)
        i += 1
    return ntot


def eebls_gpu(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
              ignore_negative_delta_sols=False,
              nstreams=5, noverlap=3, dlogq=0.2, max_memory=None,
              freq_batch_size=None, functions=None, **kwargs):

    """
    Box-Least Squares, accelerated with PyCUDA

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    freqs: array_like, float
        Frequencies
    qmin: float or array_like
        Minimum q value(s) to test for each frequency
    qmax: float or array_like
        Maximum q value(s) to test for each frequency
    ignore_negative_delta_sols: bool
        Whether or not to ignore solutions with a negative delta (i.e. an inverted dip)
    nstreams: int, optional (default: 5)
        Number of CUDA streams to utilize.
    noverlap: int, optional (default: 3)
        Number of overlapping q bins to use
    dlogq: float, optional, (default: 0.5)
        logarithmic spacing of :math:`q` values, where :math:`d\log q = dq / q`
    freq_batch_size: int, optional (default: None)
        Number of frequencies to compute in a single batch; determines
        this automatically based on ``max_memory``
    max_memory: float, optional (default: None)
        Maximum memory to use in bytes. Will ignore this if
        ``freq_batch_size`` is specified, and will use the total free memory
        as returned by ``pycuda.driver.mem_get_info`` if this is ``None``.
    functions: tuple of CUDA functions
        returned by ``compile_bls``

    Returns
    -------
    bls: array_like, float
        BLS periodogram, normalized to :math:`1 - \chi^2(f) / \chi^2_0`
    qphi_sols: list of ``(q, phi)`` tuples
        Best ``(q, phi)`` solution at each frequency

    """

    def locext(ext, arr, imin=None, imax=None):
        if isinstance(arr, (float, int)):
            return arr
        return ext(arr[slice(imin, imax)])

    if functions is None:
        functions = compile_bls(**kwargs)

    ndata = len(t)
    
    # Estimate memory usage
    if freq_batch_size is None:
        if max_memory is None:
            free = cp.cuda.runtime.memGetInfo()[0]
            max_memory = int(0.9 * free)

        real_type_size = np.float32(1).nbytes
        
        mem0 = ndata * 3 * real_type_size + len(freqs) * 5 * real_type_size
        nbins0_max = int(np.floor(1./locext(max, qmax)))
        nbinsf_max = int(np.ceil(1./locext(min, qmin)))
        nbins_tot_max = count_tot_nbins(nbins0_max, nbinsf_max, dlogq)

        mem_per_f = 4 * nstreams * nbins_tot_max * noverlap * real_type_size
        freq_batch_size = int((max_memory - mem0) / mem_per_f)
        if freq_batch_size <= 0:
            raise MemoryError("Not enough memory to allocate even one batch.")

    # Allocate memory
    bls_mem = BLSMemory(len(t), len(freqs), rtype=cp.float32)
    bls_mem.setdata(t, y, dy, qmin=qmin, qmax=qmax, freqs=freqs, transfer=True)

    # Prepare temp GPU arrays
    gs = freq_batch_size * count_tot_nbins(
        int(np.floor(1./locext(max, qmax))),
        int(np.ceil(1./locext(min, qmin))),
        dlogq
    ) * noverlap

    block_size = kwargs.get('block_size', _default_block_size)

    # Temp arrays per stream
    streams = [cp.cuda.Stream() for _ in range(nstreams)]
    yw_g_bins = [cp.zeros(gs, dtype=cp.float32) for _ in range(nstreams)]
    w_g_bins = [cp.zeros(gs, dtype=cp.float32) for _ in range(nstreams)]
    bls_tmp_gs = [cp.zeros(gs, dtype=cp.float32) for _ in range(nstreams)]
    bls_tmp_sol_gs = [cp.zeros(gs, dtype=cp.int32) for _ in range(nstreams)]

    bls_g = cp.zeros(len(freqs), dtype=cp.float32)
    bls_sol_g = cp.zeros(len(freqs), dtype=cp.int32)
    bls_best_phi = cp.zeros(len(freqs), dtype=cp.float32)
    bls_best_q = cp.zeros(len(freqs), dtype=cp.float32)

    bin_func = functions['bin_and_phase_fold_bst_multifreq']
    bls_func = functions['binned_bls_bst']
    max_func = functions['reduction_max']
    store_func = functions['store_best_sols']

    YY = bls_mem.yy
    nbatches = int(np.ceil(len(freqs) / freq_batch_size))

    for batch in range(nbatches):
        imin = batch * freq_batch_size
        imax = min((batch + 1) * freq_batch_size, len(freqs))
        nf = imax - imin
        j = batch % nstreams

        stream = streams[j]
        with stream:
            minq = locext(min, qmin, imin, imax)
            maxq = locext(max, qmax, imin, imax)
            nbins0 = int(np.floor(1./maxq))
            nbinsf = int(np.ceil(1./minq))
            nbins_tot = count_tot_nbins(nbins0, nbinsf, dlogq)

            all_bins = nf * nbins_tot * noverlap

            # Reset temp buffers
            yw_g_bins[j].fill(0)
            w_g_bins[j].fill(0)
            bls_tmp_gs[j].fill(0)
            bls_tmp_sol_gs[j].fill(0)

            # Launch binning kernel
            bin_func(
                (int(np.ceil(ndata * nf / block_size)),), (block_size,),
                (
                    bls_mem.t_g, bls_mem.yw_g, bls_mem.w_g,
                    yw_g_bins[j], w_g_bins[j], bls_mem.freqs_g,
                    np.int32(ndata), np.int32(nf),
                    np.int32(nbins0), np.int32(nbinsf),
                    np.int32(imin), np.int32(noverlap),
                    np.float32(dlogq), np.int32(nbins_tot)
                ),
                stream=stream
            )

            # BLS kernel
            bls_func(
                (int(np.ceil(all_bins / block_size)),), (block_size,),
                (
                    yw_g_bins[j], w_g_bins[j], bls_tmp_gs[j],
                    np.int32(all_bins),
                    np.uint32(ignore_negative_delta_sols)
                ),
                stream=stream
            )

            # Reduction kernel
            _reduction_max(max_func,
                bls_tmp_gs[j], bls_tmp_sol_gs[j],
                nf, nbins_tot * noverlap, stream,
                bls_g, bls_sol_g, np.int32(imin),
                block_size
            )

            # Store best solutions
            store_func(
                (int(np.ceil(nf / block_size)),), (block_size,),
                (
                    bls_sol_g, bls_best_phi, bls_best_q,
                    np.uint32(nbins0), np.uint32(nbinsf),
                    np.uint32(noverlap), np.float32(dlogq),
                    np.uint32(nf), np.uint32(imin)
                ),
                stream=stream
            )

    cp.cuda.Device().synchronize()
    qphi_sols = list(zip(cp.asnumpy(bls_best_q), cp.asnumpy(bls_best_phi)))
    return cp.asnumpy(bls_g) / YY, qphi_sols


def eebls_transit_gpu(t, y, dy, fmax_frac=1.0, fmin_frac=1.0,
                      qmin_fac=0.5, qmax_fac=2.0, fmin=None,
                      fmax=None, freqs=None, qvals=None, use_fast=False,
                      ignore_negative_delta_sols=False,
                      **kwargs):
    """
    Compute BLS for timeseries assuming edge-on keplerian
    orbit of a planet with Mp/Ms << 1, Rp/Rs < 1, Lp/Ls << 1 and
    negligible eccentricity.

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    fmax_frac: float, optional (default: 1.0)
        Maximum frequency is `fmax_frac * fmax`, where
        `fmax` is automatically selected by `fmax_transit`.
    fmin_frac: float, optional (default: 1.5)
        Minimum frequency is `fmin_frac * fmin`, where
        `fmin` is automatically selected by `fmin_transit`.
    fmin: float, optional (default: None)
        Overrides automatic frequency minimum with this value
    fmax: float, optional (default: None)
        Overrides automatic frequency maximum with this value
    qmin_fac: float, optional (default: 0.5)
        Fraction of the fiducial q value to search
        at each frequency (minimum)
    qmax_fac: float, optional (default: 2.0)
        Fraction of the fiducial q value to search
        at each frequency (maximum)
    freqs: array_like, optional (default: None)
        Overrides the auto-generated frequency grid
    qvals: array_like, optional (default: None)
        Overrides the keplerian q values
    functions: tuple, optional (default=None)
        result of ``compile_bls(**kwargs)``.
    use_fast: bool, optional (default: False)

    ignore_negative_delta_sols: bool
        Whether or not to ignore inverted dips
    **kwargs:
        passed to `eebls_gpu`, `compile_bls`, `fmax_transit`,
        `fmin_transit`, and `transit_autofreq`


    Returns
    -------
    freqs: array_like, float
        Frequencies where BLS is evaluated
    bls: array_like, float
        BLS periodogram, normalized to :math:`1 - \chi^2(f) / \chi^2_0`
    solutions: list of ``(q, phi)`` tuples
        Best ``(q, phi)`` solution at each frequency

        .. note::

            Only returned when ``use_fast=False``.

    """

    if freqs is None:
        if qvals is not None:
            raise Exception("qvals must be None if freqs is None")
        if fmin is None:
            fmin = fmin_transit(t, **kwargs) * fmin_frac
        if fmax is None:
            fmax = fmax_transit(qmax=0.5 / qmax_fac, **kwargs) * fmax_frac
        freqs, qvals = transit_autofreq(t, fmin=fmin, fmax=fmax,
                                        qmin_fac=qmin_fac, **kwargs)
    if qvals is None:
        qvals = q_transit(freqs, **kwargs)

    qmins = qvals * qmin_fac
    qmaxes = qvals * qmax_fac
    
    # tdur_hrs = (qmins/freqs)*24
    
    # mask = tdur_hrs < 1.5
    
    # qmins[mask] = (1.5/24)*freqs[mask]
    
    if use_fast:
        powers = eebls_gpu_fast(t, y, dy, freqs,
                                qmin=qmins, qmax=qmaxes,
                                ignore_negative_delta_sols=ignore_negative_delta_sols,
                                **kwargs)

        return freqs, powers, qmins, qmaxes

    powers, sols = eebls_gpu(t, y, dy, freqs,
                             qmin=qmins, qmax=qmaxes,
                             ignore_negative_delta_sols=ignore_negative_delta_sols,
                             **kwargs)
    return freqs, powers, sols

