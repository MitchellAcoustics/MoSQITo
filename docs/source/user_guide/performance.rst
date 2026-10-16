Performance
===========

A few metrics process many independent pieces of signal (frequency bands or time
segments). MoSQITo can run those on several threads, and keeps its memory use low
by handling one band at a time.

Using several threads
---------------------

``noct_spectrum``, ``loudness_ecma``, ``roughness_ecma`` and ``roughness_dw``
accept a ``workers`` argument::

    from mosqito.sq_metrics import loudness_ecma

    N, N_time, N_spec, bark_axis, time_axis = loudness_ecma(signal, fs, workers=4)

If ``workers`` is not given, one thread is used: **nothing runs in parallel
unless you ask for it**. This avoids oversubscribing the machine when you already
run MoSQITo from your own threads or processes. The results do not depend on the
number of workers.

How many workers are useful depends on the metric, because the work is split into
independent tasks and extra threads beyond the number of tasks stay idle:

.. list-table::
   :header-rows: 1

   * - Function
     - Independent tasks
   * - ``loudness_ecma``, ``roughness_ecma``
     - 53 frequency bands
   * - ``noct_spectrum``
     - one per band, about 28 for third octaves between 24 Hz and 12.5 kHz
   * - ``roughness_dw``
     - one per 200 ms segment, about 10 per second of signal at the default overlap

More workers than tasks have no effect, and beyond the number of physical cores
of your machine they bring nothing.
The filters and FFTs run in SciPy and NumPy, which release Python's global
interpreter lock, so ordinary threads are enough. Each thread of ``roughness_ecma``
holds the working arrays of one band, which grow with the signal duration.

Memory
------

``loudness_ecma`` and ``roughness_ecma`` filter, segment and reduce one band at a
time. Previously all 53 bands were held at once in an array repeating every sample
two to four times; the peak memory of ``loudness_ecma`` was about 2.9 GB for 20 s
of signal.

Measured speed
--------------

Measured with ``benchmarks/bench_metrics.py``, best of several runs (``--repeat``,
default 3), on a 4-core machine. The comparison is with the code before this
change. Peak memory is the peak traced by ``tracemalloc``, not the resident set
size of the process.

.. list-table:: 5 s of signal, time in seconds (peak memory in MB)
   :header-rows: 1

   * - Function
     - Before
     - Now, 1 worker
     - Now, 4 workers
   * - ``loudness_zwtv``
     - 10.4 (250)
     - 1.82 (212)
     - not applicable
   * - ``sharpness_din_tv``
     - 10.2 (250)
     - 1.64 (212)
     - not applicable
   * - ``loudness_ecma``
     - 1.59 (723)
     - 0.64 (18)
     - 0.19 (37)
   * - ``roughness_ecma``
     - 7.3 (2312)
     - 3.97 (143)
     - 1.45 (217)
   * - ``roughness_dw``, tonal signal
     - 1.62 (26)
     - 1.28 (41)
     - 0.38 (129)
   * - ``roughness_dw``, broadband noise
     - 19.8 (26)
     - 1.50 (46)
     - 0.45 (150)
   * - ``noct_spectrum``
     - 0.134 (6)
     - 0.105 (6)
     - 0.050 (23)

Timings on a shared machine vary by tens of percent between runs, so read them as
orders of magnitude. For 20 s of signal ``loudness_ecma`` went from 9.5 s to 3.4 s
with one worker and 0.8 s with four. Results are unchanged to within about 1e-12
relative.

Optional compilation with numba
-------------------------------

The time-varying Zwicker loudness (``loudness_zwtv``) and the sharpness computed
from it (``sharpness_din_tv``) spend most of their time in two loops that cannot
be vectorised, because each step depends on the previous one. Installing the
``performance`` extra compiles them with `numba <https://numba.pydata.org>`_::

    pip install "mosqito[performance]"

Results are identical to the plain Python ones (the compiled and plain versions
are tested to give exactly the same numbers), and nothing else needs to change.

* numba is only imported, and the loops only compiled, the first time a metric
  that uses them is computed. ``import mosqito`` is not slower.
* The first call compiles: about 1 s the very first time. The compiled code is
  cached on disk (in ``__pycache__`` next to the module, or in the directory given
  by the ``NUMBA_CACHE_DIR`` environment variable if that is not writable), so
  later Python sessions only pay a few tens of milliseconds to load it.
* Set ``MOSQITO_DISABLE_NUMBA=1`` to turn it off. If numba is not installed or does
  not work, MoSQITo silently uses plain Python.

Measured on a 4-core machine (best of runs), compared with the release before the
performance work:

.. list-table:: ``loudness_zwtv``, time in seconds
   :header-rows: 1

   * - Signal
     - Previous
     - Plain Python
     - With numba
   * - 1 s
     - 1.82
     - 0.34
     - 0.07
   * - 5 s
     - 9.8
     - 2.2
     - 0.41

For the two loops alone, on 2000 frames: the nonlinear decay takes 1815 ms
previously, 189 ms in plain Python and 16 ms compiled; the specific loudness
slopes take 137 ms, 148 ms and 4 ms. The slopes keep their vectorised
implementation when numba is not installed, because a plain Python loop is slower
than it.
