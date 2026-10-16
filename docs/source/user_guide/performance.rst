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

If ``workers`` is not given, the ``MOSQITO_WORKERS`` environment variable is used,
and if that is not set either, one thread is used: **nothing runs in parallel
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
     - one per 200 ms segment, about 9 per second of signal at the default overlap

Beyond the number of physical cores of your machine, more workers bring nothing.
The filters and FFTs run in SciPy and NumPy, which release Python's global
interpreter lock, so ordinary threads are enough. Each thread of ``roughness_ecma``
holds the working arrays of one band, which grow with the signal duration.

Memory
------

``loudness_ecma`` and ``roughness_ecma`` filter, segment and reduce one band at a
time. Previously all 53 bands were held at once in an array repeating every sample
two to four times, about 2.9 GB for 20 s of signal in ``loudness_ecma``.

Measured speed
--------------

Measured with ``benchmarks/bench_metrics.py``, best of two runs, on a 4-core
machine. The comparison is with the previous release.

*(Table to be added with the benchmark results.)*
