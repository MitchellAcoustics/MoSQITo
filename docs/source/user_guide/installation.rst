Installation
============

MOSQITO is available on pip. Simply type in a shell the following command: ::

  pip install mosqito

This command line should download and install MOSQITO on your computer, along with the dependencies needed to compute SQ metrics.

All SQ metrics computation functions are accessible via the following commands: ::

  from mosqito import <function> 

or ::

  from mosqito.<module> import <function>
  

Optional: faster time-varying loudness
--------------------------------------

Two loops of the Zwicker time-varying loudness (and of the sharpness computed from
it) cannot be vectorised. They can be compiled with `numba <https://numba.pydata.org>`_
if you install the ``performance`` extra: ::

  pip install "mosqito[performance]"

Nothing else changes: results are identical, and without numba the same
computation simply takes longer. See :doc:`performance` for the measured gain and
how to turn it off.
