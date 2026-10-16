# -*- coding: utf-8 -*-
"""Optional numba acceleration of the few loops that cannot be vectorised.

Install the extra to use it: ``pip install "mosqito[performance]"``. Without
numba everything still works and gives the same results, only slower.

numba is imported and the kernels are compiled on their first call, not when
mosqito is imported, so users who have numba installed do not pay for it
unless they compute a metric that uses it. Set ``MOSQITO_DISABLE_NUMBA=1`` to
turn it off.
"""
import functools
import importlib.util
import os
import threading
import warnings

DISABLE_ENV_VAR = "MOSQITO_DISABLE_NUMBA"


def _numba_wanted():
    if os.environ.get(DISABLE_ENV_VAR, "").strip().lower() in ("1", "true", "yes"):
        return False
    try:
        return importlib.util.find_spec("numba") is not None
    except (ImportError, ValueError):
        return False


# True if numba is installed and not disabled (checked without importing it). Set
# to False if it later turns out not to work.
HAS_NUMBA = _numba_wanted()


class _LazyJit:
    """A function that is compiled by numba on first call when numba is usable."""

    def __init__(self, func, options):
        functools.update_wrapper(self, func)
        self.py_func = func
        self._options = options
        self._compiled = None
        self._lock = threading.Lock()

    @property
    def is_compiled(self):
        """True once the function has been compiled by numba."""
        return self._compiled is not None and self._compiled is not self.py_func

    def _build(self):
        global HAS_NUMBA
        if not HAS_NUMBA:
            return self.py_func
        try:
            import numba

            try:
                return numba.njit(cache=True, nogil=True, **self._options)(self.py_func)
            except Exception:  # e.g. no writable cache location
                return numba.njit(nogil=True, **self._options)(self.py_func)
        except Exception as exc:
            HAS_NUMBA = False
            warnings.warn("numba could not be used (%s); falling back to plain Python." % exc)
            return self.py_func

    def __call__(self, *args, **kwargs):
        if self._compiled is None:
            with self._lock:
                if self._compiled is None:
                    self._compiled = self._build()
                    first = True
                else:
                    first = False
        else:
            first = False
        if first and self._compiled is not self.py_func:
            try:
                return self._compiled(*args, **kwargs)
            except Exception as exc:
                if not type(exc).__module__.startswith("numba"):
                    raise
                global HAS_NUMBA
                HAS_NUMBA = False
                self._compiled = self.py_func
                warnings.warn("numba failed to compile %s (%s); falling back to plain Python." % (self.py_func.__name__, exc))
        return self._compiled(*args, **kwargs)


def jit(**options):
    """Decorator: compile with numba (``njit``, cached, GIL released) when available."""

    def decorate(func):
        return _LazyJit(func, options)

    return decorate
