"""Backend selection for simpleinfo.fastinfo."""

from __future__ import annotations

from importlib.util import find_spec

if find_spec("numba") is None:
    from . import _fallback as _impl
    BACKEND = "numpy"
else:
    from . import _numba as _impl
    BACKEND = "numba"


def calcinfo(x, xb, y, yb, *, bias=False, validate=True, threads=None):
    return _impl.calcinfo(x, xb, y, yb, bias=bias, validate=validate, threads=threads)


def calcinfomatched(x, xb, y, yb, *, bias=False, validate=True, threads=None):
    return _impl.calcinfomatched(x, xb, y, yb, bias=bias, validate=validate, threads=threads)


def calccmi(x, xb, y, yb, z, zb, *, bias=False, validate=True, threads=None):
    return _impl.calccmi(x, xb, y, yb, z, zb, bias=bias, validate=validate, threads=threads)


def calccondcmi(x, xb, y, yb, z, zb, k, kb, *, validate=True, threads=None):
    return _impl.calccondcmi(x, xb, y, yb, z, zb, k, kb, validate=validate, threads=threads)


def calcinfoperm(x, xb, y, yb, nperm, *, bias=False, validate=True, threads=None, rng=None, seed=None):
    return _impl.calcinfoperm(
        x, xb, y, yb, nperm, bias=bias, validate=validate, threads=threads, rng=rng, seed=seed
    )


def calcinfoperm_slice(x, xb, y, yb, nperm, *, bias=False, validate=True, threads=None, seed=None):
    return _impl.calcinfoperm_slice(
        x, xb, y, yb, nperm, bias=bias, validate=validate, threads=threads, seed=seed
    )


def calcinfo_slice(x, xb, y, yb, *, bias=False, validate=True, threads=None):
    return _impl.calcinfo_slice(x, xb, y, yb, bias=bias, validate=validate, threads=threads)


def calccmi_slice(x, xb, y, yb, z, zb, *, bias=False, validate=True, threads=None):
    return _impl.calccmi_slice(x, xb, y, yb, z, zb, bias=bias, validate=validate, threads=threads)


def eqpop(x, nb, *, validate=True, warn_on_ties=True):
    return _impl.eqpop(x, nb, validate=validate, warn_on_ties=warn_on_ties)


def eqpop_slice(x, nb, *, validate=True, warn_on_ties=True, threads=None):
    return _impl.eqpop_slice(x, nb, validate=validate, warn_on_ties=warn_on_ties, threads=threads)


def eqpop_sorted(x_sorted, nb, *, validate=True, warn_on_ties=True):
    return _impl.eqpop_sorted(x_sorted, nb, validate=validate, warn_on_ties=warn_on_ties)


def eqpop_sorted_slice(x_sorted, nb, *, validate=True, warn_on_ties=True, threads=None):
    return _impl.eqpop_sorted_slice(x_sorted, nb, validate=validate, warn_on_ties=warn_on_ties, threads=threads)


def calcpairwiseinfo(x, xb, y, yb, *, bias=False, validate=True):
    return _impl.calcpairwiseinfo(x, xb, y, yb, bias=bias, validate=validate)


def calcpairwiseinfo_slice(x, xb, y, yb, *, bias=False, validate=True):
    return _impl.calcpairwiseinfo_slice(x, xb, y, yb, bias=bias, validate=validate)


def set_threads(threads):
    if hasattr(_impl, "set_threads"):
        return _impl.set_threads(threads)
    return None
