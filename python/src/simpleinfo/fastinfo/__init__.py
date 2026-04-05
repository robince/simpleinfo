"""Optimized-style API surface for simpleinfo."""

from ._api import (
    calccmi,
    calccmi_slice,
    calccondcmi,
    calcinfo,
    calcinfomatched,
    calcinfoperm,
    calcinfoperm_slice,
    calcinfo_slice,
    calcpairwiseinfo,
    calcpairwiseinfo_slice,
    eqpop,
    eqpop_slice,
    eqpop_sorted,
    eqpop_sorted_slice,
    set_threads,
)

__all__ = [
    "calccmi",
    "calccmi_slice",
    "calccondcmi",
    "calcinfo",
    "calcinfomatched",
    "calcinfoperm",
    "calcinfoperm_slice",
    "calcinfo_slice",
    "calcpairwiseinfo",
    "calcpairwiseinfo_slice",
    "eqpop",
    "eqpop_slice",
    "eqpop_sorted",
    "eqpop_sorted_slice",
    "set_threads",
]
