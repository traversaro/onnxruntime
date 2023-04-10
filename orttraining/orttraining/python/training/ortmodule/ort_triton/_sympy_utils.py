# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import re
from typing import List

import sympy



def sympy_symbol(name):
    if isinstance(name, int):
        return sympy.Integer(name)
    if isinstance(name, list):
        return [sympy_symbol(x) for x in name]
    if isinstance(name, str):
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    return sympy.Symbol(name, integer=True, positive=True)


def sympy_dot(seq1, seq2):
    assert len(seq1) == len(seq2)
    return sympy.expand(sum(a * b for a, b in zip(seq1, seq2)))


def parse_shape(shape: List):
    symbol_shapes = []
    for dim in shape:
        symbol_dim = dim
        # TODO: str (such as "1024") can be parsed to int?
        if isinstance(dim, str):
            symbol_dim = sympy.Symbol(re.sub(r"[^a-zA-Z0-9_]+", "_", dim))
        elif isinstance(dim, int):
            symbol_dim = sympy.Integer(dim)
        symbol_shapes.append(symbol_dim)
    return symbol_shapes
