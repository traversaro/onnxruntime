# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import functools
import json
import os
import sys
import traceback
from types import ModuleType
from typing import List, Tuple

import onnx
from torch._C import _from_dlpack
from torch.utils.dlpack import to_dlpack

from onnxruntime.training import ortmodule

from ._codecache import PyCodeCache
from ._codegen import codegen
from ._op_config import get_supported_ops
from ._sorted_graph import SortedGraph
from ._sympy_utils import sympy_symbol
from ._utils import gen_unique_name

_DEBUG_MODE = ortmodule._defined_from_envvar("ORTMODULE_TRITON_DEBUG", 0) != 0


@functools.lru_cache(None)
def _gen_module(sorted_graph: SortedGraph) -> Tuple[str, str, ModuleType]:
    func_name = gen_unique_name("call")
    src_code = codegen(func_name, sorted_graph)
    return func_name, src_code, PyCodeCache().load(src_code)


class ModuleCache:
    cache = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def load(cls, onnx_key: int, onnx_str: bytes, shapes: List[List[int]]):
        key = hash(f"{onnx_key}|{str(shapes).replace(' ', '')}") % (10**8)
        if key not in cls.cache:
            model = onnx.load_model_from_string(onnx_str)
            sorted_graph = SortedGraph(model, [[sympy_symbol(dim) for dim in shape] for shape in shapes])
            func_name, src_code, mod = _gen_module(sorted_graph)
            if _DEBUG_MODE:
                file_name = f"{func_name}_{onnx_key}"
                py_file_path = f"triton_debug/{file_name}.py"
                os.makedirs(os.path.dirname(py_file_path), exist_ok=True)
                with open(py_file_path, "w") as f:
                    f.write(src_code)
                sorted_graph.save_onnx(f"triton_debug/{file_name}")
            cls.cache[key] = (func_name, mod)
        return cls.cache[key]


def get_config() -> str:
    config = {"ops": get_supported_ops(), "initializer": "scalar"}
    return json.dumps(config)


def execute_triton_op(func_name: str, onnx_key: int, onnx_str: bytes, *tensors):
    try:
        torch_tensors = [_from_dlpack(tensor) for tensor in tensors]
        if not onnx_str:
            assert func_name
            func = getattr(sys.modules[".".join(__name__.split(".")[:-1])], func_name)
        else:
            concrete_shapes = [list(tensor.size()) for tensor in torch_tensors]
            func_name, mod = ModuleCache.load(onnx_key, onnx_str, concrete_shapes)
            func = getattr(mod, func_name)
        output = func(*torch_tensors)
    except Exception as e:
        traceback.print_exc()
        raise e
    if isinstance(output, tuple):
        return tuple([to_dlpack(tensor) for tensor in output])
    return to_dlpack(output)
