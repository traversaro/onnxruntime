# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from abc import abstractmethod
from enum import Enum, unique
from typing import Any, Dict, List

import sympy
from onnx import NodeProto, TensorProto

from ._sympy_utils import parse_shape
from ._utils import get_attribute


class CodegenContext:
    def __init__(self, var_map: Dict[str, str]):
        self._var_map: Dict[str, str] = {}
        self._var_map.update(var_map)

    def get_variable_name(self, name: str) -> str:
        return self._var_map[name]

    def get_internal_variable_name(self, name: str) -> str:
        var_name = self._var_map[name]
        return self._var_map[var_name] if var_name in self._var_map else var_name


class NodeVisitor:
    @abstractmethod
    def codegen(self, node: Any, context: CodegenContext, indent: int) -> str:
        pass


@unique
class SpecialVar(str, Enum):
    rbase = "rbase"
    xblock = "XBLOCK"
    rblock = "RBLOCK"


class TensorInfo:
    def __init__(self, dtype, shape: List[Any]):
        self._dtype: TensorProto.DataType = dtype
        self._shape: List = parse_shape(shape)

    @property
    def dtype(self) -> TensorProto.DataType:
        return self._dtype

    @property
    def shape(self) -> List[sympy.Expr]:
        return self._shape


def _infer_elementwise_shape(input_infos: List[TensorInfo]) -> List[sympy.Expr]:
    max_len = max([len(input_info.shape) for input_info in input_infos])
    output_shape: List[sympy.Expr] = [sympy.Integer(1)] * max_len
    for input_info in input_infos:
        offset = max_len - len(input_info.shape)
        for i in range(len(input_info.shape)):
            if not input_info.shape[i].is_number or input_info.shape[i] != 1:
                output_shape[i + offset] = input_info.shape[i]
    return output_shape


def _infer_elementwise(node: NodeProto, input_infos: List[TensorInfo]) -> List[TensorInfo]:
    return [TensorInfo(input_infos[0].dtype, _infer_elementwise_shape(input_infos))]


def _infer_where(node: NodeProto, input_infos: List[TensorInfo]) -> List[TensorInfo]:
    return [TensorInfo(input_infos[1].dtype, _infer_elementwise_shape(input_infos))]


def _infer_reduction(node: NodeProto, input_infos: List[TensorInfo]) -> List[TensorInfo]:
    # Support reduction on the last axis only for now.
    return [TensorInfo(input_infos[0].dtype, input_infos[0].shape[:-1] + [sympy.Integer(1)])]


def _infer_unary(node: NodeProto, input_infos: List[TensorInfo]) -> List[TensorInfo]:
    return [input_infos[0]]


def _infer_cast(node: NodeProto, input_infos: List[TensorInfo]) -> List[TensorInfo]:
    dtype = get_attribute(node, "to", TensorProto.UNDEFINED)
    assert dtype != TensorProto.UNDEFINED
    return [TensorInfo(dtype, input_infos[0].shape)]


def _infer_dropout(node: NodeProto, input_infos: List[TensorInfo]) -> List[TensorInfo]:
    return [input_infos[0], TensorInfo(TensorProto.BOOL, input_infos[0].shape)]


class TypeAndShapeInfer:
    _INFER_FUNC_MAP = {
        "Add": _infer_elementwise,
        "Sub": _infer_elementwise,
        "Mul": _infer_elementwise,
        "Div": _infer_elementwise,
        "Pow": _infer_elementwise,
        "Sqrt": _infer_elementwise,
        "Exp": _infer_elementwise,
        "Where": _infer_where,
        "Rsqrt": _infer_elementwise,
        "Cast": _infer_cast,
        "Dropout": _infer_dropout,
        "DropoutGrad": _infer_unary,
        "Identity": _infer_unary,
        "ReduceSum": _infer_reduction,
        "ReduceMax": _infer_reduction,
        "ReduceMin": _infer_reduction,
    }

    @classmethod
    def infer(cls, node: NodeProto, input_infos: List[TensorInfo]) -> List[TensorInfo]:
        if node.op_type not in cls._INFER_FUNC_MAP:
            raise NotImplementedError(f"Unsupported op type: {node.op_type}")
        return cls._INFER_FUNC_MAP[node.op_type](node, input_infos)
