# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from abc import abstractmethod
from typing import Dict, List, Optional, Set

import numpy as np
import sympy
from onnx import NodeProto

from ._common import CodegenContext, NodeVisitor, TensorInfo
from ._sympy_utils import parse_shape
from ._utils import gen_unique_name, gen_variable_name, to_numpy_type


class TensorArg:
    def __init__(self, name: str, tensor_info: Optional[TensorInfo] = None, data: Optional[np.ndarray] = None):
        self._name: str = name
        self._data: Optional[np.ndarray] = data
        if data is not None:
            self._dtype: np.dtype = data.dtype
            self._shape: List[sympy.Expr] = parse_shape(list(data.shape))
        else:
            assert tensor_info is not None
            self._dtype: np.dtype = to_numpy_type(tensor_info.dtype)
            self._shape: List[sympy.Expr] = tensor_info.shape
        self.cross_kernels: bool = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> List[sympy.Expr]:
        return self._shape

    @property
    def data(self) -> Optional[np.ndarray]:
        return self._data


class OffsetCalculator:
    def __init__(self, target_shape: List[sympy.Expr], reduce_axis: int):
        self.target_shape: List[sympy.Expr] = target_shape
        self.is_reduction: bool = reduce_axis != -1
        self.reduce_axis: int = reduce_axis
        self.x_dims: List[sympy.Expr] = [target_shape[dim] for dim in range(len(target_shape)) if dim != reduce_axis]
        self.x_rank: int = len(self.x_dims)
        self.x_numel: sympy.Expr = sympy.prod(self.x_dims) if self.x_rank > 0 else sympy.Integer(1)
        self.r_numel: sympy.Expr = target_shape[reduce_axis] if self.is_reduction else sympy.Integer(1)
        self.x_strides: List[sympy.Expr] = []
        if self.x_rank > 0:
            self.x_strides.append(sympy.Integer(1))
            for i in range(self.x_rank - 2, -1, -1):
                self.x_strides.insert(0, self.x_strides[0] * self.x_dims[i + 1])
        self.input_strides: Dict[str, List[sympy.Expr]] = dict()
        self.x_compute_dims: Set[int] = set()
        self.recompute: bool = not self.r_numel.is_number or self.r_numel > sympy.Integer(1024)

    def get_input_strides(self, name: str) -> List[sympy.Expr]:
        assert name in self.input_strides
        return self.input_strides[name]

    def get_x_input_strides(self, name: str) -> List[sympy.Expr]:
        return [dim for idx, dim in enumerate(self.get_input_strides(name)) if idx != self.reduce_axis]

    def is_same_x_shape(self, name: str) -> bool:
        if self.is_reduction and self.reduce_axis != 0 and self.reduce_axis != self.x_rank:
            return False
        return all([dim != sympy.Integer(0) for dim in self.get_x_input_strides(name)])

    def register_tensor_arg(self, tensor_arg: TensorArg):
        if tensor_arg.name in self.input_strides:
            return
        strides = []
        input_shape = tensor_arg.shape
        pos1 = len(self.target_shape) - 1
        pos2 = len(input_shape) - 1
        assert pos1 >= pos2
        running_stride = sympy.Integer(1)
        while pos1 >= 0:
            if pos2 >= 0 and self.target_shape[pos1] == input_shape[pos2]:
                strides.insert(0, running_stride)
                running_stride = running_stride * input_shape[pos2]
            else:
                strides.insert(0, sympy.Integer(0))
            pos1 -= 1
            pos2 -= 1
        self.input_strides[tensor_arg.name] = strides
        if not self.is_same_x_shape(tensor_arg.name):
            for idx, dim in enumerate(self.get_x_input_strides(tensor_arg.name)):
                if dim != sympy.Integer(0):
                    self.x_compute_dims.add(idx)

    def is_single_element(self, name: str) -> bool:
        strides = self.get_input_strides(name)
        if self.is_reduction:
            return strides[self.reduce_axis] == sympy.Integer(0)
        return all([dim == sympy.Integer(0) for dim in strides])


class IRNode:
    def __init__(self, inputs: List[TensorArg], outputs: List[TensorArg]):
        self.inputs: List[TensorArg] = inputs
        self.outputs: List[TensorArg] = outputs

    @abstractmethod
    def codegen(self, visitor: NodeVisitor, context: CodegenContext, indent: int = 0) -> str:
        return visitor.codegen(self, context, indent)


class ComputeNode(IRNode):
    def __init__(self, node: NodeProto, inputs: List[TensorArg], outputs: List[TensorArg]):
        super().__init__(inputs, outputs)
        self.node: NodeProto = node
        self.op_type_: str = node.op_type

    @property
    def op_type(self):
        return self.op_type_


class ReduceNode(ComputeNode):
    def __init__(self, node: NodeProto, inputs: List[TensorArg], outputs: List[TensorArg], recompute: bool):
        super().__init__(node, inputs, outputs)
        self.recompute: bool = recompute
        op_type = self.op_type
        assert op_type == "ReduceSum" or op_type == "ReduceMax" or op_type == "ReduceMin"
        self.default_value: str = (
            "0.0" if op_type == "ReduceSum" else ('float("-inf")' if op_type == "ReduceMax" else 'float("inf")')
        )
        self.triton_func: str = (
            "tl.sum" if op_type == "ReduceSum" else ("tl.max" if op_type == "ReduceMax" else "tl.min")
        )


class DropoutNode(ComputeNode):
    def __init__(
        self,
        node: NodeProto,
        inputs: List[TensorArg],
        outputs: List[TensorArg],
        offset_calc: OffsetCalculator,
    ):
        super().__init__(node, inputs, outputs)
        self.offset_calc: OffsetCalculator = offset_calc
        self.offset_calc.register_tensor_arg(inputs[0])
        # It's odd to generate rand for single element
        # assert not self.offset_calc.single_element
        self.global_offset: sympy.Expr = sympy.Integer(0)


class IONode(IRNode):
    def __init__(self, tensor_arg: TensorArg, offset_calc: OffsetCalculator, is_load: bool):
        super().__init__([], [])
        self.tensor_arg: TensorArg = tensor_arg
        self.is_load: bool = is_load
        self.offset_calc: OffsetCalculator = offset_calc
        self.offset_calc.register_tensor_arg(tensor_arg)


class KernelNode(IRNode):
    def __init__(self, inputs: List[TensorArg], outputs: List[TensorArg], target_shape: List, reduce_axis: int):
        super().__init__(inputs, outputs)
        self.name: str = gen_unique_name("kernel")
        self.intermediate_var: Set[str] = set()
        self.constants: Dict[str, TensorArg] = dict()
        self.target_shape: List[sympy.Expr] = target_shape
        self.sub_nodes: List[IRNode] = []
        self.var_map: Dict[str, str] = dict()
        self.symbolic_shape_variables: List[str] = []
        self.dims_offset_compute: Set[int] = set()
        self.has_dropout: bool = False
        self.offset_calc: OffsetCalculator = OffsetCalculator(target_shape, reduce_axis)

    def gen_variable_names(self):
        existing_names = set()
        for input in self.inputs:
            name = gen_variable_name(input.name, "in", existing_names)
            self.var_map[input.name] = name
            self.var_map[name] = "t_" + name
        for output in self.outputs:
            name = gen_variable_name(output.name, "out", existing_names)
            self.var_map[output.name] = name
            self.var_map[name] = "t_" + name
        for name in self.intermediate_var:
            self.var_map[name] = gen_variable_name(name, "t", existing_names)
        for constant_name in self.constants.keys():
            self.var_map[constant_name] = gen_variable_name(constant_name, "c", existing_names)
            if self.constants[constant_name].data is not None:
                value = self.constants[constant_name].data
                if value is not None and value.size == 1:
                    variable_name = self.var_map[constant_name]
                    assert variable_name not in self.var_map
                    self.var_map[variable_name] = str(np.array(value.item(), value.dtype))

        self.symbolic_shape_variables = [str(dim) for dim in self.target_shape if dim.is_symbol]


class ElementwiseKernelNode(KernelNode):
    def __init__(self, inputs: List[TensorArg], outputs: List[TensorArg], target_shape: List[sympy.Expr]):
        super().__init__(inputs, outputs, target_shape, -1)


class ReduceKernelNode(KernelNode):
    def __init__(
        self,
        inputs: List[TensorArg],
        outputs: List[TensorArg],
        target_shape: List[sympy.Expr],
        reduce_axis: int,
    ):
        rank = len(target_shape)
        assert reduce_axis >= -rank and reduce_axis <= rank - 1
        if reduce_axis < 0:
            reduce_axis = rank + reduce_axis
        super().__init__(inputs, outputs, target_shape, reduce_axis)


class ModuleNode(IRNode):
    def __init__(
        self,
        func_name: str,
        inputs: List[TensorArg],
        outputs: List[TensorArg],
        constants: List[TensorArg],
        kernels: List[KernelNode],
    ):
        super().__init__(inputs, outputs)
        self.func_name: str = func_name
        self.constants: List[TensorArg] = constants
        self.kernels: List[KernelNode] = kernels
        self.var_map: Dict[str, str] = dict()
        existing_names = set()
        # Currently need inputs and outputs only. Will need intermediate vars and constants later.
        for input in self.inputs:
            name = gen_variable_name(input.name, "in", existing_names)
            self.var_map[input.name] = name
        for output in self.outputs:
            name = gen_variable_name(output.name, "out", existing_names)
            self.var_map[output.name] = name
        running_offset = sympy.Integer(0)
        self.has_dropout: bool = False
        for kernel in self.kernels:
            for ir_node in kernel.sub_nodes:
                if isinstance(ir_node, DropoutNode):
                    ir_node.global_offset = running_offset
                    running_offset = running_offset + sympy.prod(ir_node.outputs[0].shape)
                    self.has_dropout = True
