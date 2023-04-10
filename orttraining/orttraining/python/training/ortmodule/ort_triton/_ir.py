# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import secrets
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
        self._name = name
        self._data = data
        if data is not None:
            self._dtype = data.dtype
            self._shape = parse_shape(list(data.shape))
        else:
            assert tensor_info is not None
            self._dtype = to_numpy_type(tensor_info.dtype)
            self._shape = tensor_info.shape
        self.cross_kernels = False

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def data(self):
        return self._data


class IRNode:
    def __init__(self, inputs: List[TensorArg], outputs: List[TensorArg]):
        self.inputs: List[TensorArg] = inputs
        self.outputs: List[TensorArg] = outputs
        self.has_offset_node: bool = False

    @abstractmethod
    def codegen(self, visitor: NodeVisitor, context: CodegenContext, indent: int = 0) -> str:
        return visitor.codegen(self, context, indent)


class ComputeNode(IRNode):
    def __init__(self, node: NodeProto, inputs: List[TensorArg], outputs: List[TensorArg]):
        super().__init__(inputs, outputs)
        self.node = node
        self.op_type_ = node.op_type
        self.op_name = node.name

    @property
    def op_type(self):
        return self.op_type_


class ReduceNode(ComputeNode):
    def __init__(self, node: NodeProto, inputs: List[TensorArg], outputs: List[TensorArg]):
        super().__init__(node, inputs, outputs)


class DropoutNode(ComputeNode):
    def __init__(
        self,
        node: NodeProto,
        inputs: List[TensorArg],
        outputs: List[TensorArg],
        kernel_target_shape: List,
        is_reduction: bool,
    ):
        super().__init__(node, inputs, outputs)
        self.offset_node = (
            ReduceOffsetNode(outputs[0].shape, kernel_target_shape)
            if is_reduction
            else ElementwiseOffsetNode(outputs[0].shape, kernel_target_shape)
        )
        self.has_offset_node = True
        # It's odd to generate rand for single element
        assert not self.offset_node.single_element
        self.global_offset = sympy.Integer(0)


class OffsetNode(IRNode):
    def __init__(self):
        super().__init__([], [])

    def _gen_strides(self, input_shape: List, target_shape: List) -> List:
        strides = []
        if input_shape != target_shape:
            pos1 = len(target_shape) - 1
            pos2 = len(input_shape) - 1
            running_stride = sympy.Integer(1)
            while pos1 >= 0:
                if pos2 >= 0 and target_shape[pos1] == input_shape[pos2]:
                    strides.insert(0, running_stride)
                    running_stride = running_stride * input_shape[pos2]
                else:
                    strides.insert(0, sympy.Integer(0))
                pos1 -= 1
                pos2 -= 1
        return strides


class ElementwiseOffsetNode(OffsetNode):
    def __init__(self, input_shape: List, target_shape: List):
        super().__init__()
        self.strides = self._gen_strides(input_shape, target_shape)
        self.same_shape = len(self.strides) == 0
        self.single_element = False


class ReduceOffsetNode(OffsetNode):
    def __init__(self, input_shape: List, target_shape: List):
        super().__init__()
        self.strides = self._gen_strides(input_shape[:-1], target_shape[:-1])
        self.last_dim = target_shape[-1]
        self.same_x_shape = len(self.strides) == 0
        self.single_element = len(input_shape) == 0 or input_shape[-1] == sympy.Integer(1)


class IONode(IRNode):
    def __init__(self, tensor_arg: TensorArg, kernel_target_shape: List, is_reduction: bool, is_load: bool):
        super().__init__([], [])
        self.tensor_info = tensor_arg
        self.is_load = is_load
        self.offset_node = (
            ReduceOffsetNode(tensor_arg.shape, kernel_target_shape)
            if is_reduction
            else ElementwiseOffsetNode(tensor_arg.shape, kernel_target_shape)
        )
        self.has_offset_node = True


class KernelNode(IRNode):
    def __init__(self, inputs: List[TensorArg], outputs: List[TensorArg], target_tensor_info: TensorInfo):
        super().__init__(inputs, outputs)
        self.name: str = gen_unique_name("kernel")
        self.intermediate_var: Set[str] = set()
        self.constants: Dict[str, TensorArg] = dict()
        self.target_tensor_info: TensorInfo = target_tensor_info
        self.sub_nodes: List[IRNode] = []
        self.var_map: Dict[str, str] = dict()
        self.symbolic_shape_variables: List[str] = []
        self.dims_offset_compute: Set[int] = set()
        self.has_dropout = False

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

        self.symbolic_shape_variables = [dim for dim in self.target_tensor_info.shape if dim.is_symbol]


class ElementwiseKernelNode(KernelNode):
    def __init__(self, inputs: List[TensorArg], outputs: List[TensorArg], target_tensor_info: TensorInfo):
        super().__init__(inputs, outputs, target_tensor_info)


class ReduceKernelNode(KernelNode):
    def __init__(self, inputs: List[TensorArg], outputs: List[TensorArg], target_tensor_info: TensorInfo):
        super().__init__(inputs, outputs, target_tensor_info)
        last_dim = target_tensor_info.shape[-1] if target_tensor_info.shape is not None else None
        self.recompute: bool = last_dim is not None and last_dim.is_number and last_dim > 1024


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
        self.func_name = func_name
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
        self.has_dropout = False
        for kernel in self.kernels:
            for ir_node in kernel.sub_nodes:
                if isinstance(ir_node, DropoutNode):
                    ir_node.global_offset = running_offset
                    running_offset = running_offset + np.prod(ir_node.outputs[0].shape)
                    self.has_dropout = True
