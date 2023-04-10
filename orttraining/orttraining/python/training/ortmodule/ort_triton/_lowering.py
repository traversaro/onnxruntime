# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import sympy
from onnx import NodeProto

from ._common import TensorInfo
from ._ir import (
    ComputeNode,
    DropoutNode,
    ElementwiseKernelNode,
    IONode,
    KernelNode,
    ModuleNode,
    ReduceKernelNode,
    ReduceNode,
    TensorArg,
)
from ._op_config import is_reduction_node
from ._sorted_graph import SortedGraph
from ._utils import to_numpy_array


def _group_nodes(sorted_graph: SortedGraph) -> Iterable[List[NodeProto]]:
    # TODO: need to graph nodes according to shapes and reduction axes.
    return [sorted_graph.sorted_nodes]


def _extract_io(sorted_graph: SortedGraph) -> Tuple[List[TensorArg], List[TensorArg], List[TensorArg]]:
    input_args = [
        TensorArg(input.name, sorted_graph.node_arg_infos[input.name]) for input in sorted_graph.original_graph.input
    ]
    output_args = [
        TensorArg(output.name, sorted_graph.node_arg_infos[output.name])
        for output in sorted_graph.original_graph.output
    ]
    const_args = []
    for initializer in sorted_graph.original_graph.initializer:
        data = to_numpy_array(initializer)
        const_args.append(TensorArg(initializer.name, data=data))
    for const_node in sorted_graph.const_nodes:
        data = to_numpy_array(const_node)
        const_args.append(TensorArg(const_node.output[0], data=data))
    return input_args, output_args, const_args


def _get_node_io(
    node: NodeProto, arg_cache: Dict[str, TensorArg], node_arg_infos: Dict[str, TensorInfo]
) -> Tuple[List[TensorArg], List[TensorArg]]:
    input_args = []
    for input in node.input:
        if input in arg_cache:
            input_args.append(arg_cache[input])
        else:
            input_args.append(TensorArg(input, node_arg_infos[input]))
            arg_cache[input] = input_args[-1]
    output_args = []
    for output in node.output:
        if output in arg_cache:
            output_args.append(arg_cache[output])
        else:
            output_args.append(TensorArg(output, node_arg_infos[output]))
            arg_cache[output] = output_args[-1]
    return input_args, output_args


def _create_load_or_store(tensor_arg: TensorArg, kernel_node: KernelNode, is_load: bool):
    is_reduction = isinstance(kernel_node, ReduceKernelNode)
    return IONode(tensor_arg, kernel_node.target_tensor_info.shape, is_reduction, is_load)


def _insert_load_and_store(kernel_node: KernelNode):
    input_name_map = [input.name for input in kernel_node.inputs]
    output_name_map = [output.name for output in kernel_node.outputs]
    new_sub_nodes = []
    load_cache = set()
    for node in kernel_node.sub_nodes:
        for input in node.inputs:
            if input.name in kernel_node.constants or input.name in input_name_map:
                if (input.data is not None and input.data.size == 1) or input.name in load_cache:
                    continue
                load_cache.add(input.name)
                new_sub_nodes.append(_create_load_or_store(input, kernel_node, True))
        new_sub_nodes.append(node)
        for output in node.outputs:
            if output.name in output_name_map:
                load_cache.add(output.name)
                new_sub_nodes.append(_create_load_or_store(output, kernel_node, False))
    kernel_node.sub_nodes = new_sub_nodes


def _analyze_io(
    kernel_node: KernelNode,
    graph_inputs: List[TensorArg],
    graph_outputs: List[TensorArg],
    graph_consts: List[TensorArg],
    consumer_counts: Dict[str, int],
    arg_cache: Dict[str, TensorArg],
):
    inputs = defaultdict(lambda: 0)
    outputs = set()
    constants = set()

    graph_input_names = set([arg.name for arg in graph_inputs])
    graph_output_names = set(arg.name for arg in graph_outputs)
    const_names = set([arg.name for arg in graph_consts])

    for node in kernel_node.sub_nodes:
        for arg in node.inputs:
            if arg.name not in const_names:
                inputs[arg.name] += 1
            else:
                constants.add(arg)

    outputs = set()
    for node in kernel_node.sub_nodes:
        for arg in node.outputs:
            name = arg.name
            outputs.add(name)
            if name not in graph_output_names:
                kernel_node.intermediate_var.add(name)
                if consumer_counts[name] > inputs[name]:
                    arg.cross_kernels = True
                    kernel_node.outputs.append(arg)
            else:
                kernel_node.outputs.append(arg)
            if name in inputs:
                inputs[name] = 0

    for input in inputs:
        assert inputs[input] >= 0 and input in arg_cache
        if inputs[input] > 0:
            arg = arg_cache[input]
            if input not in graph_input_names and input not in outputs:
                arg.cross_kernels = True
            kernel_node.inputs.append(arg)

    for arg in constants:
        kernel_node.constants[arg.name] = arg

    _insert_load_and_store(kernel_node)
    for node in kernel_node.sub_nodes:
        if node.has_offset_node:
            assert hasattr(node, "offset_node")
            for idx, dim in enumerate(node.offset_node.strides):
                if dim != sympy.Integer(0):
                    kernel_node.dims_offset_compute.add(idx)
    kernel_node.gen_variable_names()


def _lower(sorted_graph: SortedGraph) -> Tuple[List[TensorArg], List[TensorArg], List[TensorArg], List[KernelNode]]:
    inputs, outputs, constants = _extract_io(sorted_graph)
    grouped_nodes = _group_nodes(sorted_graph)
    arg_cache: Dict[str, TensorArg] = dict()
    for input_arg in inputs:
        arg_cache[input_arg.name] = input_arg
    for output_arg in outputs:
        arg_cache[output_arg.name] = output_arg
    for const_arg in constants:
        arg_cache[const_arg.name] = const_arg
    kernel_nodes = []
    for group in grouped_nodes:
        is_reduction_kernel = any(is_reduction_node(node) for node in group)
        # TODO: it's possible that the last node's first output's shape is not the final output shape.
        target_tensor_info = sorted_graph.node_arg_infos[group[-1].output[0]]
        kernel_nodes.append(
            ReduceKernelNode([], [], target_tensor_info)
            if is_reduction_kernel
            else ElementwiseKernelNode([], [], target_tensor_info)
        )
        sub_nodes = []
        for node in group:
            node_inputs, node_outputs = _get_node_io(node, arg_cache, sorted_graph.node_arg_infos)
            if node.op_type == "Dropout":
                sub_nodes.append(
                    DropoutNode(node, node_inputs, node_outputs, target_tensor_info.shape, is_reduction_kernel)
                )
                kernel_nodes[-1].has_dropout = True
            elif is_reduction_node(node):
                sub_nodes.append(ReduceNode(node, node_inputs, node_outputs))
            else:
                sub_nodes.append(ComputeNode(node, node_inputs, node_outputs))
        kernel_nodes[-1].sub_nodes = sub_nodes

    consumer_counts = {}
    for node in sorted_graph.sorted_nodes:
        for input in node.input:
            if input not in consumer_counts:
                consumer_counts[input] = 0
            consumer_counts[input] += 1
    for kernel_node in kernel_nodes:
        _analyze_io(kernel_node, inputs, outputs, constants, consumer_counts, arg_cache)

    return inputs, outputs, constants, kernel_nodes


def lower(func_name: str, sorted_graph: SortedGraph) -> ModuleNode:
    inputs, outputs, constants, kernels = _lower(sorted_graph)
    # Support single kernel only for a module for now.
    assert len(kernels) == 1
    return ModuleNode(func_name, inputs, outputs, constants, kernels)
