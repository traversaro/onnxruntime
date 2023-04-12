# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import re
import uuid
from collections import defaultdict
from typing import Any, List

import numpy as np
from onnx import NodeProto, TensorProto, helper, numpy_helper


def gen_unique_name(prefix: str) -> str:
    return prefix + "_" + uuid.uuid4().hex[:8]


def _topological_soft_internal(node, visited, output_consumers, sorted_nodes):
    visited.add(node.name)
    for next_node in output_consumers[node.name]:
        if next_node.name not in visited:
            _topological_soft_internal(next_node, visited, output_consumers, sorted_nodes)

    sorted_nodes.insert(0, node)


def topological_sort(inputs: List[str], nodes: List[NodeProto]) -> List[NodeProto]:
    # inputs constains the graph inputs and initializers. need to add constant nodes.
    const_nodes = []
    non_const_nodes = []
    for node in nodes:
        if node.name == "":
            node.name = gen_unique_name(node.op_type)
        if node.op_type == "Constant":
            inputs.append(node.output[0])
            const_nodes.append(node)
        else:
            non_const_nodes.append(node)
    graph_input_consumers = defaultdict(list)
    output_consumers = defaultdict(list)
    input_set = set(inputs)
    for node in non_const_nodes:
        for input in node.input:
            if input in input_set:
                graph_input_consumers[input].append(node)
        for output in node.output:
            if not output:
                continue
            for consumer in non_const_nodes:
                if output in consumer.input:
                    output_consumers[node.name].append(consumer)

    visited = set()
    sorted_nodes = []
    for input in inputs:
        for node in graph_input_consumers[input]:
            if node.name not in visited:
                _topological_soft_internal(node, visited, output_consumers, sorted_nodes)

    return const_nodes + sorted_nodes


def get_attribute(node: NodeProto, attr_name: str, default_value: Any = None) -> Any:
    found = [attr for attr in node.attribute if attr.name == attr_name]
    if found:
        return helper.get_attribute_value(found[0])
    return default_value


def to_numpy_array(node: Any) -> np.ndarray:
    tensor = node
    if isinstance(node, NodeProto):
        tensor = get_attribute(node, "value")
    assert isinstance(tensor, TensorProto)
    return numpy_helper.to_array(tensor)


_TENSOR_TYPE_TO_NP_TYPE = {
    int(TensorProto.FLOAT): np.dtype("float32"),
    int(TensorProto.UINT8): np.dtype("uint8"),
    int(TensorProto.INT8): np.dtype("int8"),
    int(TensorProto.UINT16): np.dtype("uint16"),
    int(TensorProto.INT16): np.dtype("int16"),
    int(TensorProto.INT32): np.dtype("int32"),
    int(TensorProto.INT64): np.dtype("int64"),
    int(TensorProto.BOOL): np.dtype("bool"),
    int(TensorProto.FLOAT16): np.dtype("float16"),
    # Native numpy does not support bfloat16 so now use float32 for bf16 values
    int(TensorProto.BFLOAT16): np.dtype("float32"),
    int(TensorProto.DOUBLE): np.dtype("float64"),
    int(TensorProto.COMPLEX64): np.dtype("complex64"),
    int(TensorProto.COMPLEX128): np.dtype("complex128"),
    int(TensorProto.UINT32): np.dtype("uint32"),
    int(TensorProto.UINT64): np.dtype("uint64"),
    int(TensorProto.STRING): np.dtype("object"),
}


def to_numpy_type(tensor_type: TensorProto.DataType) -> np.dtype:
    return _TENSOR_TYPE_TO_NP_TYPE[tensor_type] if not isinstance(tensor_type, np.dtype) else tensor_type


def gen_variable_name(name: str, prefix: str, existing_names: set) -> str:
    pos = name.rfind("/")
    if pos != -1:
        name = name[pos + 1 :]
    pos = name.rfind(".")
    if pos != -1:
        name = name[pos + 1 :]
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    if len(name) > 20:
        name = name[-20:]

    name = f"{prefix}_{name}"
    while name in existing_names:
        name = name + "_1"

    existing_names.add(name)
    return name


def may_add_brackets(name: str) -> str:
    if not re.match("^[A-Za-z0-9_.]*$", name):
        return f"({name})"
    return name
