# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import copy
from typing import List

import numpy as np
import onnx
from onnx import ModelProto

from ._common import TensorInfo, TypeAndShapeInfer
from ._de_compose import DecomposeDispatch
from ._utils import get_attribute, to_numpy_array, topological_sort


class SortedGraph(object):
    def __init__(self, model: ModelProto, input_shapes: List[List]):
        self._model = model
        self._graph = model.graph
        self._input_shapes = input_shapes
        self._sorted_nodes = topological_sort(
            [input.name for input in self._graph.input] + [initializer.name for initializer in self._graph.initializer],
            self._graph.node,
        )
        self._node_arg_infos = {}

        for idx, input in enumerate(self._graph.input):
            self._node_arg_infos[input.name] = TensorInfo(input.type.tensor_type.elem_type, self._input_shapes[idx])
        for initializer in self._graph.initializer:
            self._node_arg_infos[initializer.name] = TensorInfo(
                initializer.data_type,
                list(to_numpy_array(initializer).shape),
            )

        self._decompose()

        initializers = {}
        for initializer in self._graph.initializer:
            initializers[initializer.name] = initializer
        self._sorted_initializers = []
        for node in self._sorted_nodes:
            for input in node.input:
                if input in initializers:
                    self._sorted_initializers.append(initializers[input])
                    initializers.pop(input)

        self._const_nodes = [node for node in self._sorted_nodes if node.op_type == "Constant"]
        self._sorted_nodes = [node for node in self._sorted_nodes if node.op_type != "Constant"]

    def __str__(self):
        graph_inputs = []
        name_map = {}
        for idx, input in enumerate(self._graph.input):
            shape_str = str(self._input_shapes[idx]).replace(" ", "")
            graph_inputs.append(f"({str(input.type.tensor_type.elem_type)},{shape_str})")
            name_map[input.name] = f"i{idx}"
        graph_inputs_str = ",".join(graph_inputs)
        constants = []
        for idx, initializer in enumerate(self._sorted_initializers):
            data_str = (
                np.array2string(to_numpy_array(initializer), separator=",").replace("\n", "").replace(" ", "")
            )
            constants.append(f"({initializer.data_type},{data_str})")
            name_map[initializer.name] = f"c{idx}"
        for idx, node in enumerate(self._const_nodes):
            value_attr = get_attribute(node, "value")
            data_str = (
                np.array2string(to_numpy_array(value_attr), separator=",")
                .replace("\n", "")
                .replace(" ", "")
            )
            constants.append(f"({value_attr.data_type},{data_str})")
            name_map[node.output[0]] = f"c{idx + len(self._sorted_initializers)}"
        constants_str = ",".join(constants)
        for idx, output in enumerate(self._graph.output):
            name_map[output.name] = f"o{idx}"
        nodes = []
        for node_idx, node in enumerate(self._sorted_nodes):
            inputs = []
            for input in node.input:
                inputs.append(name_map[input] if input in name_map else input)
            inputs_str = ",".join(inputs)
            outputs = []
            for idx, output in enumerate(node.output):
                if output in name_map:
                    outputs.append(name_map[output])
                else:
                    name_map[output] = f"t{node_idx}_{idx}"
                    outputs.append(name_map[output])
            outputs_str = ",".join(outputs)
            attributes = []
            for attr in node.attribute:
                fields = [str(f[1]) for f in attr.ListFields()]
                attributes.append(f"{fields[0]}:{fields[2]}={fields[1]}")
            attributes_str = ",".join(attributes)
            nodes.append(f"{node.op_type}[{attributes_str}]({inputs_str})->({outputs_str})")
        nodes_str = ",".join(nodes)
        return f"{graph_inputs_str}|{str(len(self._graph.output))}|{constants_str}|{nodes_str}"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    @property
    def const_nodes(self):
        return self._const_nodes

    @property
    def sorted_nodes(self):
        return self._sorted_nodes

    @property
    def original_graph(self):
        return self._graph

    @property
    def node_arg_infos(self):
        return self._node_arg_infos

    def _decompose(self):
        dispatch = DecomposeDispatch()
        pos = 0
        while pos < len(self._sorted_nodes):
            node = self._sorted_nodes[pos]
            if node.op_type in dispatch:
                new_nodes = dispatch(node, node_arg_infos=self._node_arg_infos)
                new_nodes = topological_sort(node.input, new_nodes)
                self._sorted_nodes[pos : pos + 1] = new_nodes
                continue
            if node.op_type == "Constant":
                value_attr = get_attribute(node, "value")
                self._node_arg_infos[node.output[0]] = TensorInfo(
                    value_attr.data_type,
                    list(to_numpy_array(value_attr).shape),
                )
            else:
                input_infos = []
                for input in node.input:
                    input_infos.append(self._node_arg_infos[input])
                output_infos = TypeAndShapeInfer.infer(node, input_infos)
                if len(node.output) == 1:
                    self._node_arg_infos[node.output[0]] = output_infos
                else:
                    for idx, output in enumerate(node.output):
                        self._node_arg_infos[output] = output_infos[idx]
            pos += 1

    def save_onnx(self, file_path_prefix):
        # TODO: put shapes to the graphs.
        onnx.save(self._model, file_path_prefix + "_original.onnx")
        processed_model = copy.deepcopy(self._model)
        processed_model.graph.ClearField("node")
        processed_model.graph.node.extend(self._const_nodes)
        processed_model.graph.node.extend(self._sorted_nodes)
        onnx.save(processed_model, file_path_prefix + "_processed.onnx")
