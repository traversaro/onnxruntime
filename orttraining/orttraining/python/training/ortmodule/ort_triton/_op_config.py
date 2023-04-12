# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnx import NodeProto

_ELEMENTWISE_OPS = {
    "Add": {"domain": "", "versions": [7, 13, 14], "is_no_op": False, "conditions": {}},
    "Sub": {"domain": "", "versions": [7, 13, 14], "is_no_op": False, "conditions": {}},
    "Mul": {"domain": "", "versions": [7, 13, 14], "is_no_op": False, "conditions": {}},
    "Div": {"domain": "", "versions": [7, 13, 14], "is_no_op": False, "conditions": {}},
    "Pow": {"domain": "", "versions": [7, 12, 13, 5], "is_no_op": False, "conditions": {}},
    "Sqrt": {"domain": "", "versions": [6, 13], "is_no_op": False, "conditions": {}},
    "Exp": {"domain": "", "versions": [6, 13], "is_no_op": False, "conditions": {}},
    "Where": {"domain": "", "versions": [9, 16], "is_no_op": False, "conditions": {}},
    "Cast": {"domain": "", "versions": [6, 9, 13], "is_no_op": False, "conditions": {}},
    "Dropout": {"domain": "", "versions": [6, 9, 13], "is_no_op": False, "conditions": {}},
    "DropoutGrad": {"domain": "com.microsoft", "versions": [1], "is_no_op": False, "conditions": {}},
    "Identity": {"domain": "", "versions": [6, 9, 13], "is_no_op": True, "conditions": {}},
    # "Tanh", "Erf", "Gelu", "FastGelu", "Relu", "Equal", "Not"
}

_REDUCTION_OPS = {
    "ReduceMean": {"domain": "", "versions": [11, 13], "is_no_op": False, "conditions": {"axes": "[-1]"}},
    "ReduceSum": {"domain": "", "versions": [11, 13], "is_no_op": False, "conditions": {"axes": "[-1]"}},
    "ReduceMax": {"domain": "", "versions": [11, 12, 13], "is_no_op": False, "conditions": {"axes": "[-1]"}},
    "ReduceMin": {"domain": "", "versions": [11, 12, 13], "is_no_op": False, "conditions": {"axes": "[-1]"}},
    "Softmax": {"domain": "", "versions": [11, 13], "is_no_op": False, "conditions": {"axis": "-1"}},
    "SoftmaxGrad_13": {"domain": "com.microsoft", "versions": [1], "is_no_op": False, "conditions": {"axis": "-1"}},
    "LayerNormalization": {"domain": "", "versions": [1], "is_no_op": False, "conditions": {"axis": "-1"}},
    # "LayerNormalizationGrad": {
    #     "domain": "com.microsoft",
    #     "versions": [1],
    #     "is_no_op": False,
    #     "conditions": {"axis": "-1"},
    # },
}


def _contains(node_or_op_type, ops) -> bool:
    op_type = node_or_op_type.op_type if isinstance(node_or_op_type, NodeProto) else node_or_op_type
    return isinstance(op_type, str) and op_type in ops


def is_elementwise_node(node_or_op_type) -> bool:
    return _contains(node_or_op_type, _ELEMENTWISE_OPS)


def is_reduction_node(node_or_op_type) -> bool:
    return _contains(node_or_op_type, _REDUCTION_OPS)


def get_supported_ops():
    return {**_ELEMENTWISE_OPS, **_REDUCTION_OPS}
