# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Dict, List

import numpy as np
import sympy
import torch
from sympy.codegen.rewriting import create_expand_pow_optimization

from ._common import CodegenContext, NodeVisitor, SpecialVar
from ._ir import (
    ComputeNode,
    DropoutNode,
    ElementwiseKernelNode,
    ElementwiseOffsetNode,
    IONode,
    IRNode,
    KernelNode,
    ModuleNode,
    ReduceKernelNode,
    ReduceNode,
    ReduceOffsetNode,
)
from ._lowering import lower
from ._sorted_graph import SortedGraph
from ._sympy_utils import sympy_dot, sympy_symbol
from ._utils import may_add_brackets


class TritonCodegen(NodeVisitor):
    def __init__(self):
        super().__init__()

    def codegen(self, node: IRNode, context: CodegenContext, indent: int):
        func = getattr(self, node.__class__.__name__)
        assert func is not None, "unimplemented node: %s" % node.__class__.__name__
        return func(node, context, indent)

    def ElementwiseOffsetNode(self, node: ElementwiseOffsetNode, context: CodegenContext, indent: int):
        offset_str = "xindex"
        mask_str = "xmask"
        if not node.same_shape:
            idx_var = [f"x{idx}" for idx in range(len(node.strides))]
            expand_opt = create_expand_pow_optimization(6)
            offset_str = str(expand_opt(sympy_dot(sympy_symbol(idx_var), node.strides)))
            if offset_str == "0":
                offset_str = ""
                mask_str = ""
        # return offset and mask separated by "|".
        return f"{offset_str}|{mask_str}"

    def ReduceOffsetNode(self, node: ReduceOffsetNode, context: CodegenContext, indent: int):
        xoffset_str = "xoffset"
        if not node.same_x_shape:
            idx_var = [f"x{idx}" for idx in range(len(node.strides))]
            expand_opt = create_expand_pow_optimization(6)
            xoffset_str = str(expand_opt(sympy_dot(sympy_symbol(idx_var), node.strides)))
            if xoffset_str == "0":
                xoffset_str = ""
        if xoffset_str != "" and not node.single_element:
            if xoffset_str == "1":
                xoffset_str = f"{node.last_dim}"
            else:
                xoffset_str = f"{may_add_brackets(xoffset_str)} * {node.last_dim}"

        offset_str = xoffset_str
        mask_str = ""
        if not node.single_element:
            offset_str = f"{xoffset_str} + rindex" if xoffset_str != "" else "rindex"
            mask_str = "rmask"
        # return offset and mask separated by "|".
        return f"{offset_str}|{mask_str}"

    def IONode(self, node: IONode, context: CodegenContext, indent: int):
        space_indent = " " * indent
        name = node.tensor_info.name
        var_name = context.get_variable_name(name)
        internal_var_name = context.get_internal_variable_name(name)
        assert var_name != internal_var_name, f"variable name {var_name} and its internal variable name should not be the same."

        offset_str, mask_str = tuple(node.offset_node.codegen(self, context, indent).split("|"))
        if offset_str != "":
            offset_str = f" + {offset_str}"
        if mask_str != "":
            mask_str = f", {mask_str}"
        if node.is_load and mask_str != "":
            mask_str += ", other=0.0"

        if node.is_load:
            code = f"{space_indent}{internal_var_name} = tl.load({var_name}{offset_str}{mask_str})\n"
            # if node.offset_node.single_element:
            #     code += f"{space_indent}{internal_var_name} = tl.broadcast_to({internal_var_name}, [{SpecialVar.rblock}])\n"
            return code
        return f"{space_indent}tl.store({var_name}{offset_str}, {internal_var_name}{mask_str})\n"

    def _gen_kernel_signature(self, node: KernelNode, context: CodegenContext, block_var: str, indent: int) -> str:
        input_args = [context.get_variable_name(input.name) for input in node.inputs]
        input_args_str = ", ".join(input_args)
        if input_args_str != "":
            input_args_str += ", "

        output_args = [context.get_variable_name(output.name) for output in node.outputs]
        output_args_str = ", ".join(output_args) + ", "

        other_input_args = "seed_cuda, " if node.has_dropout else ""
        # Support symbolic shape if any.
        symbolic_shape_args_str = ", ".join(node.symbolic_shape_variables)
        if symbolic_shape_args_str != "":
            other_input_args += f"{symbolic_shape_args_str}, "

        space_indent = " " * indent
        return (
            f"{space_indent}@triton.jit\n"
            f"{space_indent}def {node.name}({input_args_str}{output_args_str}"
            f"{other_input_args}{block_var}: tl.constexpr):\n"
        )

    def _gen_strides(self, shape: List) -> List:
        rank = len(shape)
        if rank == 0:
            return []
        strides = [sympy.Integer(1)]
        if rank > 1:
            for i in range(len(shape) - 2, -1, -1):
                strides.insert(0, strides[0] * shape[i + 1])
        return strides

    def ElementwiseKernelNode(self, node: ElementwiseKernelNode, context: CodegenContext, indent: int) -> str:
        src_code = self._gen_kernel_signature(node, context, SpecialVar.xblock, indent)
        indent += 4
        space_indent = " " * indent

        target_shape = node.target_tensor_info.shape
        strides = self._gen_strides(target_shape)
        rank = len(strides)
        xnumel = strides[0] * target_shape[0] if rank > 0 else sympy.Integer(1)
        src_code += f"{space_indent}xoffset = tl.program_id(0) * {SpecialVar.xblock}\n"
        src_code += f"{space_indent}xindex = xoffset + tl.arange(0, {SpecialVar.xblock})\n"
        src_code += f"{space_indent}xmask = xindex < {xnumel}\n"
        for idx in range(len(target_shape)):
            if idx in node.dims_offset_compute:
                div_str = f" // {may_add_brackets(str(strides[idx]))}" if idx != rank - 1 else ""
                mod_str = f" % {may_add_brackets(str(target_shape[idx]))}" if idx != 0 else ""
                src_code += f"{space_indent}x{idx} = xindex{div_str}{mod_str}\n"
        src_code += "\n"

        if node.has_dropout:
            src_code += f"{space_indent}t_seed_cuda = tl.load(seed_cuda)\n"
            src_code += f"{space_indent}t_seed_cuda = tl.broadcast_to(t_seed_cuda, [{SpecialVar.xblock}])\n"

        for ir_node in node.sub_nodes:
            src_code += ir_node.codegen(self, context, indent)
        return src_code

    def ReduceKernelNode(self, node: ReduceKernelNode, context: CodegenContext, indent: int):
        src_code = self._gen_kernel_signature(node, context, SpecialVar.rblock, indent)
        indent += 4
        space_indent = " " * indent

        # Support reduce on last axis only for now.
        target_shape = node.target_tensor_info.shape
        x_shape = target_shape[:-1]
        strides = self._gen_strides(x_shape)
        x_rank = len(strides)
        rnumel = target_shape[-1]
        src_code += f"{space_indent}xoffset = tl.program_id(0)\n"
        src_code += f"{space_indent}{SpecialVar.rbase} = tl.arange(0, {SpecialVar.rblock})\n"
        src_code += f"{space_indent}rindex = {SpecialVar.rbase}\n"
        src_code += f"{space_indent}rmask = rindex < {rnumel}\n"
        for idx in range(len(x_shape)):
            if idx in node.dims_offset_compute:
                div_str = f" // {may_add_brackets(str(strides[idx]))}" if idx != x_rank - 1 else ""
                mod_str = f" % {may_add_brackets(str(x_shape[idx]))}" if idx != 0 else ""
                src_code += f"{space_indent}x{idx} = xoffset{div_str}{mod_str}\n"
        src_code += "\n"

        if node.has_dropout:
            src_code += f"{space_indent}t_seed_cuda = tl.load(seed_cuda)\n"
            src_code += f"{space_indent}t_seed_cuda = tl.broadcast_to(t_seed_cuda, [{SpecialVar.rblock}])\n"

        for ir_node in node.sub_nodes:
            src_code += ir_node.codegen(self, context, indent)
        return src_code

    _COMPUTE_CODE_TEMPLATES = {
        "Add": "{indent}{o0} = {i0} + {i1}\n",
        "Sub": "{indent}{o0} = {i0} - {i1}\n",
        "Mul": "{indent}{o0} = {i0} * {i1}\n",
        "Div": "{indent}{o0} = {i0} / {i1}\n",
        "Relu": "{indent}{o0} = tl.maximum({i0}, '0.f')\n",
        "Pow": "{indent}{o0} = tl.libdevice.pow({i0}, {i1})\n",
        "Pow2": "{indent}{o0} = {i0} * {i0}\n",
        "Pow3": "{indent}{o0} = {i0} * {i0} * {i0}\n",
        "Sqrt": "{indent}{o0} = tl.sqrt({i0})\n",
        "Rsqrt": "{indent}{o0} = 1.0 / tl.sqrt({i0})\n",
        "Cast": "{indent}{o0} = {i0}.to(tl.{dtype})\n",
        "CastBool": "{indent}{o0} = {i0} != 0\n",
        "Erf": "{indent}{o0} = tl.libdevice.erf({i0})\n",
        "Gelu": "{indent}{o0} = (tl.libdevice.erf({i0} / 1.41421356237) + 1.0) * 0.5\n",
        "Exp": "{indent}{o0} = tl.exp({i0})\n",
        "Tanh": "{indent}{o0} = tl.libdevice.tanh({i0})\n",
        "Where": "{indent}{o0} = tl.where({i0}, {i1}, {i2})\n",
        "Sigmoid": "{indent}{o0} = tl.sigmoid({i0})\n",
        "Log": "{indent}{o0} = tl.log({i0})\n",
        "DropoutGrad": "{indent}p = 1 - {i2}\n{indent}{o0} = tl.where({i1}, {i0} / p, 0.0)\n",
        "Identity": "{indent}{o0} = {i0}\n",
    }

    def ComputeNode(self, node: ComputeNode, context: CodegenContext, indent: int):
        space_indent = " " * indent
        kwargs = {}
        for idx, input in enumerate(node.inputs):
            kwargs[f"i{idx}"] = context.get_internal_variable_name(input.name)
        for idx, output in enumerate(node.outputs):
            kwargs[f"o{idx}"] = context.get_internal_variable_name(output.name)

        op_type = node.op_type
        if op_type == "Pow":
            if kwargs["i1"] == 2:
                op_type = "Pow2"
            elif kwargs["i1"] == 3:
                op_type = "Pow3"
            elif kwargs["i1"] == 0.5:
                op_type = "Sqrt"

        if op_type == "Cast":
            from_dtype = node.inputs[0].dtype.type
            to_dtype = node.outputs[0].dtype.type
            if from_dtype == to_dtype:
                op_type = "Identity"
            elif to_dtype == np.bool_:
                op_type = "CastBool"
            else:
                kwargs["dtype"] = to_dtype.__name__

        return TritonCodegen._COMPUTE_CODE_TEMPLATES[op_type].format(indent=space_indent, **kwargs)

    def ReduceNode(self, node: ReduceNode, context: CodegenContext, indent: int):
        space_indent = " " * indent
        op_type = node.op_type
        assert op_type == "ReduceSum" or op_type == "ReduceMax" or op_type == "ReduceMin"
        # Support reduce on last axis only for now.
        input_var_name = context.get_internal_variable_name(node.inputs[0].name)
        output_var_name = context.get_internal_variable_name(node.outputs[0].name)
        default_value = "0.0" if op_type == "ReduceSum" else ('float("-inf")' if op_type == "ReduceMax" else 'float("inf")')
        code = f"{space_indent}{input_var_name} = tl.where(rmask, {input_var_name}, {default_value})\n"
        triton_func = "sum" if op_type == "ReduceSum" else ("max" if op_type == "ReduceMax" else "min")
        code += f"{space_indent}{output_var_name} = tl.{triton_func}({input_var_name}, 0)\n"
        return code

    def DropoutNode(self, node: DropoutNode, context: CodegenContext, indent: int):
        space_indent = " " * indent
        input_var_name = context.get_internal_variable_name(node.inputs[0].name)
        p_var_name = context.get_internal_variable_name(node.inputs[1].name)
        output_var_name = context.get_internal_variable_name(node.outputs[0].name)
        mask_var_name = context.get_internal_variable_name(node.outputs[1].name) if len(node.outputs) >= 2 else "dropout_mask_output"
        offset_str = f"{node.global_offset} + " if node.global_offset != sympy.Integer(0) else ""
        offset_str += node.offset_node.codegen(self, context, indent).split("|")[0]
        return (
            f"{space_indent}p = 1 - {p_var_name}\n"
            f"{space_indent}random = tl.rand(t_seed_cuda, {offset_str})\n"
            f"{space_indent}{mask_var_name} = random < p\n"
            f"{space_indent}{output_var_name} = tl.where({mask_var_name}, {input_var_name} / p, 0.0)\n"
        )

    def ModuleNode(self, node: ModuleNode, context: CodegenContext, indent: int):
        code = """
import triton
import triton.language as tl
import torch

"""

        for kernel_node in node.kernels:
            code += kernel_node.codegen(self, CodegenContext(kernel_node.var_map), indent)

        input_args = ", ".join([context.get_variable_name(input.name) for input in node.inputs])

        space_indent = " " * indent
        code += f"\n\n{space_indent}def {node.func_name}({input_args}):\n"

        indent += 4
        space_indent = " " * indent

        # Allocate output tensor.
        for output in node.outputs:
            torch_dtype = torch.from_numpy(np.zeros(1, dtype=output.dtype)).dtype
            # Workaround for DLPack which doesn't support bool.
            if torch_dtype == torch.bool:
                torch_dtype = torch.uint8
            code += f"{space_indent}{context.get_variable_name(output.name)} = torch.empty({tuple(output.shape)}, dtype={torch_dtype}, device='cuda')\n"

        if node.has_dropout:
            code += f"\n{space_indent}seed_cuda = torch.randint(2**31, size=(), dtype=torch.int64, device='cuda')\n"

        # TODO: support multiple blocks.
        assert len(node.kernels) == 1
        kernel_args_str = ", ".join([context.get_variable_name(input.name) for input in node.kernels[0].inputs])
        if kernel_args_str != "":
            kernel_args_str += ", "
        kernel_args_str += ", ".join([context.get_variable_name(output.name) for output in node.kernels[0].outputs])
        # TODO: support other kinds of variable args, such as symbolic shape variable.
        if node.kernels[0].has_dropout:
            kernel_args_str += ", seed_cuda"

        if isinstance(node.kernels[0], ReduceKernelNode):
            target_shape = node.kernels[0].target_tensor_info.shape
            last_dim = target_shape[-1]
            code += f"""
{space_indent}n_last_dim = {last_dim}
{space_indent}paralleled_blocks = {np.prod(target_shape) // last_dim}
{space_indent}{node.kernels[0].name}[(paralleled_blocks,)]({kernel_args_str}, RBLOCK={("1024" if node.kernels[0].recompute else "triton.next_power_of_2(n_last_dim)")})
"""
        else:
            code += f"""
{space_indent}n_elements = {np.prod(node.kernels[0].target_tensor_info.shape)}
{space_indent}grid = lambda meta: (triton.cdiv(n_elements, meta['XBLOCK']),)
{space_indent}{node.kernels[0].name}[grid]({kernel_args_str}, XBLOCK=1024)
"""

        return_output_str = ", ".join([context.get_variable_name(output.name) for output in node.outputs])
        code += f"{space_indent}return {return_output_str}\n"
        return code


def codegen(func_name: str, sorted_graph: SortedGraph) -> str:
    module_node = lower(func_name, sorted_graph)
    return module_node.codegen(TritonCodegen(), CodegenContext(module_node.var_map))
