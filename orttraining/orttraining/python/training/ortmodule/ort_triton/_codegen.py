# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Tuple

import numpy as np
import sympy
import torch
from sympy.codegen.rewriting import create_expand_pow_optimization

from ._common import CodegenContext, NodeVisitor, SpecialVar
from ._ir import (
    ComputeNode,
    DropoutNode,
    ElementwiseKernelNode,
    IONode,
    IRNode,
    KernelNode,
    ModuleNode,
    OffsetCalculator,
    ReduceKernelNode,
    ReduceNode,
)
from ._lowering import lower
from ._sorted_graph import SortedGraph
from ._sympy_utils import parse_shape, sympy_dot
from ._utils import may_add_brackets


class TritonCodegen(NodeVisitor):
    def __init__(self):
        super().__init__()

    def codegen(self, node: IRNode, context: CodegenContext, indent: int) -> str:
        func = getattr(self, node.__class__.__name__)
        assert func is not None, "unimplemented node: %s" % node.__class__.__name__
        return func(node, context, indent)

    def _get_elementwise_offset_mask(self, offset_calc: OffsetCalculator, arg_name: str) -> Tuple[str, str]:
        if offset_calc.is_single_element(arg_name):
            return "", ""
        if offset_calc.is_same_x_shape(arg_name):
            return "xindex", "xmask"
        strides = offset_calc.get_input_strides(arg_name)
        idx_var = [f"x{idx}" for idx in range(len(strides))]
        expand_opt = create_expand_pow_optimization(6)
        offset_str = str(expand_opt(sympy_dot(parse_shape(idx_var), strides)))
        return offset_str, "xmask"

    def _get_reduce_offset_mask(self, offset_calc: OffsetCalculator, arg_name: str) -> Tuple[str, str]:
        strides = offset_calc.get_input_strides(arg_name)
        if offset_calc.is_same_x_shape(arg_name):
            xoffset_str = (
                "xoffset"
                if offset_calc.reduce_axis == 0 or offset_calc.is_single_element(arg_name)
                else f"xoffset * {offset_calc.r_numel}"
            )
        else:
            x_strides = [strides[dim] for dim in range(len(strides)) if dim != offset_calc.reduce_axis]
            idx_var = [f"x{idx}" for idx in range(len(x_strides))]
            expand_opt = create_expand_pow_optimization(6)
            xoffset_str = str(expand_opt(sympy_dot(parse_shape(idx_var), x_strides)))
            if xoffset_str == "0":
                xoffset_str = ""

        if offset_calc.is_single_element(arg_name):
            return xoffset_str, ""
        reduce_stride = strides[offset_calc.reduce_axis]
        roffset_str = "rindex" if reduce_stride == sympy.Integer(1) else f"rindex * {reduce_stride}"
        offset_str = f"{xoffset_str} + {roffset_str}" if xoffset_str != "" else roffset_str
        return offset_str, "rmask"

    def _get_offset_mask(self, node: OffsetCalculator, arg_name: str) -> Tuple[str, str]:
        return (
            self._get_reduce_offset_mask(node, arg_name)
            if node.is_reduction
            else self._get_elementwise_offset_mask(node, arg_name)
        )

    def IONode(self, node: IONode, context: CodegenContext, indent: int) -> str:
        space_indent = " " * indent
        name = node.tensor_arg.name
        var_name = context.get_variable_name(name)
        internal_var_name = context.get_internal_variable_name(name)
        assert (
            var_name != internal_var_name
        ), f"variable name {var_name} and its internal variable name should not be the same."

        offset_str, mask_str = self._get_offset_mask(node.offset_calc, node.tensor_arg.name)
        if offset_str != "":
            offset_str = f" + {offset_str}"
        if mask_str != "":
            mask_str = f", {mask_str}"
        if node.is_load and mask_str != "":
            mask_str += ", other=0.0"

        if node.is_load:
            return f"{space_indent}{internal_var_name} = tl.load({var_name}{offset_str}{mask_str})\n"
        return f"{space_indent}tl.store({var_name}{offset_str}, {internal_var_name}{mask_str})\n"

    def _gen_kernel_signature(self, node: KernelNode, context: CodegenContext, indent: int) -> str:
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

        blocks_str = f"{SpecialVar.rblock if node.offset_calc.is_reduction else SpecialVar.xblock}: tl.constexpr"

        space_indent = " " * indent
        return (
            f"{space_indent}@triton.jit\n"
            f"{space_indent}def {node.name}({input_args_str}{output_args_str}{other_input_args}{blocks_str}):\n"
        )

    def ElementwiseKernelNode(self, node: ElementwiseKernelNode, context: CodegenContext, indent: int) -> str:
        src_code = self._gen_kernel_signature(node, context, indent)
        indent += 4
        space_indent = " " * indent

        offset_calc = node.offset_calc
        src_code += f"{space_indent}xoffset = tl.program_id(0) * {SpecialVar.xblock}\n"
        src_code += f"{space_indent}xindex = xoffset + tl.arange(0, {SpecialVar.xblock})\n"
        src_code += f"{space_indent}xmask = xindex < {offset_calc.x_numel}\n"
        for idx in range(offset_calc.x_rank):
            if idx in offset_calc.x_compute_dims:
                div_str = (
                    f" // {may_add_brackets(str(offset_calc.x_strides[idx]))}" if idx != offset_calc.x_rank - 1 else ""
                )
                mod_str = f" % {may_add_brackets(str(offset_calc.x_dims[idx]))}" if idx != 0 else ""
                src_code += f"{space_indent}x{idx} = xindex{div_str}{mod_str}\n"
        src_code += "\n"

        if node.has_dropout:
            src_code += f"{space_indent}t_seed_cuda = tl.load(seed_cuda)\n"
            src_code += f"{space_indent}t_seed_cuda = tl.broadcast_to(t_seed_cuda, [{SpecialVar.xblock}])\n"

        for ir_node in node.sub_nodes:
            src_code += ir_node.codegen(self, context, indent)
        return src_code

    def ReduceKernelNode(self, node: ReduceKernelNode, context: CodegenContext, indent: int) -> str:
        src_code = self._gen_kernel_signature(node, context, indent)
        indent += 4
        space_indent = " " * indent

        offset_calc = node.offset_calc
        src_code += f"{space_indent}xoffset = tl.program_id(0)\n"
        src_code += f"{space_indent}{SpecialVar.rbase} = tl.arange(0, {SpecialVar.rblock})\n"
        for idx in range(offset_calc.x_rank):
            if idx in offset_calc.x_compute_dims:
                div_str = (
                    f" // {may_add_brackets(str(offset_calc.x_strides[idx]))}" if idx != offset_calc.x_rank - 1 else ""
                )
                mod_str = f" % {may_add_brackets(str(offset_calc.x_dims[idx]))}" if idx != 0 else ""
                src_code += f"{space_indent}x{idx} = xoffset{div_str}{mod_str}\n"
        src_code += "\n"

        if node.has_dropout:
            src_code += f"{space_indent}t_seed_cuda = tl.load(seed_cuda)\n"
            src_code += f"{space_indent}t_seed_cuda = tl.broadcast_to(t_seed_cuda, [{SpecialVar.rblock}])\n"

        if offset_calc.recompute:
            pos = 0
            nodes_to_skip = set()
            while True:
                while pos < len(node.sub_nodes) and not isinstance(node.sub_nodes[pos], ReduceNode):
                    pos += 1
                if pos < len(node.sub_nodes):
                    reduce_node = node.sub_nodes[pos]
                    assert isinstance(reduce_node, ReduceNode)
                    tmp_var_name = "tmp_" + context.get_internal_variable_name(reduce_node.outputs[0].name)
                    src_code += (
                        f"{space_indent}{tmp_var_name} = "
                        f"tl.zeros([{SpecialVar.rblock}], tl.float32) + {reduce_node.default_value}\n"
                    )
                src_code += f"{space_indent}for roffset in range(0, {offset_calc.r_numel}, {SpecialVar.rblock}):\n"
                src_code += f"{space_indent}    rindex = {SpecialVar.rbase} + roffset\n"
                src_code += f"{space_indent}    rmask = rindex < {offset_calc.r_numel}\n"
                end = pos + 1 if pos < len(node.sub_nodes) else pos
                for i in range(end):
                    if i not in nodes_to_skip:
                        sub_node = node.sub_nodes[i]
                        src_code += sub_node.codegen(self, context, indent + 4)
                        if isinstance(sub_node, IONode) and not sub_node.is_load:
                            nodes_to_skip.add(i)
                if pos < len(node.sub_nodes):
                    nodes_to_skip.add(pos)
                    pos += 1
                    if (
                        pos < len(node.sub_nodes)
                        and isinstance(node.sub_nodes[pos], IONode)
                        and not node.sub_nodes[pos].is_load
                    ):
                        src_code += node.sub_nodes[pos].codegen(self, context, indent)
                        nodes_to_skip.add(pos)
                        pos += 1
                if pos >= len(node.sub_nodes):
                    break
        else:
            src_code += f"{space_indent}rindex = {SpecialVar.rbase}\n"
            src_code += f"{space_indent}rmask = rindex < {offset_calc.r_numel}\n"
            for ir_node in node.sub_nodes:
                src_code += ir_node.codegen(self, context, indent)
        return src_code

    _COMPUTE_CODE_TEMPLATES = {
        "Add": "{indent}{o0} = {i0} + {i1}\n",
        "Sub": "{indent}{o0} = {i0} - {i1}\n",
        "Mul": "{indent}{o0} = {i0} * {i1}\n",
        "Div": "{indent}{o0} = {i0} / {i1}\n",
        "Relu": "{indent}{o0} = tl.maximum({i0}, 0.0)\n",
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

    def ComputeNode(self, node: ComputeNode, context: CodegenContext, indent: int) -> str:
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

    def ReduceNode(self, node: ReduceNode, context: CodegenContext, indent: int) -> str:
        space_indent = " " * indent
        input_var_name = context.get_internal_variable_name(node.inputs[0].name)
        output_var_name = context.get_internal_variable_name(node.outputs[0].name)
        if node.recompute:
            tmp_output_var_name = "tmp_" + output_var_name
            if node.op_type == "ReduceSum":
                code = (
                    f"{space_indent}{tmp_output_var_name} = "
                    f"tl.where(rmask, {tmp_output_var_name} + {input_var_name}, {tmp_output_var_name})\n"
                )
            elif node.op_type == "ReduceMax":
                code = (
                    f"{space_indent}{tmp_output_var_name} = tl.where("
                    f"rmask & ({tmp_output_var_name} < {input_var_name}), {input_var_name}, {tmp_output_var_name})\n"
                )
            else:
                assert node.op_type == "ReduceMin"
                code = (
                    f"{space_indent}{tmp_output_var_name} = tl.where("
                    f"rmask & ({tmp_output_var_name} > {input_var_name}), {input_var_name}, {tmp_output_var_name})\n"
                )
        else:
            code = f"{space_indent}{input_var_name} = tl.where(rmask, {input_var_name}, {node.default_value})\n"
        result_space_indent = " " * (indent - 4) if node.recompute else space_indent
        actual_input_var_name = "tmp_" + output_var_name if node.recompute else input_var_name
        code += f"{result_space_indent}{output_var_name} = {node.triton_func}({actual_input_var_name}, 0)\n"
        return code

    def DropoutNode(self, node: DropoutNode, context: CodegenContext, indent: int) -> str:
        space_indent = " " * indent
        input_var_name = context.get_internal_variable_name(node.inputs[0].name)
        p_var_name = context.get_internal_variable_name(node.inputs[1].name)
        output_var_name = context.get_internal_variable_name(node.outputs[0].name)
        mask_var_name = (
            context.get_internal_variable_name(node.outputs[1].name)
            if len(node.outputs) >= 2
            else "dropout_mask_output"
        )
        offset_str = f"{node.global_offset} + " if node.global_offset != sympy.Integer(0) else ""
        offset_str += self._get_offset_mask(node.offset_calc, node.inputs[0].name)[0]
        return (
            f"{space_indent}p = 1 - {p_var_name}\n"
            f"{space_indent}random = tl.rand(t_seed_cuda, {offset_str})\n"
            f"{space_indent}{mask_var_name} = random < p\n"
            f"{space_indent}{output_var_name} = tl.where({mask_var_name}, {input_var_name} / p, 0.0)\n"
        )

    def ModuleNode(self, node: ModuleNode, context: CodegenContext, indent: int) -> str:
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
            code += (
                f"{space_indent}{context.get_variable_name(output.name)} = "
                f'torch.empty({tuple(output.shape)}, dtype={torch_dtype}, device="cuda")\n'
            )

        if node.has_dropout:
            code += f'\n{space_indent}seed_cuda = torch.randint(2**31, size=(), dtype=torch.int64, device="cuda")\n'

        # TODO: support multiple blocks.
        assert len(node.kernels) == 1
        kernel_node = node.kernels[0]
        kernel_args_str = ", ".join([context.get_variable_name(input.name) for input in kernel_node.inputs])
        if kernel_args_str != "":
            kernel_args_str += ", "
        kernel_args_str += ", ".join([context.get_variable_name(output.name) for output in kernel_node.outputs])
        # TODO: support other kinds of variable args, such as symbolic shape variable.
        if kernel_node.has_dropout:
            kernel_args_str += ", seed_cuda"

        if isinstance(kernel_node, ReduceKernelNode):
            rblock_str = "1024" if kernel_node.offset_calc.recompute else "triton.next_power_of_2(n_reduce_dim)"
            code += f"""
{space_indent}n_reduce_dim = {kernel_node.offset_calc.r_numel}
{space_indent}{kernel_node.name}[({kernel_node.offset_calc.x_numel},)]({kernel_args_str}, RBLOCK={rblock_str})
"""
        else:
            x_numel = kernel_node.offset_calc.x_numel
            small_size = x_numel.is_number and x_numel < 1024
            xblock_str = "triton.next_power_of_2(n_elements)" if small_size else "1024"
            code += f"""
{space_indent}n_elements = {x_numel}
{space_indent}grid = lambda meta: (triton.cdiv(n_elements, meta[\"XBLOCK\"]),)
{space_indent}{kernel_node.name}[grid]({kernel_args_str}, XBLOCK={xblock_str})
"""

        return_output_str = ", ".join([context.get_variable_name(output.name) for output in node.outputs])
        code += f"{space_indent}return {return_output_str}\n"
        return code


def codegen(func_name: str, sorted_graph: SortedGraph) -> str:
    module_node = lower(func_name, sorted_graph)
    return module_node.codegen(TritonCodegen(), CodegenContext(module_node.var_map))
