// port-lint: source cpu_code_gen.rs
package io.github.kotlinmania.ug.cpu_code_gen

import io.github.kotlinmania.ug.Const
import io.github.kotlinmania.ug.DType
import io.github.kotlinmania.ug.UgError
import io.github.kotlinmania.ug.lang.BinaryOp
import io.github.kotlinmania.ug.lang.ReduceOp
import io.github.kotlinmania.ug.lang.Type
import io.github.kotlinmania.ug.lang.UnaryOp
import io.github.kotlinmania.ug.lang.ssa.A
import io.github.kotlinmania.ug.lang.ssa.Instr
import io.github.kotlinmania.ug.lang.ssa.Kernel
import io.github.kotlinmania.ug.lang.ssa.Special
import io.github.kotlinmania.ug.lang.ssa.VarId

private fun formatV(v: VarId): String = "__var${v.asUsize()}"

private fun formatD(dtype: DType): String =
    when (dtype) {
        DType.BF16 -> "__bf16"
        DType.F16 -> "_Float16"
        DType.F32 -> "float"
        DType.I32 -> "int"
        DType.I64 -> "long long"
    }

private fun formatF32(v: Float): String =
    when {
        v.isNaN() -> "0. / 0."
        v == Float.POSITIVE_INFINITY -> "1. / 0."
        v == Float.NEGATIVE_INFINITY -> "-1. / 0."
        else -> v.toString()
    }

private fun formatC(c: Const): String =
    when (c) {
        is Const.BF16 -> formatF32(c.toHalfBF16().toFloat())
        is Const.F16 -> formatF32(c.toHalfF16().toFloat())
        is Const.F32 -> formatF32(c.value)
        is Const.I32 -> c.value.toString()
        is Const.I64 -> c.value.toString()
    }

private fun formatA(a: A): String =
    when (a) {
        is A.Var -> formatV(a.id)
        is A.Constant -> formatC(a.value)
    }

public fun gen(w: Appendable, funcName: String, kernel: Kernel) {
    val instrs = kernel.instrs()
    val containsReduceLocal = instrs.any { it is Instr.ReduceLocal }
    if (containsReduceLocal) {
        throw UgError.Msg("TODO: add the reduce code if needed")
    }

    w.append("#include<math.h>\n")
    w.append("void ").append(funcName).append("(\n")
    val args = kernel.args()
    for ((argIdx, kernelArg) in args.withIndex()) {
        val isLast = argIdx == args.size - 1
        val delim = if (isLast) "" else ","
        val tyStr =
            when (val t = kernelArg.arg.type) {
                is Type.Value -> formatD(t.dtype)
                is Type.Ptr -> "${formatD(t.dtype)}*"
            }
        w.append("  ").append(tyStr).append(' ').append(formatV(VarId.new(kernelArg.index))).append(delim).append('\n')
    }
    w.append(") {\n")

    var depth = 0
    for ((varIdIdx, instr) in instrs.withIndex()) {
        val varId = formatV(VarId.new(varIdIdx))
        val indent = " ".repeat(2 * depth + 2)
        when (instr) {
            is Instr.DefineGlobal -> {}
            is Instr.DefineLocal -> {
                w.append(indent).append("__shared__ ").append(formatD(instr.dtype)).append(' ').append(varId)
                    .append('[').append(instr.size.toString()).append("];\n")
            }

            is Instr.DefineAcc -> {
                w.append(indent).append(formatD(instr.value.dtype)).append(' ').append(varId)
                    .append(" = ").append(formatC(instr.value)).append(";\n")
            }

            is Instr.Constant -> {
                w.append(indent).append(formatD(instr.value.dtype)).append(' ').append(varId)
                    .append(" = ").append(formatC(instr.value)).append(";\n")
            }

            is Instr.If -> {
                w.append(indent).append("if (").append(formatA(instr.cond)).append(") {\n")
                depth += 1
            }

            is Instr.Range -> {
                w.append(indent).append("for (int ").append(varId).append(" = ")
                    .append(formatA(instr.lo)).append("; ").append(varId).append(" < ")
                    .append(formatA(instr.up)).append("; ").append(varId).append("+=")
                    .append(instr.step.toString()).append(") {\n")
                depth += 1
            }

            is Instr.EndIf, is Instr.EndRange -> {
                if (depth == 0) throw UgError.Msg("unmatched EndRange")
                depth -= 1
                val curIndent = " ".repeat(2 * depth + 2)
                w.append(curIndent).append("}\n")
            }

            is Instr.Load -> {
                w.append(indent).append(formatD(instr.dtype)).append(' ').append(varId)
                    .append(" = ").append(formatV(instr.src)).append('[').append(formatA(instr.offset)).append("];\n")
            }

            is Instr.Assign -> {
                w.append(indent).append(formatV(instr.dst)).append(" = ").append(formatA(instr.src)).append(";\n")
            }

            is Instr.Store -> {
                w.append(indent).append(formatV(instr.dst)).append('[').append(formatA(instr.offset))
                    .append("] = ").append(formatA(instr.value)).append(";\n")
            }

            is Instr.Binary -> {
                val opStr =
                    when (instr.op) {
                        BinaryOp.Add -> "${formatA(instr.lhs)} + ${formatA(instr.rhs)}"
                        BinaryOp.Mul -> "${formatA(instr.lhs)} * ${formatA(instr.rhs)}"
                        BinaryOp.Sub -> "${formatA(instr.lhs)} - ${formatA(instr.rhs)}"
                        BinaryOp.Div -> "${formatA(instr.lhs)} / ${formatA(instr.rhs)}"
                        BinaryOp.Min -> "fmin(${formatA(instr.lhs)}, ${formatA(instr.rhs)})"
                        BinaryOp.Max -> "fmax(${formatA(instr.lhs)}, ${formatA(instr.rhs)})"
                        BinaryOp.Mod -> "${formatA(instr.lhs)} % ${formatA(instr.rhs)}"
                    }
                w.append(indent).append(formatD(instr.dtype)).append(' ').append(varId)
                    .append(" = ").append(opStr).append(";\n")
            }

            is Instr.Unary -> {
                val opStr =
                    when (val op = instr.op) {
                        UnaryOp.Sqrt -> "sqrtf"
                        UnaryOp.Exp -> "expf"
                        UnaryOp.Sin -> "sinf"
                        UnaryOp.Cos -> "cosf"
                        UnaryOp.Neg -> "-"
                        UnaryOp.Id -> ""
                        is UnaryOp.Cast -> "(${formatD(op.dtype)})"
                    }
                w.append(indent).append(formatD(instr.dtype)).append(' ').append(varId)
                    .append(" = ").append(opStr).append('(').append(formatA(instr.arg)).append(");\n")
            }

            is Instr.SpecialInstr -> {
                when (instr.special) {
                    Special.ThreadIdx -> throw UgError.Msg("thread-id is currently not supported in the cpu backend")
                    Special.BlockIdx -> throw UgError.Msg("block-id is currently not supported in the cpu backend")
                }
            }

            is Instr.Barrier -> {}

            is Instr.ReduceLocal -> {
                val opStr =
                    when (instr.op) {
                        ReduceOp.Sum -> "block_reduce_sum"
                        ReduceOp.Min -> "block_reduce_min"
                        ReduceOp.Max -> "block_reduce_max"
                    }
                w.append(indent).append(formatD(instr.dtype)).append(' ').append(varId)
                    .append(" = ").append(opStr).append('(').append(formatA(instr.arg)).append(");\n")
            }
        }
    }
    w.append("}\n")
    if (depth > 0) throw UgError.Msg("unmatched Range")
}

public fun gen(funcName: String, kernel: Kernel): String {
    val sb = StringBuilder()
    gen(sb, funcName, kernel)
    return sb.toString()
}
