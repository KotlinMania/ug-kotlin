// port-lint: source interpreter.rs
package io.github.kotlinmania.ug.interpreter

import io.github.kotlinmania.ug.Const
import io.github.kotlinmania.ug.DType
import io.github.kotlinmania.ug.UgError
import io.github.kotlinmania.ug.lang.BinaryOp
import io.github.kotlinmania.ug.lang.ReduceOp
import io.github.kotlinmania.ug.lang.UnaryOp
import io.github.kotlinmania.ug.lang.ssa.A
import io.github.kotlinmania.ug.lang.ssa.Instr
import io.github.kotlinmania.ug.lang.ssa.Kernel
import io.github.kotlinmania.ug.lang.ssa.Special
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.sqrt

public sealed class Buffer {
    public data class F32(
        public val data: FloatArray,
    ) : Buffer() {
        public constructor(data: List<Float>) : this(data.toFloatArray())

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is F32) return false
            return data.contentEquals(other.data)
        }

        override fun hashCode(): Int = data.contentHashCode()
    }

    public data class I32(
        public val data: IntArray,
    ) : Buffer() {
        public constructor(data: List<Int>) : this(data.toIntArray())

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is I32) return false
            return data.contentEquals(other.data)
        }

        override fun hashCode(): Int = data.contentHashCode()
    }

    public data class I64(
        public val data: LongArray,
    ) : Buffer() {
        public constructor(data: List<Long>) : this(data.toLongArray())

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is I64) return false
            return data.contentEquals(other.data)
        }

        override fun hashCode(): Int = data.contentHashCode()
    }

    public fun asF32(): FloatArray =
        when (this) {
            is F32 -> data
            else -> UgError.Msg("buffer is not f32").raise()
        }

    public fun asI32(): IntArray =
        when (this) {
            is I32 -> data
            else -> UgError.Msg("buffer is not i32").raise()
        }

    public fun asI64(): LongArray =
        when (this) {
            is I64 -> data
            else -> UgError.Msg("buffer is not i64").raise()
        }
}

public sealed class Value {
    public data object None : Value()

    public data class Ptr(
        public val index: Int,
    ) : Value()

    public data class LocalPtr(
        public val index: Int,
    ) : Value()

    public data class F32Val(
        public val values: FloatArray,
    ) : Value() {
        override fun equals(other: Any?): Boolean = other is F32Val && values.contentEquals(other.values)

        override fun hashCode(): Int = values.contentHashCode()
    }

    public data class I32Val(
        public val values: IntArray,
    ) : Value() {
        override fun equals(other: Any?): Boolean = other is I32Val && values.contentEquals(other.values)

        override fun hashCode(): Int = values.contentHashCode()
    }
}

public fun evalSsa(
    kernel: Kernel,
    buffers: List<Buffer>,
    args: List<Buffer> = emptyList(),
    gridIdx: Int = 0,
    warpSize: Int = 1,
) {
    val instrs = kernel.instrs
    val values = arrayOfNulls<Value>(instrs.size)
    val localBuffers = mutableListOf<Buffer>()

    fun getA(a: A): Value =
        when (a) {
            is A.Var -> {
                values[a.id.id] ?: UgError.Msg("uninitialized var ${a.id}").raise()
            }

            is A.Constant -> {
                when (val c = a.value) {
                    is Const.F32 -> Value.F32Val(FloatArray(warpSize) { c.value })
                    is Const.I32 -> Value.I32Val(IntArray(warpSize) { c.value })
                    is Const.I64 -> Value.I32Val(IntArray(warpSize) { c.value.toInt() })
                    is Const.F16 -> Value.F32Val(FloatArray(warpSize) { c.toHalfF16().toFloat() })
                    is Const.BF16 -> Value.F32Val(FloatArray(warpSize) { c.toHalfBF16().toFloat() })
                }
            }
        }

    var currentIdx = 0
    while (currentIdx < instrs.size) {
        val instr = instrs[currentIdx]
        var jumpIdx: Int? = null

        when (instr) {
            is Instr.DefineGlobal -> {
                values[currentIdx] = Value.Ptr(instr.index)
            }

            is Instr.DefineLocal -> {
                val buf =
                    when (instr.dtype) {
                        DType.F32 -> Buffer.F32(FloatArray(instr.size))
                        DType.I32 -> Buffer.I32(IntArray(instr.size))
                        DType.I64 -> Buffer.I64(LongArray(instr.size))
                        else -> Buffer.F32(FloatArray(instr.size))
                    }
                localBuffers.add(buf)
                values[currentIdx] = Value.LocalPtr(localBuffers.size - 1)
            }

            is Instr.DefineAcc -> {
                values[currentIdx] =
                    when (val c = instr.value) {
                        is Const.F32 -> Value.F32Val(FloatArray(warpSize) { c.value })
                        is Const.I32 -> Value.I32Val(IntArray(warpSize) { c.value })
                        is Const.I64 -> Value.I32Val(IntArray(warpSize) { c.value.toInt() })
                        else -> Value.F32Val(FloatArray(warpSize) { 0f })
                    }
            }

            is Instr.Constant -> {
                values[currentIdx] =
                    when (val c = instr.value) {
                        is Const.F32 -> Value.F32Val(FloatArray(warpSize) { c.value })
                        is Const.I32 -> Value.I32Val(IntArray(warpSize) { c.value })
                        is Const.I64 -> Value.I32Val(IntArray(warpSize) { c.value.toInt() })
                        else -> Value.F32Val(FloatArray(warpSize) { 0f })
                    }
            }

            is Instr.SpecialInstr -> {
                values[currentIdx] =
                    when (instr.special) {
                        Special.BlockIdx -> Value.I32Val(IntArray(warpSize) { gridIdx })
                        Special.ThreadIdx -> Value.I32Val(IntArray(warpSize) { it })
                    }
            }

            is Instr.Range -> {
                val current = values[currentIdx]
                if (current == null) {
                    values[currentIdx] = getA(instr.lo)
                } else {
                    val curInts = (current as Value.I32Val).values
                    val upInts = (getA(instr.up) as Value.I32Val).values
                    val nextInts = IntArray(warpSize) { curInts[it] + instr.step }
                    var allJump = true
                    for (i in 0 until warpSize) {
                        if (nextInts[i] < upInts[i]) {
                            allJump = false
                        }
                    }
                    if (allJump) {
                        values[currentIdx] = null
                        jumpIdx = instr.endIdx.id + 1
                    } else {
                        values[currentIdx] = Value.I32Val(nextInts)
                    }
                }
            }

            is Instr.EndRange -> {
                jumpIdx = instr.startIdx.id
            }

            is Instr.If -> {
                val condInts = (getA(instr.cond) as Value.I32Val).values
                val allZero = condInts.all { it == 0 }
                if (allZero) {
                    jumpIdx = instr.endIdx.id + 1
                }
            }

            is Instr.EndIf -> {}

            is Instr.Assign -> {
                values[instr.dst.id] = getA(instr.src)
            }

            is Instr.Load -> {
                val off = (getA(instr.offset) as Value.I32Val).values
                val ptrVal = values[instr.src.id] ?: UgError.Msg("src pointer not found").raise()
                val buf =
                    when (ptrVal) {
                        is Value.Ptr -> buffers[ptrVal.index]
                        is Value.LocalPtr -> localBuffers[ptrVal.index]
                        else -> UgError.Msg("unexpected pointer value").raise()
                    }
                values[currentIdx] =
                    when (buf) {
                        is Buffer.F32 -> Value.F32Val(FloatArray(warpSize) { buf.data[off[it]] })
                        is Buffer.I32 -> Value.I32Val(IntArray(warpSize) { buf.data[off[it]] })
                        is Buffer.I64 -> Value.I32Val(IntArray(warpSize) { buf.data[off[it]].toInt() })
                    }
            }

            is Instr.Store -> {
                val off = (getA(instr.offset) as Value.I32Val).values
                val value = getA(instr.value)
                val ptrVal = values[instr.dst.id] ?: UgError.Msg("dst pointer not found").raise()
                val buf =
                    when (ptrVal) {
                        is Value.Ptr -> buffers[ptrVal.index]
                        is Value.LocalPtr -> localBuffers[ptrVal.index]
                        else -> UgError.Msg("unexpected pointer value").raise()
                    }
                when (buf) {
                    is Buffer.F32 -> {
                        val fVal =
                            when (value) {
                                is Value.F32Val -> value.values
                                is Value.I32Val -> FloatArray(warpSize) { value.values[it].toFloat() }
                                else -> FloatArray(warpSize) { 0f }
                            }
                        for (i in 0 until warpSize) {
                            buf.data[off[i]] = fVal[i]
                        }
                    }

                    is Buffer.I32 -> {
                        val iVal =
                            when (value) {
                                is Value.I32Val -> value.values
                                is Value.F32Val -> IntArray(warpSize) { value.values[it].toInt() }
                                else -> IntArray(warpSize) { 0 }
                            }
                        for (i in 0 until warpSize) {
                            buf.data[off[i]] = iVal[i]
                        }
                    }

                    is Buffer.I64 -> {
                        val iVal =
                            when (value) {
                                is Value.I32Val -> value.values
                                is Value.F32Val -> IntArray(warpSize) { value.values[it].toInt() }
                                else -> IntArray(warpSize) { 0 }
                            }
                        for (i in 0 until warpSize) {
                            buf.data[off[i]] = iVal[i].toLong()
                        }
                    }
                }
            }

            is Instr.Binary -> {
                val lhs = getA(instr.lhs)
                val rhs = getA(instr.rhs)
                values[currentIdx] =
                    when {
                        lhs is Value.F32Val && rhs is Value.F32Val -> {
                            val res = FloatArray(warpSize)
                            for (i in 0 until warpSize) {
                                val l = lhs.values[i]
                                val r = rhs.values[i]
                                res[i] =
                                    when (instr.op) {
                                        BinaryOp.Add -> l + r
                                        BinaryOp.Sub -> l - r
                                        BinaryOp.Mul -> l * r
                                        BinaryOp.Div -> l / r
                                        BinaryOp.Min -> minOf(l, r)
                                        BinaryOp.Max -> maxOf(l, r)
                                        BinaryOp.Mod -> l % r
                                    }
                            }
                            Value.F32Val(res)
                        }

                        lhs is Value.I32Val && rhs is Value.I32Val -> {
                            val res = IntArray(warpSize)
                            for (i in 0 until warpSize) {
                                val l = lhs.values[i]
                                val r = rhs.values[i]
                                res[i] =
                                    when (instr.op) {
                                        BinaryOp.Add -> l + r
                                        BinaryOp.Sub -> l - r
                                        BinaryOp.Mul -> l * r
                                        BinaryOp.Div -> l / r
                                        BinaryOp.Min -> minOf(l, r)
                                        BinaryOp.Max -> maxOf(l, r)
                                        BinaryOp.Mod -> l % r
                                    }
                            }
                            Value.I32Val(res)
                        }

                        else -> {
                            UgError.Msg("type mismatch in binary op ${instr.op}").raise()
                        }
                    }
            }

            is Instr.Unary -> {
                val arg = getA(instr.arg)
                values[currentIdx] =
                    when (instr.op) {
                        UnaryOp.Id -> {
                            arg
                        }

                        UnaryOp.Neg -> {
                            when (arg) {
                                is Value.F32Val -> Value.F32Val(FloatArray(warpSize) { -arg.values[it] })
                                is Value.I32Val -> Value.I32Val(IntArray(warpSize) { -arg.values[it] })
                                else -> UgError.Msg("dtype mismatch for Neg $arg").raise()
                            }
                        }

                        UnaryOp.Exp -> {
                            when (arg) {
                                is Value.F32Val -> Value.F32Val(FloatArray(warpSize) { exp(arg.values[it]) })
                                else -> UgError.Msg("dtype mismatch for Exp $arg").raise()
                            }
                        }

                        UnaryOp.Sqrt -> {
                            when (arg) {
                                is Value.F32Val -> Value.F32Val(FloatArray(warpSize) { sqrt(arg.values[it]) })
                                else -> UgError.Msg("dtype mismatch for Sqrt $arg").raise()
                            }
                        }

                        UnaryOp.Sin -> {
                            when (arg) {
                                is Value.F32Val -> Value.F32Val(FloatArray(warpSize) { sin(arg.values[it]) })
                                else -> UgError.Msg("dtype mismatch for Sin $arg").raise()
                            }
                        }

                        UnaryOp.Cos -> {
                            when (arg) {
                                is Value.F32Val -> Value.F32Val(FloatArray(warpSize) { cos(arg.values[it]) })
                                else -> UgError.Msg("dtype mismatch for Cos $arg").raise()
                            }
                        }

                        is UnaryOp.Cast -> {
                            when (arg) {
                                is Value.F32Val -> {
                                    when (instr.dtype) {
                                        DType.F32 -> Value.F32Val(arg.values.copyOf())
                                        DType.I32 -> Value.I32Val(IntArray(warpSize) { arg.values[it].toInt() })
                                        else -> UgError.Msg("dtype mismatch for Cast $arg to ${instr.dtype}").raise()
                                    }
                                }

                                is Value.I32Val -> {
                                    when (instr.dtype) {
                                        DType.I32 -> Value.I32Val(arg.values.copyOf())
                                        DType.F32 -> Value.F32Val(FloatArray(warpSize) { arg.values[it].toFloat() })
                                        else -> UgError.Msg("dtype mismatch for Cast $arg to ${instr.dtype}").raise()
                                    }
                                }

                                else -> {
                                    UgError.Msg("dtype mismatch for Cast $arg").raise()
                                }
                            }
                        }
                    }
            }

            is Instr.Barrier -> {}

            is Instr.ReduceLocal -> {
                val arg = getA(instr.arg)
                values[currentIdx] =
                    when (arg) {
                        is Value.F32Val -> {
                            var acc =
                                when (instr.op) {
                                    ReduceOp.Sum -> 0.0f
                                    ReduceOp.Min -> Float.POSITIVE_INFINITY
                                    ReduceOp.Max -> Float.NEGATIVE_INFINITY
                                }
                            for (v in arg.values) {
                                acc =
                                    when (instr.op) {
                                        ReduceOp.Sum -> acc + v
                                        ReduceOp.Min -> minOf(acc, v)
                                        ReduceOp.Max -> maxOf(acc, v)
                                    }
                            }
                            Value.F32Val(FloatArray(warpSize) { acc })
                        }

                        is Value.I32Val -> {
                            var acc =
                                when (instr.op) {
                                    ReduceOp.Sum -> 0
                                    ReduceOp.Min -> Int.MAX_VALUE
                                    ReduceOp.Max -> Int.MIN_VALUE
                                }
                            for (v in arg.values) {
                                acc =
                                    when (instr.op) {
                                        ReduceOp.Sum -> acc + v
                                        ReduceOp.Min -> minOf(acc, v)
                                        ReduceOp.Max -> maxOf(acc, v)
                                    }
                            }
                            Value.I32Val(IntArray(warpSize) { acc })
                        }

                        else -> {
                            UgError.Msg("unexpected reduce argument").raise()
                        }
                    }
            }
        }

        currentIdx = jumpIdx ?: (currentIdx + 1)
    }
}
