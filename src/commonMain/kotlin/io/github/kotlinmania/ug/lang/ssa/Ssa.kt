// port-lint: source lang.rs
package io.github.kotlinmania.ug.lang.ssa

import io.github.kotlinmania.ug.Const
import io.github.kotlinmania.ug.DType
import io.github.kotlinmania.ug.UgError
import io.github.kotlinmania.ug.lang.Arg
import io.github.kotlinmania.ug.lang.ArgId
import io.github.kotlinmania.ug.lang.BinaryOp
import io.github.kotlinmania.ug.lang.FlopsMem
import io.github.kotlinmania.ug.lang.LaunchConfig
import io.github.kotlinmania.ug.lang.ReduceOp
import io.github.kotlinmania.ug.lang.UgType
import io.github.kotlinmania.ug.lang.UnaryOp

public data class VarId(
    public val id: Int,
) : Comparable<VarId> {
    public fun asUsize(): Int = id

    override fun compareTo(other: VarId): Int = id.compareTo(other.id)

    override fun toString(): String = if (this == NULL) "NULL" else "v$id"

    public companion object {
        public val NULL: VarId = VarId(Int.MAX_VALUE)

        public fun new(id: Int): VarId = VarId(id)

        public fun null_(): VarId = NULL
    }
}

public fun VarId.toA(): A = A.Var(this)

public enum class Special {
    ThreadIdx,
    BlockIdx,
}

public sealed class A : Comparable<A> {
    public data class Var(
        public val id: VarId,
    ) : A()

    public data class Constant(
        public val value: Const,
    ) : A()

    override fun compareTo(other: A): Int = this.toString().compareTo(other.toString())

    public companion object {
        public fun from(v: VarId): A = Var(v)

        public fun from(v: Int): A = Constant(Const.I32(v))

        public fun from(v: Long): A = Constant(Const.I64(v))

        public fun from(v: Float): A = Constant(Const.F32(v))

        public fun from(c: Const): A = Constant(c)
    }
}

public sealed class Instr {
    public data class DefineAcc(
        public val value: Const,
    ) : Instr()

    public data class DefineGlobal(
        public val index: Int,
        public val dtype: DType,
    ) : Instr()

    public data class DefineLocal(
        public val size: Int,
        public val dtype: DType,
    ) : Instr()

    public data class SpecialInstr(
        public val special: Special,
    ) : Instr()

    public data class Constant(
        public val value: Const,
    ) : Instr()

    public data class Unary(
        public val op: UnaryOp,
        public val arg: A,
        public val dtype: DType,
    ) : Instr()

    public data class Binary(
        public val op: BinaryOp,
        public val lhs: A,
        public val rhs: A,
        public val dtype: DType,
    ) : Instr()

    public data class Range(
        public val lo: A,
        public val up: A,
        public val step: Int,
        public val endIdx: VarId,
    ) : Instr()

    public data class Load(
        public val src: VarId,
        public val offset: A,
        public val dtype: DType,
    ) : Instr()

    public data class Assign(
        public val dst: VarId,
        public val src: A,
    ) : Instr()

    public data class EndRange(
        public val startIdx: VarId,
    ) : Instr()

    public data class If(
        public val cond: A,
        public val endIdx: VarId,
    ) : Instr()

    public data object EndIf : Instr()

    public data class Store(
        public val dst: VarId,
        public val offset: A,
        public val value: A,
        public val dtype: DType,
    ) : Instr()

    public data object Barrier : Instr()

    public data class ReduceLocal(
        public val op: ReduceOp,
        public val arg: A,
        public val dtype: DType,
    ) : Instr()
}

public data class KernelArg(
    public val arg: Arg,
    public val index: Int,
)

public data class Kernel(
    public val instrs: List<Instr>,
    public val args: List<KernelArg>,
    public val launchConfig: LaunchConfig,
) {
    public fun flopsMemPerThread(): FlopsMem {
        var flops = 0
        var mem = 0
        val mults = mutableListOf<Int>()
        var mult = 1
        for (instr in instrs) {
            when (instr) {
                is Instr.Load -> {
                    mem += mult * instr.dtype.sizeInBytes
                }

                is Instr.Store -> {
                    mem += mult * instr.dtype.sizeInBytes
                }

                is Instr.Range -> {
                    mults.add(mult)
                    val loVal: Long =
                        when (val lo = instr.lo) {
                            is A.Constant -> {
                                when (val c = lo.value) {
                                    is Const.I64 -> c.value
                                    is Const.I32 -> c.value.toLong()
                                    else -> UgError.Msg("range lo is not a const i32/i64").raise()
                                }
                            }

                            is A.Var -> {
                                when (val c = instrs[lo.id.id]) {
                                    is Instr.Constant -> {
                                        when (val v = c.value) {
                                            is Const.I64 -> v.value
                                            is Const.I32 -> v.value.toLong()
                                            else -> UgError.Msg("range lo is not a const").raise()
                                        }
                                    }

                                    else -> {
                                        UgError.Msg("range lo is not a const").raise()
                                    }
                                }
                            }
                        }
                    val upVal: Long =
                        when (val up = instr.up) {
                            is A.Constant -> {
                                when (val c = up.value) {
                                    is Const.I64 -> c.value
                                    is Const.I32 -> c.value.toLong()
                                    else -> UgError.Msg("range up is not a const i32/i64").raise()
                                }
                            }

                            is A.Var -> {
                                when (val c = instrs[up.id.id]) {
                                    is Instr.Constant -> {
                                        when (val v = c.value) {
                                            is Const.I64 -> v.value
                                            is Const.I32 -> v.value.toLong()
                                            else -> UgError.Msg("range up is not a const").raise()
                                        }
                                    }

                                    else -> {
                                        UgError.Msg("range up is not a const").raise()
                                    }
                                }
                            }
                        }
                    val count = maxOf(0L, upVal - loVal).toInt() / instr.step
                    mult *= count
                }

                is Instr.EndRange -> {
                    if (mults.isEmpty()) UgError.Msg("unexpected EndRange").raise()
                    mult = mults.removeAt(mults.size - 1)
                }

                is Instr.Unary, is Instr.Binary -> {
                    flops += mult
                }

                else -> {}
            }
        }
        return FlopsMem(flops, mem)
    }

    override fun toString(): String {
        val sb = StringBuilder()
        for ((varId, instr) in instrs.withIndex()) {
            sb
                .append(varId.toString().padStart(3, '0'))
                .append(' ')
                .append(instr)
                .append('\n')
        }
        return sb.toString()
    }

    public companion object {
        public fun fromInstrs(instrs: List<Instr>): Kernel {
            val args = mutableListOf<KernelArg>()
            for ((lineNo, instr) in instrs.withIndex()) {
                if (instr is Instr.DefineGlobal) {
                    if (instr.index != args.size) {
                        UgError.Msg("unexpected order for arguments in kernel $instrs").raise()
                    }
                    val argId = ArgId.new()
                    val arg = Arg.new(argId, UgType.Ptr(instr.dtype))
                    args.add(KernelArg(arg, lineNo))
                }
            }
            val launchConfig = LaunchConfig(gridDim = 1, blockDim = 1, sharedMem = 0)
            return Kernel(instrs, args, launchConfig)
        }
    }
}
