// port-lint: source block.rs
package io.github.kotlinmania.ug.block

import io.github.kotlinmania.ug.Const
import io.github.kotlinmania.ug.DType
import io.github.kotlinmania.ug.UgError
import io.github.kotlinmania.ug.lang.BinaryOp
import io.github.kotlinmania.ug.lang.UnaryOp
import io.github.kotlinmania.ug.lang.ssa.A
import io.github.kotlinmania.ug.lang.ssa.Instr
import io.github.kotlinmania.ug.lang.ssa.VarId

internal data class Id(
    val id: Int,
) : Comparable<Id> {
    fun toVarId(): VarId = VarId(id)

    fun toA(): A = A.Var(VarId(id))

    override fun compareTo(other: Id): Int = id.compareTo(other.id)

    companion object {
        private var counter: Int = 1

        fun new(): Id {
            val v = counter
            counter += 1
            return Id(v)
        }

        fun fromVarId(v: VarId): Id = Id(v.id)
    }
}

internal data class Range(
    val rangeId: Id,
    val erangeId: Id,
) {
    fun id(): Id = rangeId
}

internal class Block(
    val instrs: MutableList<Pair<Id, Instr>> = mutableListOf(),
) {
    internal fun range(lo: A, up: A, step: Int): Range {
        val rangeId = Id.new()
        val erangeId = Id.new()
        val range = Instr.Range(lo = lo, up = up, step = step, endIdx = erangeId.toVarId())
        instrs.add(Pair(rangeId, range))
        return Range(rangeId, erangeId)
    }

    internal fun endRange(range: Range) {
        val erange = Instr.EndRange(startIdx = range.rangeId.toVarId())
        instrs.add(Pair(range.erangeId, erange))
    }

    internal fun add(srcId: Id, v: Int): Id {
        if (v == 0) {
            return srcId
        }
        val dstId = Id.new()
        instrs.add(
            Pair(
                dstId,
                Instr.Binary(
                    op = BinaryOp.Add,
                    lhs = srcId.toA(),
                    rhs = A.from(v),
                    dtype = DType.I32,
                ),
            ),
        )
        return dstId
    }

    internal fun mul(srcId: Id, v: Int): Id {
        if (v == 1) {
            return srcId
        }
        val dstId = Id.new()
        instrs.add(
            Pair(
                dstId,
                Instr.Binary(
                    op = BinaryOp.Mul,
                    lhs = srcId.toA(),
                    rhs = A.from(v),
                    dtype = DType.I32,
                ),
            ),
        )
        return dstId
    }

    internal fun cst(c: Const): Id = push(Instr.Constant(c))

    internal fun cst(c: Int): Id = cst(Const.I32(c))

    internal fun push(inst: Instr): Id {
        val id = Id.new()
        instrs.add(Pair(id, inst))
        return id
    }

    internal fun unary(op: UnaryOp, arg: A, dtype: DType): Id {
        val id = Id.new()
        instrs.add(Pair(id, Instr.Unary(op, arg, dtype)))
        return id
    }

    internal fun unary(op: UnaryOp, arg: Id, dtype: DType): Id = unary(op, arg.toA(), dtype)

    internal fun binary(op: BinaryOp, lhs: A, rhs: A, dtype: DType): Id {
        val id = Id.new()
        instrs.add(Pair(id, Instr.Binary(op, lhs, rhs, dtype)))
        return id
    }

    internal fun binary(op: BinaryOp, lhs: Id, rhs: Id, dtype: DType): Id = binary(op, lhs.toA(), rhs.toA(), dtype)

    internal fun relocate(): List<Instr> {
        val perId = mutableMapOf<Id, VarId>()
        for ((lineIdx, pair) in instrs.withIndex()) {
            val varId = VarId.new(lineIdx)
            perId[pair.first] = varId
        }
        val result = mutableListOf<Instr>()
        for ((_, instr) in instrs) {
            fun getId(id: VarId): VarId =
                perId[Id.fromVarId(id)] ?: UgError.Msg("id not found $id").raise()

            fun getA(a: A): A =
                when (a) {
                    is A.Var -> A.Var(getId(a.id))
                    is A.Constant -> a
                }

            val relocated =
                when (instr) {
                    is Instr.Store -> {
                        Instr.Store(
                            dst = getId(instr.dst),
                            offset = getA(instr.offset),
                            value = getA(instr.value),
                            dtype = instr.dtype,
                        )
                    }

                    is Instr.If -> {
                        Instr.If(
                            cond = getA(instr.cond),
                            endIdx = getId(instr.endIdx),
                        )
                    }

                    is Instr.Range -> {
                        Instr.Range(
                            lo = getA(instr.lo),
                            up = getA(instr.up),
                            step = instr.step,
                            endIdx = getId(instr.endIdx),
                        )
                    }

                    is Instr.Load -> {
                        Instr.Load(
                            src = getId(instr.src),
                            offset = getA(instr.offset),
                            dtype = instr.dtype,
                        )
                    }

                    is Instr.Constant -> {
                        instr
                    }

                    is Instr.Binary -> {
                        Instr.Binary(
                            op = instr.op,
                            lhs = getA(instr.lhs),
                            rhs = getA(instr.rhs),
                            dtype = instr.dtype,
                        )
                    }

                    is Instr.Unary -> {
                        Instr.Unary(
                            op = instr.op,
                            arg = getA(instr.arg),
                            dtype = instr.dtype,
                        )
                    }

                    is Instr.DefineAcc -> {
                        instr
                    }

                    is Instr.Assign -> {
                        Instr.Assign(
                            dst = getId(instr.dst),
                            src = getA(instr.src),
                        )
                    }

                    is Instr.SpecialInstr -> {
                        instr
                    }

                    is Instr.DefineLocal -> {
                        instr
                    }

                    is Instr.DefineGlobal -> {
                        instr
                    }

                    is Instr.Barrier -> {
                        instr
                    }

                    is Instr.ReduceLocal -> {
                        Instr.ReduceLocal(
                            op = instr.op,
                            arg = getA(instr.arg),
                            dtype = instr.dtype,
                        )
                    }

                    is Instr.EndIf -> {
                        instr
                    }

                    is Instr.EndRange -> {
                        Instr.EndRange(
                            startIdx = getId(instr.startIdx),
                        )
                    }
                }
            result.add(relocated)
        }
        return result
    }

    internal companion object {
        internal fun empty(): Block = Block()

        internal fun new(instrs: List<Pair<Id, Instr>>): Block = Block(instrs.toMutableList())
    }
}
