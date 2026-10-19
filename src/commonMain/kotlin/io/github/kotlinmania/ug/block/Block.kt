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

public data class Id(
    public val id: Int,
) : Comparable<Id> {
    public fun toVarId(): VarId = VarId(id)

    public fun toA(): A = A.Var(VarId(id))

    override fun compareTo(other: Id): Int = id.compareTo(other.id)

    public companion object {
        private var counter: Int = 1

        public fun new(): Id {
            val v = counter
            counter += 1
            return Id(v)
        }

        public fun fromVarId(v: VarId): Id = Id(v.id)
    }
}

public data class Range(
    public val rangeId: Id,
    public val erangeId: Id,
) {
    public fun id(): Id = rangeId
}

public class Block(
    public val instrs: MutableList<Pair<Id, Instr>> = mutableListOf(),
) {
    public fun range(lo: A, up: A, step: Int): Range {
        val rangeId = Id.new()
        val erangeId = Id.new()
        val range = Instr.Range(lo = lo, up = up, step = step, endIdx = erangeId.toVarId())
        instrs.add(Pair(rangeId, range))
        return Range(rangeId, erangeId)
    }

    public fun endRange(range: Range) {
        val erange = Instr.EndRange(startIdx = range.rangeId.toVarId())
        instrs.add(Pair(range.erangeId, erange))
    }

    public fun add(srcId: Id, v: Int): Id {
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

    public fun mul(srcId: Id, v: Int): Id {
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

    public fun cst(c: Const): Id = push(Instr.Constant(c))

    public fun cst(c: Int): Id = cst(Const.I32(c))

    public fun push(inst: Instr): Id {
        val id = Id.new()
        instrs.add(Pair(id, inst))
        return id
    }

    public fun unary(op: UnaryOp, arg: A, dtype: DType): Id {
        val id = Id.new()
        instrs.add(Pair(id, Instr.Unary(op, arg, dtype)))
        return id
    }

    public fun unary(op: UnaryOp, arg: Id, dtype: DType): Id = unary(op, arg.toA(), dtype)

    public fun binary(op: BinaryOp, lhs: A, rhs: A, dtype: DType): Id {
        val id = Id.new()
        instrs.add(Pair(id, Instr.Binary(op, lhs, rhs, dtype)))
        return id
    }

    public fun binary(op: BinaryOp, lhs: Id, rhs: Id, dtype: DType): Id = binary(op, lhs.toA(), rhs.toA(), dtype)

    public fun relocate(): List<Instr> {
        val perId = mutableMapOf<Id, VarId>()
        for ((lineIdx, pair) in instrs.withIndex()) {
            val varId = VarId.new(lineIdx)
            perId[pair.first] = varId
        }
        val result = mutableListOf<Instr>()
        for ((_, instr) in instrs) {
            fun getId(id: VarId): VarId =
                perId[Id.fromVarId(id)] ?: throw UgError.Msg("id not found $id")

            fun getA(a: A): A =
                when (a) {
                    is A.Var -> A.Var(getId(a.id))
                    is A.Constant -> a
                }

            val relocated =
                when (instr) {
                    is Instr.Store ->
                        Instr.Store(
                            dst = getId(instr.dst),
                            offset = getA(instr.offset),
                            value = getA(instr.value),
                            dtype = instr.dtype,
                        )

                    is Instr.If ->
                        Instr.If(
                            cond = getA(instr.cond),
                            endIdx = getId(instr.endIdx),
                        )

                    is Instr.Range ->
                        Instr.Range(
                            lo = getA(instr.lo),
                            up = getA(instr.up),
                            step = instr.step,
                            endIdx = getId(instr.endIdx),
                        )

                    is Instr.Load ->
                        Instr.Load(
                            src = getId(instr.src),
                            offset = getA(instr.offset),
                            dtype = instr.dtype,
                        )

                    is Instr.Constant -> instr

                    is Instr.Binary ->
                        Instr.Binary(
                            op = instr.op,
                            lhs = getA(instr.lhs),
                            rhs = getA(instr.rhs),
                            dtype = instr.dtype,
                        )

                    is Instr.Unary ->
                        Instr.Unary(
                            op = instr.op,
                            arg = getA(instr.arg),
                            dtype = instr.dtype,
                        )

                    is Instr.DefineAcc -> instr

                    is Instr.Assign ->
                        Instr.Assign(
                            dst = getId(instr.dst),
                            src = getA(instr.src),
                        )

                    is Instr.SpecialInstr -> instr
                    is Instr.DefineLocal -> instr
                    is Instr.DefineGlobal -> instr
                    is Instr.Barrier -> instr

                    is Instr.ReduceLocal ->
                        Instr.ReduceLocal(
                            op = instr.op,
                            arg = getA(instr.arg),
                            dtype = instr.dtype,
                        )

                    is Instr.EndIf -> instr

                    is Instr.EndRange ->
                        Instr.EndRange(
                            startIdx = getId(instr.startIdx),
                        )
                }
            result.add(relocated)
        }
        return result
    }

    public companion object {
        public fun empty(): Block = Block()

        public fun new(instrs: List<Pair<Id, Instr>>): Block = Block(instrs.toMutableList())
    }
}
