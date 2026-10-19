// port-lint: source lower.rs
package io.github.kotlinmania.ug.lower

import io.github.kotlinmania.ug.Const
import io.github.kotlinmania.ug.DType
import io.github.kotlinmania.ug.UgError
import io.github.kotlinmania.ug.block.Block
import io.github.kotlinmania.ug.block.Id
import io.github.kotlinmania.ug.lang.ArgId
import io.github.kotlinmania.ug.lang.BinaryOp
import io.github.kotlinmania.ug.lang.Expr
import io.github.kotlinmania.ug.lang.ExprNode
import io.github.kotlinmania.ug.lang.IndexExpr
import io.github.kotlinmania.ug.lang.IndexExprNode
import io.github.kotlinmania.ug.lang.Kernel
import io.github.kotlinmania.ug.lang.Ops
import io.github.kotlinmania.ug.lang.StridedSlice
import io.github.kotlinmania.ug.lang.UgType
import io.github.kotlinmania.ug.lang.ssa.A
import io.github.kotlinmania.ug.lang.ssa.Instr
import io.github.kotlinmania.ug.lang.ssa.Special
import io.github.kotlinmania.ug.lang.ssa.VarId
import io.github.kotlinmania.ug.lang.ssa.toA

internal fun lowerIndexExprNode(node: IndexExprNode): Pair<Id, Block> {
    val dstId = Id.new()
    val block =
        when (val e = node.expr) {
            is IndexExpr.Add -> {
                val (lhsId, lhsB) = lowerIndexExprNode(e.lhs)
                val (rhsId, rhsB) = lowerIndexExprNode(e.rhs)
                val instr =
                    Instr.Binary(
                        op = BinaryOp.Add,
                        lhs = lhsId.toA(),
                        rhs = rhsId.toA(),
                        dtype = DType.I32,
                    )
                Block((lhsB.instrs + rhsB.instrs + Pair(dstId, instr)).toMutableList())
            }

            is IndexExpr.Mul -> {
                val (lhsId, lhsB) = lowerIndexExprNode(e.lhs)
                val (rhsId, rhsB) = lowerIndexExprNode(e.rhs)
                val instr =
                    Instr.Binary(
                        op = BinaryOp.Mul,
                        lhs = lhsId.toA(),
                        rhs = rhsId.toA(),
                        dtype = DType.I32,
                    )
                Block((lhsB.instrs + rhsB.instrs + Pair(dstId, instr)).toMutableList())
            }

            is IndexExpr.Constant -> {
                val instr = Instr.Constant(Const.I32(e.value))
                Block(mutableListOf(Pair(dstId, instr)))
            }

            is IndexExpr.ProgramId -> {
                val instr = Instr.SpecialInstr(Special.BlockIdx)
                Block(mutableListOf(Pair(dstId, instr)))
            }
        }
    return Pair(dstId, block)
}

internal fun lowerStridedSlice(
    slice: StridedSlice,
    rangeId: Id,
    perArg: Map<ArgId, VarId>,
): Triple<VarId, VarId, Block> {
    val (offI, offB) = lowerIndexExprNode(slice.offset)
    val (strideI, strideB) = lowerIndexExprNode(slice.stride)
    val ptrI = perArg[slice.ptr.id] ?: throw UgError.Msg("unknown arg ${slice.ptr.id}")

    return if (slice.offset.asConst() == 0 && slice.stride.asConst() == 1) {
        Triple(ptrI, rangeId.toVarId(), Block.empty())
    } else if (slice.stride.asConst() == 1) {
        val indexI = Id.new()
        val indexB =
            listOf(
                Pair(
                    indexI,
                    Instr.Binary(
                        op = BinaryOp.Add,
                        lhs = rangeId.toA(),
                        rhs = offI.toA(),
                        dtype = DType.I32,
                    ),
                ),
            )
        val instrs = offB.instrs + strideB.instrs + indexB
        Triple(ptrI, indexI.toVarId(), Block(instrs.toMutableList()))
    } else {
        val mulI = Id.new()
        val indexI = Id.new()
        val indexB =
            listOf(
                Pair(
                    mulI,
                    Instr.Binary(
                        op = BinaryOp.Mul,
                        lhs = rangeId.toA(),
                        rhs = strideI.toA(),
                        dtype = DType.I32,
                    ),
                ),
                Pair(
                    indexI,
                    Instr.Binary(
                        op = BinaryOp.Add,
                        lhs = mulI.toA(),
                        rhs = offI.toA(),
                        dtype = DType.I32,
                    ),
                ),
            )
        val instrs = offB.instrs + strideB.instrs + indexB
        Triple(ptrI, indexI.toVarId(), Block(instrs.toMutableList()))
    }
}

internal fun lowerExprNode(
    node: ExprNode,
    rangeId: Id,
    perArg: Map<ArgId, VarId>,
): Pair<Id, Block> {
    val dstId = Id.new()
    val dtype = node.dtype
    val block =
        when (val e = node.expr) {
            is Expr.Load -> {
                val (ptrI, offI, srcB) = lowerStridedSlice(e.slice, rangeId, perArg)
                val instr = Instr.Load(src = ptrI, offset = offI.toA(), dtype = dtype)
                srcB.instrs.add(Pair(dstId, instr))
                srcB
            }

            is Expr.Constant -> Block(mutableListOf(Pair(dstId, Instr.Constant(e.c))))
            is Expr.Range -> throw UgError.Msg("range is not supported yet")
            is Expr.Unary -> {
                val (argId, argB) = lowerExprNode(e.arg, rangeId, perArg)
                val instr = Instr.Unary(op = e.op, arg = argId.toA(), dtype = dtype)
                Block((argB.instrs + Pair(dstId, instr)).toMutableList())
            }

            is Expr.Binary -> {
                val (lhsId, lhsB) = lowerExprNode(e.lhs, rangeId, perArg)
                val (rhsId, rhsB) = lowerExprNode(e.rhs, rangeId, perArg)
                val instr = Instr.Binary(op = e.op, lhs = lhsId.toA(), rhs = rhsId.toA(), dtype = dtype)
                Block((lhsB.instrs + rhsB.instrs + Pair(dstId, instr)).toMutableList())
            }
        }
    return Pair(dstId, block)
}

public fun lowerKernel(kernel: Kernel): io.github.kotlinmania.ug.lang.ssa.Kernel {
    val instrs = mutableListOf<Pair<Id, Instr>>()
    val perArg = mutableMapOf<ArgId, VarId>()
    for ((index, arg) in kernel.args.withIndex()) {
        val id = Id.new()
        val dtype =
            when (val t = arg.type) {
                is UgType.Ptr -> t.dtype
                is UgType.Value -> throw UgError.Msg("non-pointer arguments are not supported yet")
            }
        instrs.add(Pair(id, Instr.DefineGlobal(index, dtype)))
        perArg[arg.id] = id.toVarId()
    }

    for (op in kernel.ops) {
        val store = op as Ops.Store
        val len = store.dst.len
        val (lenI, lenB) = lowerIndexExprNode(len)
        instrs.addAll(lenB.instrs)
        val loId = Id.new()
        instrs.add(Pair(loId, Instr.Constant(Const.I32(0))))

        val rangeId = Id.new()
        val erangeId = Id.new()
        val range =
            Instr.Range(
                lo = loId.toA(),
                up = lenI.toA(),
                step = 1,
                endIdx = erangeId.toVarId(),
            )
        instrs.add(Pair(rangeId, range))

        val (srcI, srcB) = lowerExprNode(store.src, rangeId, perArg)
        instrs.addAll(srcB.instrs)
        val (ptrI, offI, dstB) = lowerStridedSlice(store.dst, rangeId, perArg)
        instrs.addAll(dstB.instrs)
        val storeInstr =
            Instr.Store(
                dst = ptrI,
                offset = offI.toA(),
                value = srcI.toA(),
                dtype = store.src.dtype,
            )
        instrs.add(Pair(Id.new(), storeInstr))

        val erange = Instr.EndRange(startIdx = rangeId.toVarId())
        instrs.add(Pair(erangeId, erange))
    }

    val block = Block(instrs)
    val relocated = block.relocate()
    return io.github.kotlinmania.ug.lang.ssa.Kernel.fromInstrs(relocated)
}

public fun Kernel.lower(): io.github.kotlinmania.ug.lang.ssa.Kernel = lowerKernel(this)
