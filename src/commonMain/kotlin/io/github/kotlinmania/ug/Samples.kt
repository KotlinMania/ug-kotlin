// port-lint: source samples.rs
package io.github.kotlinmania.ug

import io.github.kotlinmania.ug.block.Block
import io.github.kotlinmania.ug.block.Id
import io.github.kotlinmania.ug.lang.BinaryOp
import io.github.kotlinmania.ug.lang.ReduceOp
import io.github.kotlinmania.ug.lang.UnaryOp
import io.github.kotlinmania.ug.lang.ssa.A
import io.github.kotlinmania.ug.lang.ssa.Instr
import io.github.kotlinmania.ug.lang.ssa.Kernel
import io.github.kotlinmania.ug.lang.ssa.Special
import io.github.kotlinmania.ug.lang.ssa.VarId

public object Samples {
    public object ssa {
        private fun arg(index: Int, dtype: DType): Instr =
            Instr.DefineGlobal(index, dtype)

        public fun simpleAdd(vecLen: Int): Kernel {
            val v = { i: Int -> VarId(i) }
            val a = { i: Int -> A.Var(VarId(i)) }
            val i32 = { i: Int -> A.Constant(Const.I32(i)) }
            val dtype = DType.I32
            val instrs =
                listOf(
                    /* 0 */ arg(0, DType.I32),
                    /* 1 */ arg(1, DType.I32),
                    /* 2 */ arg(2, DType.I32),
                    /* 3 */ Instr.Range(lo = i32(0), up = i32(vecLen), endIdx = v(8), step = 1),
                    /* 4 */ Instr.Load(src = v(1), offset = a(3), dtype = dtype),
                    /* 5 */ Instr.Load(src = v(2), offset = a(3), dtype = dtype),
                    /* 6 */ Instr.Binary(op = BinaryOp.Add, lhs = a(4), rhs = a(5), dtype = dtype),
                    /* 7 */ Instr.Store(dst = v(0), offset = a(3), value = a(6), dtype = dtype),
                    /* 8 */ Instr.EndRange(startIdx = v(3)),
                )
            return Kernel.fromInstrs(instrs)
        }

        public fun simpleDotprod(vecLen: Int): Kernel {
            val v = { i: Int -> VarId(i) }
            val a = { i: Int -> A.Var(VarId(i)) }
            val dtype = DType.F32
            val instrs =
                listOf(
                    /* 0 */ arg(0, DType.F32),
                    /* 1 */ arg(1, DType.F32),
                    /* 2 */ arg(2, DType.F32),
                    /* 3 */ Instr.Constant(Const.I32(0)),
                    /* 4 */ Instr.Constant(Const.I32(vecLen)),
                    /* 5 */ Instr.DefineAcc(Const.F32(0.0f)),
                    /* 6 */ Instr.Range(lo = a(3), up = a(4), endIdx = v(12), step = 1),
                    /* 7 */ Instr.Load(src = v(1), offset = a(6), dtype = dtype),
                    /* 8 */ Instr.Load(src = v(2), offset = a(6), dtype = dtype),
                    /* 9 */ Instr.Binary(op = BinaryOp.Mul, lhs = a(7), rhs = a(8), dtype = dtype),
                    /* 10*/ Instr.Binary(op = BinaryOp.Add, lhs = a(9), rhs = a(5), dtype = dtype),
                    /* 11*/ Instr.Assign(dst = v(5), src = a(10)),
                    /* 12*/ Instr.EndRange(startIdx = v(6)),
                    /* 13*/ Instr.Store(dst = v(0), offset = a(3), value = a(5), dtype = dtype),
                )
            return Kernel.fromInstrs(instrs)
        }

        public fun exp(blockSize: Int): Kernel {
            val b = Block.empty()
            val dtype = DType.F32
            val srcI = b.push(arg(0, DType.F32))
            val dstI = b.push(arg(1, DType.F32))
            val gI = b.push(Instr.SpecialInstr(Special.BlockIdx))
            val lI = b.push(Instr.SpecialInstr(Special.ThreadIdx))
            val offI0 = b.mul(gI, blockSize)
            val offI = b.binary(BinaryOp.Add, offI0, lI, DType.I32)
            val loadI = b.push(Instr.Load(src = srcI.toVarId(), offset = offI.toA(), dtype = dtype))
            val valueI = b.unary(UnaryOp.Exp, loadI, dtype)
            b.push(
                Instr.Store(
                    dst = dstI.toVarId(),
                    offset = offI.toA(),
                    value = valueI.toA(),
                    dtype = dtype,
                ),
            )
            val instrs = b.relocate()
            return Kernel.fromInstrs(instrs)
        }

        public fun expBlock(dim2: Int, blockSize: Int): Kernel {
            if (dim2 % blockSize != 0) {
                ugBail("last-dim $dim2 must be divisible by block size $blockSize")
            }

            val b = Block.empty()
            val dtype = DType.F32
            val srcI = b.push(arg(0, DType.F32))
            val dstI = b.push(arg(1, DType.F32))
            val gI = b.push(Instr.SpecialInstr(Special.BlockIdx))
            val lI = b.push(Instr.SpecialInstr(Special.ThreadIdx))
            val offI0 = b.mul(gI, dim2)
            val offIBase = b.binary(BinaryOp.Add, offI0, lI, DType.I32)

            var i = 0
            while (i < dim2) {
                val offI = b.add(offIBase, i)
                val loadI = b.push(Instr.Load(src = srcI.toVarId(), offset = offI.toA(), dtype = dtype))
                val valueI = b.unary(UnaryOp.Exp, loadI, dtype)
                b.push(
                    Instr.Store(
                        dst = dstI.toVarId(),
                        offset = offI.toA(),
                        value = valueI.toA(),
                        dtype = dtype,
                    ),
                )
                i += blockSize
            }
            return Kernel.fromInstrs(b.relocate())
        }

        public fun softmaxBarrier(dim1: Int, dim2: Int): Kernel {
            val b = Block.empty()
            val dtype = DType.F32
            val srcI = b.push(arg(0, DType.F32))
            val dstI = b.push(arg(1, DType.F32))
            val stoI = b.push(Instr.DefineLocal(size = 2 * dim2, dtype = DType.F32)).toVarId()
            val gI = b.push(Instr.SpecialInstr(Special.BlockIdx))
            val lI = b.push(Instr.SpecialInstr(Special.ThreadIdx))
            val baseOffI = b.mul(gI, dim2)
            val globalOffI = b.binary(BinaryOp.Add, baseOffI, lI, DType.I32)
            val loadI = b.push(Instr.Load(src = srcI.toVarId(), offset = globalOffI.toA(), dtype = dtype))

            b.push(Instr.Store(dst = stoI, offset = lI.toA(), value = loadI.toA(), dtype = dtype))
            b.push(Instr.Barrier)
            var offset = 1
            var vI = loadI
            while (offset < dim2) {
                val offI = b.add(lI, offset)
                val mI = b.push(Instr.Load(src = stoI, offset = offI.toA(), dtype = dtype))
                vI = b.binary(BinaryOp.Max, mI, vI, dtype)
                b.push(Instr.Store(dst = stoI, offset = lI.toA(), value = vI.toA(), dtype = dtype))
                b.push(Instr.Barrier)
                offset *= 2
            }
            val maxI = b.push(Instr.Load(src = stoI, offset = A.from(0), dtype = dtype))

            val valueSub = b.binary(BinaryOp.Sub, loadI, maxI, dtype)
            val valueExp = b.unary(UnaryOp.Exp, valueSub, dtype)

            b.push(Instr.Store(dst = stoI, offset = lI.toA(), value = valueExp.toA(), dtype = dtype))
            b.push(Instr.Barrier)
            offset = 1
            var sumExpI = valueExp
            while (offset < dim2) {
                val offI = b.add(lI, offset)
                val mI = b.push(Instr.Load(src = stoI, offset = offI.toA(), dtype = dtype))
                sumExpI = b.binary(BinaryOp.Add, mI, sumExpI, dtype)
                b.push(Instr.Store(dst = stoI, offset = lI.toA(), value = sumExpI.toA(), dtype = dtype))
                b.push(Instr.Barrier)
                offset *= 2
            }
            val sumI = b.push(Instr.Load(src = stoI, offset = A.from(0), dtype = dtype))

            val valueI = b.binary(BinaryOp.Div, valueExp, sumI, dtype)
            b.push(
                Instr.Store(
                    dst = dstI.toVarId(),
                    offset = globalOffI.toA(),
                    value = valueI.toA(),
                    dtype = dtype,
                ),
            )
            return Kernel.fromInstrs(b.relocate())
        }

        public fun softmaxReduce(dim1: Int, dim2: Int): Kernel {
            val b = Block.empty()
            val dtype = DType.F32
            val srcI = b.push(arg(0, dtype))
            val dstI = b.push(arg(1, dtype))
            val gI = b.push(Instr.SpecialInstr(Special.BlockIdx))
            val lI = b.push(Instr.SpecialInstr(Special.ThreadIdx))
            val baseOffI = b.mul(gI, dim2)
            val globalOffI = b.binary(BinaryOp.Add, baseOffI, lI, DType.I32)
            val loadI = b.push(Instr.Load(src = srcI.toVarId(), offset = globalOffI.toA(), dtype = dtype))
            val maxI = b.push(Instr.ReduceLocal(op = ReduceOp.Max, arg = loadI.toA(), dtype = dtype))
            val valSub = b.binary(BinaryOp.Sub, loadI, maxI, dtype)
            val valExp = b.unary(UnaryOp.Exp, valSub, dtype)
            val sumI = b.push(Instr.ReduceLocal(op = ReduceOp.Sum, arg = valExp.toA(), dtype = dtype))
            val valueI = b.binary(BinaryOp.Div, valExp, sumI, dtype)
            b.push(
                Instr.Store(
                    dst = dstI.toVarId(),
                    offset = globalOffI.toA(),
                    value = valueI.toA(),
                    dtype = dtype,
                ),
            )
            return Kernel.fromInstrs(b.relocate())
        }

        public fun softmaxBlock(dim1: Int, dim2: Int, blockSize: Int): Kernel {
            if (dim2 % blockSize != 0) {
                ugBail("last-dim $dim2 must be divisible by block size $blockSize")
            }
            val perBlock = dim2 / blockSize
            val b = Block.empty()
            val dtype = DType.F32
            val srcI = b.push(arg(0, dtype))
            val dstI = b.push(arg(1, dtype))
            val gI = b.push(Instr.SpecialInstr(Special.BlockIdx))
            val lI = b.push(Instr.SpecialInstr(Special.ThreadIdx))
            val baseOffI = b.mul(gI, dim2)
            val globalOffI = b.binary(BinaryOp.Add, baseOffI, lI, DType.I32)

            val loadIs = mutableListOf<Id>()
            var maxI = b.push(Instr.Constant(Const.F32(Float.NEGATIVE_INFINITY)))
            var i = 0
            while (i < dim2) {
                val offset = b.add(globalOffI, i).toA()
                val loadI = b.push(Instr.Load(src = srcI.toVarId(), offset = offset, dtype = dtype))
                maxI = b.binary(BinaryOp.Max, maxI, loadI, dtype)
                loadIs.add(loadI)
                i += blockSize
            }
            val redMaxI = b.push(Instr.ReduceLocal(op = ReduceOp.Max, arg = maxI.toA(), dtype = dtype))

            val valueIs = ArrayList<Id>(perBlock)
            var sumI = b.push(Instr.Constant(Const.I32(0)))
            for (loadI in loadIs) {
                val valueSub = b.binary(BinaryOp.Sub, loadI, redMaxI, dtype)
                val valueExp = b.unary(UnaryOp.Exp, valueSub, dtype)
                sumI = b.binary(BinaryOp.Add, valueExp, sumI, dtype)
                valueIs.add(valueExp)
            }
            val redSumI = b.push(Instr.ReduceLocal(op = ReduceOp.Sum, arg = sumI.toA(), dtype = dtype))

            for ((idx, valueItem) in valueIs.withIndex()) {
                val off = idx * blockSize
                val offset = b.add(globalOffI, off).toA()
                val valDiv = b.binary(BinaryOp.Div, valueItem, redSumI, dtype)
                b.push(Instr.Store(dst = dstI.toVarId(), offset = offset, value = valDiv.toA(), dtype = dtype))
            }
            return Kernel.fromInstrs(b.relocate())
        }

        public fun softmax(dim1: Int, dim2: Int): Kernel = softmaxBarrier(dim1, dim2)
    }

    public object op {
        public fun softmax(dim1: Int, dim2: Int): io.github.kotlinmania.ug.lang.op.Kernel {
            val layout = Layout.fromShape(Shape(dim1, dim2))
            val srcPtr =
                io.github.kotlinmania.ug.lang.Arg
                    .ptr(DType.F32)
            val dstPtr =
                io.github.kotlinmania.ug.lang.Arg
                    .ptr(DType.F32)
            val src =
                io.github.kotlinmania.ug.lang.op
                    .load(srcPtr.id, layout, DType.F32)
            val srcMax =
                io.github.kotlinmania.ug.lang.op
                    .reduce(ReduceOp.Max, src, 1)
            val srcMaxBc =
                io.github.kotlinmania.ug.lang.op
                    .broadcast(srcMax, Shape(dim1, dim2))
            val diff =
                io.github.kotlinmania.ug.lang.op
                    .binary(BinaryOp.Sub, src, srcMaxBc)
            val exp =
                io.github.kotlinmania.ug.lang.op
                    .unary(UnaryOp.Exp, diff)
            val sumExp =
                io.github.kotlinmania.ug.lang.op
                    .reduce(ReduceOp.Sum, exp, 1)
            val sumExpBc =
                io.github.kotlinmania.ug.lang.op
                    .broadcast(sumExp, Shape(dim1, dim2))
            val sm =
                io.github.kotlinmania.ug.lang.op
                    .binary(BinaryOp.Div, exp, sumExpBc)
            val st =
                io.github.kotlinmania.ug.lang.op
                    .store(dstPtr.id, layout, sm)
            return io.github.kotlinmania.ug.lang.op.Kernel(
                name = "softmax_${dim1}_$dim2",
                args = listOf(srcPtr, dstPtr),
                ops = listOf(st),
            )
        }
    }

    public fun simpleAdd(blockSize: Int): io.github.kotlinmania.ug.lang.Kernel {
        val lhsPtr =
            io.github.kotlinmania.ug.lang.Arg
                .ptr(DType.F32)
        val rhsPtr =
            io.github.kotlinmania.ug.lang.Arg
                .ptr(DType.F32)
        val dstPtr =
            io.github.kotlinmania.ug.lang.Arg
                .ptr(DType.F32)
        val offset =
            io.github.kotlinmania.ug.lang.IndexExprNode
                .programId()
                .mul(
                    io.github.kotlinmania.ug.lang.IndexExprNode
                        .cst(blockSize),
                )
        val stride =
            io.github.kotlinmania.ug.lang.IndexExprNode
                .cst(1)
        val len =
            io.github.kotlinmania.ug.lang.IndexExprNode
                .cst(blockSize)
        val lhs =
            io.github.kotlinmania.ug.lang.ExprNode
                .load(lhsPtr, offset, len, stride)
        val rhs =
            io.github.kotlinmania.ug.lang.ExprNode
                .load(rhsPtr, offset, len, stride)
        val op =
            io.github.kotlinmania.ug.lang.Ops.Store(
                dst =
                    io.github.kotlinmania.ug.lang
                        .StridedSlice(dstPtr, offset, len, stride),
                src = lhs.add(rhs),
            )
        return io.github.kotlinmania.ug.lang.Kernel(
            name = "simple_add_$blockSize",
            args = listOf(lhsPtr, rhsPtr, dstPtr),
            ops = listOf(op),
        )
    }
}
