// port-lint: source lower_op.rs
package io.github.kotlinmania.ug.lower_op

import io.github.kotlinmania.ug.Const
import io.github.kotlinmania.ug.DType
import io.github.kotlinmania.ug.Layout
import io.github.kotlinmania.ug.Shape
import io.github.kotlinmania.ug.UgError
import io.github.kotlinmania.ug.block.Block
import io.github.kotlinmania.ug.block.Id
import io.github.kotlinmania.ug.block.Range
import io.github.kotlinmania.ug.lang.ArgId
import io.github.kotlinmania.ug.lang.BinaryOp
import io.github.kotlinmania.ug.lang.LaunchConfig
import io.github.kotlinmania.ug.lang.ReduceOp
import io.github.kotlinmania.ug.lang.Type
import io.github.kotlinmania.ug.lang.op.Ast
import io.github.kotlinmania.ug.lang.op.AstInner
import io.github.kotlinmania.ug.lang.op.Kernel
import io.github.kotlinmania.ug.lang.op.LayoutOp
import io.github.kotlinmania.ug.lang.ssa.A
import io.github.kotlinmania.ug.lang.ssa.Instr
import io.github.kotlinmania.ug.lang.ssa.KernelArg
import io.github.kotlinmania.ug.lang.ssa.Special
import io.github.kotlinmania.ug.lang.ssa.VarId

public data class ThreadBlock(
    public val dim: Int,
    public val blockDim: Int,
)

public data class Opts(
    public val threadBlock: ThreadBlock? = null,
    public val blockAxis: Int? = null,
) {
    public fun withGlobalAxis(axis: Int, blockDim: Int): Opts =
        copy(
            blockAxis = axis,
            threadBlock = ThreadBlock(dim = axis, blockDim = blockDim),
        )

    public fun withBlockAxis(axis: Int): Opts = copy(blockAxis = axis)

    public fun withThreadBlock(dim: Int, blockDim: Int): Opts =
        copy(
            threadBlock = ThreadBlock(dim = dim, blockDim = blockDim),
        )

    public companion object {
        public fun default(): Opts = Opts()
    }
}

public sealed class IndexFormula {
    public data class IdFormula(
        public val id: Id,
    ) : IndexFormula()

    public data class Constant(
        public val value: Int,
    ) : IndexFormula()

    public data class Add(
        public val lhs: IndexFormula,
        public val rhs: IndexFormula,
    ) : IndexFormula()

    public data class Mul(
        public val lhs: IndexFormula,
        public val rhs: Int,
    ) : IndexFormula()

    public data class Div(
        public val lhs: IndexFormula,
        public val rhs: Int,
    ) : IndexFormula()

    public data class Mod(
        public val lhs: IndexFormula,
        public val rhs: Int,
    ) : IndexFormula()

    public fun add(rhs: IndexFormula): IndexFormula = Add(this, rhs)

    public fun mul(rhs: Int): IndexFormula = Mul(this, rhs)

    public fun div(rhs: Int): IndexFormula = Div(this, rhs)

    public fun mod(rhs: Int): IndexFormula = Mod(this, rhs)

    public fun eval(block: Block): Id =
        when (this) {
            is IdFormula -> id
            is Constant -> block.cst(value)
            is Add -> {
                val l = lhs.eval(block)
                val r = rhs.eval(block)
                block.binary(BinaryOp.Add, l.toA(), r.toA(), DType.I32)
            }

            is Mul -> {
                val l = lhs.eval(block)
                block.mul(l, rhs)
            }

            is Div -> {
                if (rhs == 1) {
                    lhs.eval(block)
                } else {
                    val l = lhs.eval(block)
                    block.binary(BinaryOp.Div, l.toA(), A.from(rhs), DType.I32)
                }
            }

            is Mod -> {
                if (rhs == 1) {
                    block.cst(0)
                } else {
                    val l = lhs.eval(block)
                    block.binary(BinaryOp.Mod, l.toA(), A.from(rhs), DType.I32)
                }
            }
        }
}

public data class Indexes(
    public val list: MutableList<IndexFormula>,
) {
    public fun layoutOp(op: LayoutOp, shape: Shape, argShape: Shape): Indexes {
        val idxs = list.toMutableList()
        when (op) {
            is LayoutOp.Broadcast -> {
                for (dim in op.broadcastedDims) {
                    if (dim >= idxs.size) {
                        throw UgError.Msg("unexpected dim for broadcast, $dim $shape")
                    }
                    idxs[dim] = IndexFormula.Constant(0)
                }
                repeat(op.insertedDims) {
                    if (idxs.isNotEmpty()) {
                        idxs.removeAt(0)
                    }
                }
            }

            is LayoutOp.Narrow -> {
                if (op.dim >= idxs.size) {
                    throw UgError.Msg("unexpected dim for narrow, ${op.dim} $shape")
                }
                idxs[op.dim] = idxs[op.dim].add(IndexFormula.Constant(op.offset))
            }

            is LayoutOp.Transpose -> {
                if (op.dim1 >= idxs.size || op.dim2 >= idxs.size) {
                    throw UgError.Msg("unexpected dims for transpose ${op.dim1} ${op.dim2}, $shape")
                }
                val tmp = idxs[op.dim1]
                idxs[op.dim1] = idxs[op.dim2]
                idxs[op.dim2] = tmp
            }

            is LayoutOp.SplitDim -> {
                if (op.dim >= argShape.rank()) {
                    throw UgError.Msg("unexpected split dim ${op.dim} src $shape")
                }
                if (op.lhs >= shape.rank() || op.rhs >= shape.rank() || op.lhs == op.rhs) {
                    throw UgError.Msg("unexpected split dims ${op.lhs}x${op.rhs} dst $shape")
                }
                val dims = shape.dims
                val (l, r) =
                    if (op.lhs < op.rhs) {
                        val r = idxs.removeAt(op.rhs)
                        val l = idxs.removeAt(op.lhs)
                        Pair(l, r)
                    } else {
                        val l = idxs.removeAt(op.lhs)
                        val r = idxs.removeAt(op.rhs)
                        Pair(l, r)
                    }
                idxs.add(op.dim, l.mul(dims[op.rhs]).add(r))
            }

            is LayoutOp.MergeDims -> {
                if (op.dim >= shape.rank()) {
                    throw UgError.Msg("unexpected merge dim ${op.dim} dst $shape")
                }
                val idx = idxs.removeAt(op.dim)
                val argDims = argShape.dims
                val lhsIdx = if (argDims[op.lhs] <= 1) IndexFormula.Constant(0) else idx.div(argDims[op.rhs])
                val rhsIdx = if (argDims[op.rhs] <= 1) IndexFormula.Constant(0) else idx.mod(argDims[op.rhs])
                if (op.lhs < op.rhs) {
                    idxs.add(op.lhs, lhsIdx)
                    idxs.add(op.rhs, rhsIdx)
                } else {
                    idxs.add(op.rhs, rhsIdx)
                    idxs.add(op.lhs, lhsIdx)
                }
            }
        }
        return Indexes(idxs)
    }
}

public fun Layout.lower(idxs: Indexes): Pair<Id, Block> {
    if (idxs.list.size != strides.size) {
        throw UgError.Msg("len mismatch between strides $this and idxs $idxs")
    }
    val block = Block.empty()
    var accId: Id? = if (offset == 0) null else block.cst(offset)
    for (i in idxs.list.indices) {
        val idx = idxs.list[i]
        val stride = strides[i]
        val dimId = idx.eval(block)
        val mulId = block.mul(dimId, stride)
        accId =
            if (accId != null) {
                block.binary(BinaryOp.Add, mulId.toA(), accId.toA(), DType.I32)
            } else {
                mulId
            }
    }
    return if (accId != null) {
        Pair(accId, block)
    } else {
        val id = block.cst(0)
        Pair(id, block)
    }
}

public fun Ast.lower(
    idxs: Indexes,
    opts: Opts,
    perArg: Map<ArgId, VarId>,
): Pair<Id, Block> {
    val block = Block.empty()
    val dstId =
        when (val in_ = inner) {
            is AstInner.Load -> {
                val ptrI = perArg[in_.src] ?: throw UgError.Msg("unknown arg ${in_.src}")
                val (offI, offB) = in_.layout.lower(idxs)
                block.instrs.addAll(offB.instrs)
                block.push(Instr.Load(src = ptrI, offset = offI.toA(), dtype = dtype))
            }

            is AstInner.Constant -> block.cst(in_.value)
            is AstInner.Id -> throw UgError.Msg("unexpected id node in Ast")
            is AstInner.Unary -> {
                val (argId, argB) = in_.arg.lower(idxs, opts, perArg)
                block.instrs.addAll(argB.instrs)
                block.push(Instr.Unary(op = in_.op, arg = argId.toA(), dtype = dtype))
            }

            is AstInner.Binary -> {
                val (lhsId, lhsB) = in_.lhs.lower(idxs, opts, perArg)
                block.instrs.addAll(lhsB.instrs)
                val (rhsId, rhsB) = in_.rhs.lower(idxs, opts, perArg)
                block.instrs.addAll(rhsB.instrs)
                block.push(Instr.Binary(op = in_.op, lhs = lhsId.toA(), rhs = rhsId.toA(), dtype = dtype))
            }

            is AstInner.LayoutOperation -> {
                val nextIdxs = idxs.layoutOp(in_.op, shape, in_.arg.shape)
                val (argId, argB) = in_.arg.lower(nextIdxs, opts, perArg)
                block.instrs.addAll(argB.instrs)
                argId
            }

            is AstInner.Reduce -> {
                val dimLen = in_.arg.shape.dims[in_.dim]
                val initVal =
                    when (in_.op) {
                        ReduceOp.Sum -> Const.zero(dtype)
                        ReduceOp.Min -> Const.maxValue(dtype)
                        ReduceOp.Max -> Const.minValue(dtype)
                    }
                val foldOp =
                    when (in_.op) {
                        ReduceOp.Sum -> BinaryOp.Add
                        ReduceOp.Max -> BinaryOp.Max
                        ReduceOp.Min -> BinaryOp.Min
                    }
                val accId = block.push(Instr.DefineAcc(initVal))
                val r = block.range(A.from(0), A.from(dimLen), 1)
                val rIdxs = idxs.list.toMutableList()
                rIdxs[in_.dim] = IndexFormula.IdFormula(r.rangeId)
                val (argId, argB) = in_.arg.lower(Indexes(rIdxs), opts, perArg)
                block.instrs.addAll(argB.instrs)
                val nextAcc = block.binary(foldOp, accId.toA(), argId.toA(), dtype)
                block.push(Instr.Assign(dst = accId.toVarId(), src = nextAcc.toA()))
                block.endRange(r)
                accId
            }
        }
    return Pair(dstId, block)
}

public fun lowerKernel(
    kernel: Kernel,
    opts: Opts = Opts(),
): io.github.kotlinmania.ug.lang.ssa.Kernel {
    val block = Block.empty()
    val perArg = mutableMapOf<ArgId, VarId>()

    for ((index, arg) in kernel.args.withIndex()) {
        val dtype =
            when (val t = arg.type) {
                is Type.Ptr -> t.dtype
                is Type.Value -> throw UgError.Msg("non-pointer arguments are not supported yet")
            }
        val id = block.push(Instr.DefineGlobal(index = index, dtype = dtype))
        perArg[arg.id] = id.toVarId()
    }

    for (op in kernel.ops) {
        val ptrI = perArg[op.dst] ?: throw UgError.Msg("unknown arg ${op.dst}")
        val idxList = mutableListOf<IndexFormula>()
        val ranges = mutableListOf<Range>()

        val blockIdPair =
            opts.blockAxis?.let { idx ->
                val id = block.push(Instr.SpecialInstr(Special.BlockIdx))
                Pair(idx, id)
            }

        val threadIdPair =
            opts.threadBlock?.let { tb ->
                val id = block.push(Instr.SpecialInstr(Special.ThreadIdx))
                Pair(tb, id)
            }

        for ((dimIdx, len) in op.layout.dims.withIndex()) {
            val id: Id =
                if (blockIdPair != null && blockIdPair.first == dimIdx && threadIdPair != null && threadIdPair.first.dim == dimIdx) {
                    val mulId = block.mul(blockIdPair.second, threadIdPair.first.blockDim)
                    val baseId = block.binary(BinaryOp.Add, mulId.toA(), threadIdPair.second.toA(), DType.I32)
                    if (len == threadIdPair.first.blockDim) {
                        baseId
                    } else {
                        val r = block.range(baseId.toA(), A.from(len), threadIdPair.first.blockDim)
                        ranges.add(r)
                        r.rangeId
                    }
                } else if (blockIdPair != null && blockIdPair.first == dimIdx) {
                    if (len == 1) {
                        blockIdPair.second
                    } else {
                        blockIdPair.second
                    }
                } else if (threadIdPair != null && threadIdPair.first.dim == dimIdx) {
                    if (len == threadIdPair.first.blockDim) {
                        threadIdPair.second
                    } else {
                        val r = block.range(threadIdPair.second.toA(), A.from(len), threadIdPair.first.blockDim)
                        ranges.add(r)
                        r.rangeId
                    }
                } else {
                    val r = block.range(A.from(0), A.from(len), 1)
                    ranges.add(r)
                    r.rangeId
                }
            idxList.add(IndexFormula.IdFormula(id))
        }

        val idxs = Indexes(idxList)
        val (offI, offB) = op.layout.lower(idxs)
        block.instrs.addAll(offB.instrs)

        val (srcI, srcB) = op.value.lower(idxs, opts, perArg)
        block.instrs.addAll(srcB.instrs)

        val store =
            Instr.Store(
                dst = ptrI,
                offset = offI.toA(),
                value = srcI.toA(),
                dtype = op.value.dtype,
            )
        block.push(store)

        for (r in ranges.reversed()) {
            block.endRange(r)
        }
    }

    val instrs = block.relocate()
    val ssaArgs = kernel.args.mapIndexed { i, a -> KernelArg(a, i) }
    val gridDim =
        opts.blockAxis?.let { idx ->
            val local = if (opts.threadBlock?.dim == idx) opts.threadBlock.blockDim else 1
            val dimLen = kernel.ops.firstOrNull()?.layout?.dims?.getOrNull(idx) ?: 1
            (dimLen + local - 1) / local
        } ?: 1
    val blockDim = opts.threadBlock?.blockDim ?: 1
    val cfg = LaunchConfig(gridDim = gridDim, blockDim = blockDim, sharedMem = 0)
    return io.github.kotlinmania.ug.lang.ssa.Kernel(instrs, ssaArgs, cfg)
}

public fun Kernel.lower(opts: Opts = Opts.default()): io.github.kotlinmania.ug.lang.ssa.Kernel =
    lowerKernel(this, opts)
