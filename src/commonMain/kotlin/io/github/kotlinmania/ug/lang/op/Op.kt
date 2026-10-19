// port-lint: source lang.rs
package io.github.kotlinmania.ug.lang.op

import io.github.kotlinmania.ug.Const
import io.github.kotlinmania.ug.DType
import io.github.kotlinmania.ug.Layout
import io.github.kotlinmania.ug.Shape
import io.github.kotlinmania.ug.UgError
import io.github.kotlinmania.ug.lang.Arg
import io.github.kotlinmania.ug.lang.ArgId
import io.github.kotlinmania.ug.lang.BinaryOp
import io.github.kotlinmania.ug.lang.ReduceOp
import io.github.kotlinmania.ug.lang.UnaryOp

public sealed class LayoutOp : Comparable<LayoutOp> {
    public data class Broadcast(
        public val insertedDims: Int,
        public val broadcastedDims: List<Int>,
    ) : LayoutOp()

    public data class Narrow(
        public val dim: Int,
        public val offset: Int,
    ) : LayoutOp()

    public data class Transpose(
        public val dim1: Int,
        public val dim2: Int,
    ) : LayoutOp()

    public data class SplitDim(
        public val dim: Int,
        public val lhs: Int,
        public val rhs: Int,
    ) : LayoutOp()

    public data class MergeDims(
        public val dim: Int,
        public val lhs: Int,
        public val rhs: Int,
    ) : LayoutOp()

    override fun compareTo(other: LayoutOp): Int = this.toString().compareTo(other.toString())
}

public sealed class AstInner {
    public data class Id(
        public val id: Int,
    ) : AstInner()

    public data class Load(
        public val src: ArgId,
        public val layout: Layout,
    ) : AstInner()

    public data class Reduce(
        public val op: ReduceOp,
        public val arg: Ast,
        public val dim: Int,
    ) : AstInner()

    public data class Unary(
        public val op: UnaryOp,
        public val arg: Ast,
    ) : AstInner()

    public data class Binary(
        public val op: BinaryOp,
        public val lhs: Ast,
        public val rhs: Ast,
    ) : AstInner()

    public data class Constant(
        public val value: Const,
    ) : AstInner()

    public data class LayoutOperation(
        public val op: LayoutOp,
        public val arg: Ast,
    ) : AstInner()
}

public data class Ast(
    public val inner: AstInner,
    public val dtype: DType,
    public val shape: Shape,
) : Comparable<Ast> {

    public fun argIds(): Set<ArgId> {
        val ids = mutableSetOf<ArgId>()

        fun visit(ast: Ast) {
            when (val in_ = ast.inner) {
                is AstInner.LayoutOperation -> visit(in_.arg)
                is AstInner.Reduce -> visit(in_.arg)
                is AstInner.Unary -> visit(in_.arg)
                is AstInner.Binary -> {
                    visit(in_.lhs)
                    visit(in_.rhs)
                }

                is AstInner.Load -> ids.add(in_.src)
                is AstInner.Id, is AstInner.Constant -> {}
            }
        }
        visit(this)
        return ids
    }

    override fun compareTo(other: Ast): Int {
        val dc = dtype.compareTo(other.dtype)
        if (dc != 0) return dc
        return shape.compareTo(other.shape)
    }
}

public data class Store(
    public val dst: ArgId,
    public var layout: Layout,
    public var value: Ast,
) {
    public val dtype: DType get() = value.dtype

    public fun name(): String {
        val chars = StringBuilder()
        fun walk(ast: Ast) {
            when (ast.inner) {
                is AstInner.Unary -> {
                    chars.append('U')
                    walk(ast.inner.arg)
                }

                is AstInner.Binary -> {
                    chars.append('B')
                    walk(ast.inner.lhs)
                    walk(ast.inner.rhs)
                }

                is AstInner.Load -> chars.append('L')
                is AstInner.Reduce -> {
                    chars.append('R')
                    walk(ast.inner.arg)
                }

                is AstInner.LayoutOperation -> {
                    chars.append('T')
                    walk(ast.inner.arg)
                }

                is AstInner.Constant -> chars.append('C')
                is AstInner.Id -> chars.append('I')
            }
        }
        walk(value)
        return chars.take(40).toString()
    }

    public fun canMergeAll(): Boolean {
        fun walk(ast: Ast): Boolean =
            when (val in_ = ast.inner) {
                is AstInner.Unary -> walk(in_.arg)
                is AstInner.Binary -> walk(in_.lhs) && walk(in_.rhs)
                is AstInner.Load -> in_.layout.canBeCompressed()
                is AstInner.Reduce -> false
                is AstInner.LayoutOperation ->
                    when (val op = in_.op) {
                        is LayoutOp.Narrow, is LayoutOp.SplitDim, is LayoutOp.MergeDims, is LayoutOp.Transpose -> false
                        is LayoutOp.Broadcast -> op.broadcastedDims.size >= ast.shape.rank()
                    }

                is AstInner.Constant, is AstInner.Id -> true
            }

        if (!layout.canBeCompressed()) return false
        return walk(value)
    }
}

public data class Kernel(
    public val name: String,
    public val args: List<Arg>,
    public val ops: List<Store>,
) {
    public fun optimize(): Kernel {
        fun walk(v: Ast): Ast =
            when (val in_ = v.inner) {
                is AstInner.Unary -> {
                    val arg = walk(in_.arg)
                    unary(in_.op, arg)
                }

                is AstInner.Binary -> {
                    val lhs = walk(in_.lhs)
                    val rhs = walk(in_.rhs)
                    binary(in_.op, lhs, rhs)
                }

                is AstInner.Load -> load(in_.src, in_.layout.compressAll(), v.dtype)
                is AstInner.Reduce -> throw UgError.Msg("unexpected result in optimize step")
                is AstInner.LayoutOperation ->
                    when (in_.op) {
                        is LayoutOp.Broadcast -> {
                            val arg = walk(in_.arg)
                            broadcast(arg, Shape.from(v.shape.numElements()))
                        }

                        else -> throw UgError.Msg("unexpected layout op ${in_.op}")
                    }

                is AstInner.Constant, is AstInner.Id -> v
            }

        val newOps =
            ops.map { op ->
                if (op.canMergeAll()) {
                    Store(op.dst, op.layout.compressAll(), walk(op.value))
                } else {
                    op
                }
            }
        return Kernel(name, args, newOps)
    }

    public fun lower(opts: io.github.kotlinmania.ug.lowerop.Opts = io.github.kotlinmania.ug.lowerop.Opts()): io.github.kotlinmania.ug.lang.ssa.Kernel =
        io.github.kotlinmania.ug.lowerop.lowerKernel(this, opts)

    public companion object {
        public fun new(name: String, args: List<Arg>, ops: List<Store>): Kernel = Kernel(name, args, ops)
    }
}

public fun load(src: ArgId, layout: Layout, dtype: DType): Ast {
    val shape = layout.shape
    val inner = AstInner.Load(src, layout)
    return Ast(inner, dtype, shape)
}

public fun broadcast(arg: Ast, shape: Shape): Ast {
    val dtype = arg.dtype
    if (arg.shape.rank() > shape.rank()) {
        throw UgError.Msg("target shape $shape has less dimensions than original shape ${arg.shape}")
    }
    val insertedDims = shape.rank() - arg.shape.rank()
    val broadcastedDims = (0 until insertedDims).toMutableList()
    for ((dimIdx, dimLen) in arg.shape.dims.withIndex()) {
        val actualIdx = dimIdx + insertedDims
        if (shape.dims[actualIdx] != dimLen) {
            if (dimLen == 1) {
                broadcastedDims.add(actualIdx)
            } else {
                throw UgError.Msg("cannot broadcast from ${arg.shape} to $shape")
            }
        }
    }
    val op = LayoutOp.Broadcast(insertedDims, broadcastedDims)
    val inner = AstInner.LayoutOperation(op, arg)
    return Ast(inner, dtype, shape)
}

public fun broadcast(arg: Ast, shapeDims: Pair<Int, Int>): Ast = broadcast(arg, Shape(shapeDims.first, shapeDims.second))

public fun layout(op: LayoutOp, arg: Ast, shape: Shape): Ast {
    val dtype = arg.dtype
    val inner = AstInner.LayoutOperation(op, arg)
    return Ast(inner, dtype, shape)
}

public fun unary(op: UnaryOp, arg: Ast): Ast {
    val dtype =
        when (op) {
            is UnaryOp.Cast -> op.dtype
            else -> arg.dtype
        }
    val shape = arg.shape
    val inner = AstInner.Unary(op, arg)
    return Ast(inner, dtype, shape)
}

public fun reduce(op: ReduceOp, arg: Ast, dim: Int): Ast {
    val dtype = arg.dtype
    val shapeDims = arg.shape.dims.toMutableList()
    if (dim >= shapeDims.size) {
        throw UgError.Msg("no dim $dim in shape ${arg.shape}")
    }
    shapeDims[dim] = 1
    val inner = AstInner.Reduce(op, arg, dim)
    return Ast(inner, dtype, Shape(shapeDims))
}

public fun binary(op: BinaryOp, lhs: Ast, rhs: Ast): Ast {
    val dtype =
        if (lhs.dtype != rhs.dtype) {
            throw UgError.Msg("dtype mismatch in $op, lhs: ${lhs.dtype}, rhs: ${rhs.dtype}")
        } else {
            lhs.dtype
        }
    if (lhs.shape != rhs.shape && lhs.shape.rank() != 0 && rhs.shape.rank() != 0) {
        throw UgError.Msg("shape mismatch in $op, lhs: ${lhs.shape}, rhs: ${rhs.shape}")
    }
    val shape = lhs.shape
    val inner = AstInner.Binary(op, lhs, rhs)
    return Ast(inner, dtype, shape)
}

public fun broadcastBinary(op: BinaryOp, lhs: Ast, rhs: Ast): Ast {
    val shape = lhs.shape.broadcastShapeBinaryOp(rhs.shape, op.asStr)
    val l = broadcast(lhs, shape)
    val r = broadcast(rhs, shape)
    return binary(op, l, r)
}

public fun cst(v: Const): Ast {
    val dtype = v.dtype
    val inner = AstInner.Constant(v)
    return Ast(inner, dtype, Shape.SCALAR)
}

public fun cst(v: Int): Ast = cst(Const.I32(v))

public fun cst(v: Long): Ast = cst(Const.I64(v))

public fun cst(v: Float): Ast = cst(Const.F32(v))

public fun store(dst: ArgId, layout: Layout, value: Ast): Store = Store(dst, layout, value)
