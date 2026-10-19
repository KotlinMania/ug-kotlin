// port-lint: source lang.rs
package io.github.kotlinmania.ug.lang

import io.github.kotlinmania.ug.Const
import io.github.kotlinmania.ug.DType
import io.github.kotlinmania.ug.UgError

public enum class ReduceOp {
    Sum,
    Max,
    Min,
}

public sealed class UnaryOp : Comparable<UnaryOp> {
    public data object Id : UnaryOp()

    public data object Exp : UnaryOp()

    public data object Sin : UnaryOp()

    public data object Cos : UnaryOp()

    public data object Neg : UnaryOp()

    public data object Sqrt : UnaryOp()

    public data class Cast(
        public val dtype: DType,
    ) : UnaryOp()

    override fun compareTo(other: UnaryOp): Int = this.toString().compareTo(other.toString())
}

public enum class BinaryOp(
    public val asStr: String,
) {
    Add("add"),
    Sub("sub"),
    Mul("mul"),
    Div("div"),
    Mod("mod"),
    Min("min"),
    Max("max"),
}

public data class LaunchConfig(
    public val gridDim: Int,
    public val blockDim: Int,
    public val sharedMem: Int = 0,
) {
    public companion object {
        public fun forNumElems(n: Int): LaunchConfig {
            val gridDim = (n + 31) / 32
            return LaunchConfig(gridDim = gridDim, blockDim = 32, sharedMem = 0)
        }
    }
}

public sealed class UgType : Comparable<UgType> {
    public data class Value(
        public val dtype: DType,
    ) : UgType()

    public data class Ptr(
        public val dtype: DType,
    ) : UgType()

    public fun sizeInBytes(): Int =
        when (this) {
            is Ptr -> 8
            is Value -> dtype.sizeInBytes
        }

    override fun compareTo(other: UgType): Int = this.toString().compareTo(other.toString())
}

public data class ArgId(
    public val id: Int,
) : Comparable<ArgId> {
    override fun compareTo(other: ArgId): Int = id.compareTo(other.id)

    public companion object {
        private var counter: Int = 1

        public fun new(): ArgId {
            val v = counter
            counter += 1
            return ArgId(v)
        }

        public fun fromInt(v: Int): ArgId = ArgId(v)

        public fun fromUsize(v: Int): ArgId = ArgId(v)
    }
}

public data class Arg(
    public val id: ArgId,
    public val type: UgType,
) : Comparable<Arg> {
    public fun type_(): UgType = type

    override fun compareTo(other: Arg): Int = id.compareTo(other.id)

    public companion object {
        public fun new(id: ArgId, type: UgType): Arg = Arg(id, type)

        public fun value(dtype: DType): Arg = Arg(ArgId.new(), UgType.Value(dtype))

        public fun ptr(dtype: DType): Arg = Arg(ArgId.new(), UgType.Ptr(dtype))
    }
}

public sealed class IndexExpr {
    public data object ProgramId : IndexExpr()

    public data class Constant(
        public val value: Int,
    ) : IndexExpr()

    public data class Add(
        public val lhs: IndexExprNode,
        public val rhs: IndexExprNode,
    ) : IndexExpr()

    public data class Mul(
        public val lhs: IndexExprNode,
        public val rhs: IndexExprNode,
    ) : IndexExpr()
}

public data class IndexExprNode(
    public val expr: IndexExpr,
) {
    public fun add(rhs: IndexExprNode): IndexExprNode = IndexExprNode(IndexExpr.Add(this, rhs))

    public fun mul(rhs: IndexExprNode): IndexExprNode = IndexExprNode(IndexExpr.Mul(this, rhs))

    public fun asConst(): Int? =
        when (val e = expr) {
            is IndexExpr.Constant -> {
                e.value
            }

            is IndexExpr.ProgramId -> {
                null
            }

            is IndexExpr.Add -> {
                val l = e.lhs.asConst()
                val r = e.rhs.asConst()
                if (l != null && r != null) l + r else null
            }

            is IndexExpr.Mul -> {
                val l = e.lhs.asConst()
                val r = e.rhs.asConst()
                if (l != null && r != null) l * r else null
            }
        }

    public companion object {
        public fun cst(v: Int): IndexExprNode = IndexExprNode(IndexExpr.Constant(v))

        public fun programId(): IndexExprNode = IndexExprNode(IndexExpr.ProgramId)

        public fun mul(lhs: IndexExprNode, rhs: IndexExprNode): IndexExprNode = lhs.mul(rhs)
    }
}

public data class StridedSlice(
    public val ptr: Arg,
    public val offset: IndexExprNode,
    public val len: IndexExprNode,
    public val stride: IndexExprNode,
)

public sealed class Expr {
    public data class Constant(
        public val c: Const,
    ) : Expr()

    public data class Range(
        public val lo: Int,
        public val up: Int,
    ) : Expr()

    public data class Load(
        public val slice: StridedSlice,
    ) : Expr()

    public data class Unary(
        public val op: UnaryOp,
        public val arg: ExprNode,
    ) : Expr()

    public data class Binary(
        public val op: BinaryOp,
        public val lhs: ExprNode,
        public val rhs: ExprNode,
    ) : Expr()
}

public data class ExprNode(
    public val expr: Expr,
    public val dtype: DType,
) {
    public fun unary(op: UnaryOp): ExprNode = ExprNode(Expr.Unary(op, this), dtype)

    public fun binary(rhs: ExprNode, op: BinaryOp): ExprNode = ExprNode(Expr.Binary(op, this, rhs), dtype)

    public fun add(rhs: ExprNode): ExprNode = ExprNode(Expr.Binary(BinaryOp.Add, this, rhs), dtype)

    public fun mul(rhs: ExprNode): ExprNode = ExprNode(Expr.Binary(BinaryOp.Mul, this, rhs), dtype)

    public fun allArgs(args: MutableSet<Arg>) {
        when (val e = expr) {
            is Expr.Load -> {
                args.add(e.slice.ptr)
            }

            is Expr.Binary -> {
                e.lhs.allArgs(args)
                e.rhs.allArgs(args)
            }

            is Expr.Unary -> {
                e.arg.allArgs(args)
            }

            is Expr.Range, is Expr.Constant -> {}
        }
    }

    public companion object {
        public fun cst(c: Const): ExprNode = ExprNode(Expr.Constant(c), c.dtype)

        public fun load(
            ptr: Arg,
            offset: IndexExprNode,
            len: IndexExprNode,
            stride: IndexExprNode,
        ): ExprNode {
            val dtype =
                when (val t = ptr.type) {
                    is UgType.Ptr -> t.dtype
                    is UgType.Value -> UgError.Msg("non-pointer arguments are not supported yet").raise()
                }
            val ss = StridedSlice(ptr, offset, len, stride)
            return ExprNode(Expr.Load(ss), dtype)
        }
    }
}

public sealed class Ops {
    public data class Store(
        public val dst: StridedSlice,
        public val src: ExprNode,
    ) : Ops()

    public fun src(): ExprNode =
        when (this) {
            is Store -> src
        }

    public fun dst(): StridedSlice =
        when (this) {
            is Store -> dst
        }

    public fun allArgs(args: MutableSet<Arg>) {
        when (this) {
            is Store -> {
                args.add(dst.ptr)
                src.allArgs(args)
            }
        }
    }

    public companion object {
        public fun store(
            dstPtr: Arg,
            dstOffset: IndexExprNode,
            dstLen: IndexExprNode,
            dstStride: IndexExprNode,
            src: ExprNode,
        ): Ops = Store(StridedSlice(dstPtr, dstOffset, dstLen, dstStride), src)
    }
}

public data class Kernel(
    public val name: String,
    public val args: List<Arg>,
    public val ops: List<Ops>,
) {
    public companion object {
        public fun new(name: String, args: List<Arg>, ops: List<Ops>): Kernel = Kernel(name, args, ops)
    }
}

public data class FlopsMem(
    public val flops: Int,
    public val memInBytes: Int,
)
