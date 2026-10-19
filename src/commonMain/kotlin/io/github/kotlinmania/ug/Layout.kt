// port-lint: source layout.rs
package io.github.kotlinmania.ug

public class Shape(
    dims: List<Int>,
) : Comparable<Shape> {
    public val dims: List<Int> = dims.toList()

    public constructor(vararg dims: Int) : this(dims.toList())

    public fun numElements(): Int = if (dims.isEmpty()) 1 else dims.fold(1) { acc, d -> acc * d }

    public fun elemCount(): Int = numElements()

    public fun rank(): Int = dims.size

    public fun intoDims(): List<Int> = dims

    public fun strideContiguous(): List<Int> {
        if (dims.isEmpty()) return emptyList()
        val result = ArrayList<Int>(dims.size)
        var acc = 1
        for (i in dims.indices.reversed()) {
            result.add(acc)
            acc *= dims[i]
        }
        result.reverse()
        return result
    }

    public fun isContiguous(stride: List<Int>): Boolean {
        if (dims.size != stride.size) return false
        var acc = 1
        for (i in dims.indices.reversed()) {
            val dim = dims[i]
            val s = stride[i]
            if (dim > 1 && s != acc) return false
            acc *= dim
        }
        return true
    }

    public fun isFortranContiguous(stride: List<Int>): Boolean {
        if (dims.size != stride.size) return false
        var acc = 1
        for (i in dims.indices) {
            val dim = dims[i]
            val s = stride[i]
            if (dim > 1 && s != acc) return false
            acc *= dim
        }
        return true
    }

    public fun extend(additionalDims: List<Int>): Shape = Shape(dims + additionalDims)

    public fun broadcastShapeBinaryOp(rhs: Shape, op: String): Shape {
        val lhsDims = dims
        val rhsDims = rhs.dims
        val bcastNdims = maxOf(lhsDims.size, rhsDims.size)
        val bcastDims = IntArray(bcastNdims)
        for (idx in 0 until bcastNdims) {
            val revIdx = bcastNdims - idx
            val lVal = if (lhsDims.size < revIdx) 1 else lhsDims[lhsDims.size - revIdx]
            val rVal = if (rhsDims.size < revIdx) 1 else rhsDims[rhsDims.size - revIdx]
            bcastDims[idx] =
                if (lVal == rVal) {
                    lVal
                } else if (lVal == 1) {
                    rVal
                } else if (rVal == 1) {
                    lVal
                } else {
                    throw UgError.Msg("shape mismatch in binary op '$op', lhs: $this rhs: $rhs")
                }
        }
        return Shape(bcastDims.toList())
    }

    public fun dims0(): Unit {
        if (dims.isNotEmpty()) throw UgError.Msg("unexpected number of dims, expected 0 got ${dims.size} shape $this")
    }

    public fun dims1(): Int {
        if (dims.size != 1) throw UgError.Msg("unexpected number of dims, expected 1 got ${dims.size} shape $this")
        return dims[0]
    }

    public fun dims2(): Pair<Int, Int> {
        if (dims.size != 2) throw UgError.Msg("unexpected number of dims, expected 2 got ${dims.size} shape $this")
        return Pair(dims[0], dims[1])
    }

    public fun dims3(): Triple<Int, Int, Int> {
        if (dims.size != 3) throw UgError.Msg("unexpected number of dims, expected 3 got ${dims.size} shape $this")
        return Triple(dims[0], dims[1], dims[2])
    }

    public fun dims4(): List<Int> {
        if (dims.size != 4) throw UgError.Msg("unexpected number of dims, expected 4 got ${dims.size} shape $this")
        return dims
    }

    public fun dims5(): List<Int> {
        if (dims.size != 5) throw UgError.Msg("unexpected number of dims, expected 5 got ${dims.size} shape $this")
        return dims
    }

    override fun compareTo(other: Shape): Int {
        val rankComp = rank().compareTo(other.rank())
        if (rankComp != 0) return rankComp
        for (i in dims.indices) {
            val c = dims[i].compareTo(other.dims[i])
            if (c != 0) return c
        }
        return 0
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Shape) return false
        return dims == other.dims
    }

    override fun hashCode(): Int = dims.hashCode()

    override fun toString(): String = dims.toString()

    public companion object {
        public val SCALAR: Shape = Shape(emptyList())

        public fun from(vararg dims: Int): Shape = Shape(dims.toList())

        public fun from(dims: List<Int>): Shape = Shape(dims)

        public fun from(dim: Int): Shape = Shape(listOf(dim))
    }
}

public interface Dim {
    public fun toIndex(shape: Shape, op: String): Int

    public fun toIndexPlusOne(shape: Shape, op: String): Int
}

public sealed class D : Dim {
    public data object Minus1 : D() {
        override fun toIndex(shape: Shape, op: String): Int {
            val rank = shape.rank()
            if (rank >= 1) return rank - 1
            throw UgError.Msg("dim out of range in '$op', dim: -1, shape: $shape")
        }

        override fun toIndexPlusOne(shape: Shape, op: String): Int = shape.rank()
    }

    public data object Minus2 : D() {
        override fun toIndex(shape: Shape, op: String): Int {
            val rank = shape.rank()
            if (rank >= 2) return rank - 2
            throw UgError.Msg("dim out of range in '$op', dim: -2, shape: $shape")
        }

        override fun toIndexPlusOne(shape: Shape, op: String): Int {
            val rank = shape.rank()
            if (rank >= 1) return rank - 1
            throw UgError.Msg("dim out of range in '$op', dim: -2, shape: $shape")
        }
    }

    public data class Minus(
        public val u: Int,
    ) : D() {
        override fun toIndex(shape: Shape, op: String): Int {
            val rank = shape.rank()
            if (u > 0 && rank >= u) return rank - u
            throw UgError.Msg("dim out of range in '$op', dim: -$u, shape: $shape")
        }

        override fun toIndexPlusOne(shape: Shape, op: String): Int {
            val rank = shape.rank()
            if (u > 0 && rank + 1 >= u) return rank + 1 - u
            throw UgError.Msg("dim out of range in '$op', dim: -$u, shape: $shape")
        }
    }
}

public fun Int.toDimIndex(shape: Shape, op: String): Int {
    if (this >= 0) {
        if (this >= shape.rank()) throw UgError.Msg("dim out of range in '$op', dim: $this, shape: $shape")
        return this
    }
    return D.Minus(-this).toIndex(shape, op)
}

public fun Int.toDimIndexPlusOne(shape: Shape, op: String): Int {
    if (this >= 0) {
        if (this > shape.rank()) throw UgError.Msg("dim out of range in '$op', dim: $this, shape: $shape")
        return this
    }
    return D.Minus(-this).toIndexPlusOne(shape, op)
}

public class Layout(
    public val shape: Shape,
    strides: List<Int>? = null,
    public var offset: Int = 0,
) : Comparable<Layout> {
    public val strides: List<Int> = strides ?: shape.strideContiguous()

    public fun canBeCompressed(): Boolean {
        val dims = shape.dims
        if (dims.size <= 1) return false
        for (i in 0 until dims.size - 1) {
            if (this.strides[i] != this.strides[i + 1] * dims[i + 1]) {
                return false
            }
        }
        return true
    }

    public fun compressAll(): Layout {
        val dims = shape.dims
        for (i in 0 until dims.size - 1) {
            if (strides[i] != strides[i + 1] * dims[i + 1]) {
                throw UgError.Msg("cannot collapse dims, $this")
            }
        }
        val stride = strides.lastOrNull() ?: 1
        val dim = numElements()
        return Layout(Shape(listOf(dim)), listOf(stride), offset)
    }

    public fun transpose(): Layout {
        val r = rank()
        if (r < 2) return Layout(shape, strides, offset)
        val dims = shape.dims.toMutableList()
        val str = strides.toMutableList()
        val tempD = dims[r - 2]
        dims[r - 2] = dims[r - 1]
        dims[r - 1] = tempD
        val tempS = str[r - 2]
        str[r - 2] = str[r - 1]
        str[r - 1] = tempS
        return Layout(Shape(dims), str, offset)
    }

    public fun numElements(): Int = shape.numElements()

    public val dims: List<Int> get() = shape.dims

    public fun rank(): Int = shape.rank()

    public fun cContiguous(): Boolean {
        var prodL = 1
        val dims = shape.dims
        for (i in dims.indices.reversed()) {
            if (strides[i] != prodL) {
                return false
            }
            prodL *= dims[i]
        }
        return true
    }

    override fun compareTo(other: Layout): Int {
        val sc = shape.compareTo(other.shape)
        if (sc != 0) return sc
        val oc = offset.compareTo(other.offset)
        if (oc != 0) return oc
        val rankComp = strides.size.compareTo(other.strides.size)
        if (rankComp != 0) return rankComp
        for (i in strides.indices) {
            val c = strides[i].compareTo(other.strides[i])
            if (c != 0) return c
        }
        return 0
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Layout) return false
        return shape == other.shape && strides == other.strides && offset == other.offset
    }

    override fun hashCode(): Int {
        var result = shape.hashCode()
        result = 31 * result + strides.hashCode()
        result = 31 * result + offset
        return result
    }

    override fun toString(): String = "Layout(shape=$shape, strides=$strides, offset=$offset)"

    public companion object {
        public fun fromShape(shape: Shape): Layout = Layout(shape, shape.strideContiguous(), 0)

        public fun fromShape(vararg dims: Int): Layout = fromShape(Shape(*dims))

        public fun fromShape(dims: List<Int>): Layout = fromShape(Shape(dims))

        public fun fromShape(dim: Int): Layout = fromShape(Shape(dim))
    }
}
