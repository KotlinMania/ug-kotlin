// port-lint: source lazy_buffer.rs
package io.github.kotlinmania.ug

import io.github.kotlinmania.half.BF16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.ug.cpu_runtime.CpuStorage
import io.github.kotlinmania.ug.lang.BinaryOp
import io.github.kotlinmania.ug.lang.ReduceOp
import io.github.kotlinmania.ug.lang.UnaryOp
import io.github.kotlinmania.ug.lang.op.LayoutOp

public typealias CustomFCallback = (List<Slice>) -> Unit

public class CustomF(
    public val callback: CustomFCallback,
) {
    public operator fun invoke(slices: List<Slice>): Unit = callback(slices)

    override fun toString(): String = "<func>"
}

public data class LazyBufferId(
    public val id: Int,
) : Comparable<LazyBufferId> {
    public fun asUsize(): Int = id

    override fun compareTo(other: LazyBufferId): Int = id.compareTo(other.id)

    public companion object {
        private var counter: Int = 1

        public fun new(): LazyBufferId {
            val v = counter
            counter += 1
            return LazyBufferId(v)
        }
    }
}

public sealed class Op {
    public data class Unary(
        public val op: UnaryOp,
        public val arg: LazyBuffer,
    ) : Op()

    public data class Binary(
        public val op: BinaryOp,
        public val lhs: LazyBuffer,
        public val rhs: LazyBuffer,
    ) : Op()

    public data class MatMul(
        public val lhs: LazyBuffer,
        public val rhs: LazyBuffer,
        public val bmnk: List<Int>,
        public val transpose: Boolean,
    ) : Op()

    public data class Reduce(
        public val op: ReduceOp,
        public val arg: LazyBuffer,
        public val axis: Int,
    ) : Op()

    public data class Const(
        public val c: io.github.kotlinmania.ug.Const,
    ) : Op()

    public object Value : Op()

    public data class Layout(
        public val op: LayoutOp,
        public val arg: LazyBuffer,
    ) : Op()

    public data class Reshape(
        public val arg: LazyBuffer,
    ) : Op()

    public data class Custom(
        public val f: CustomF,
        public val args: List<LazyBuffer>,
    ) : Op()

    public data class CustomIp(
        public val f: CustomF,
        public val args: List<LazyBuffer>,
        public val src: LazyBuffer,
    ) : Op()

    public data class Ssa(
        public val ssa: io.github.kotlinmania.ug.lang.ssa.Kernel,
        public val args: List<LazyBuffer>,
    ) : Op()

    public data class Set(
        public val values: LazyBuffer,
        public val src: LazyBuffer,
        public val dstLayout: io.github.kotlinmania.ug.Layout,
    ) : Op()
}

public class LazyBufferInner(
    public val id: LazyBufferId,
    public var data: Slice?,
    public val op: Op,
    public val dtype: DType,
    public val shape: Shape,
    public val device: Device,
)

public class LazyBuffer(
    public val inner: LazyBufferInner,
) {
    public val id: LazyBufferId get() = inner.id
    public val op: Op get() = inner.op
    public val dtype: DType get() = inner.dtype
    public val shape: Shape get() = inner.shape
    public val device: Device get() = inner.device
    public var data: Slice?
        get() = inner.data
        set(value) {
            inner.data = value
        }

    public fun maybeAllocateUninit() {
        if (inner.data == null) {
            inner.data = inner.device.allocateUninit(inner.dtype, inner.shape.numElements())
        }
    }

    public fun inPlaceOp(): Boolean =
        when (inner.op) {
            is Op.CustomIp, is Op.Set -> true
            else -> false
        }

    public fun realized(): Boolean {
        if (inPlaceOp()) return false
        return inner.data != null
    }

    override fun toString(): String = formatTensor()

    public fun realize() {
        if (realized()) return
        val schedule = Schedule.createOne(this)
        val compiled = schedule.compile()
        compiled.run()
    }

    public fun <T : Any> dataVec(withDType: WithDType<T>): List<T> {
        realize()
        val slice = inner.data ?: throw UgError.Msg("no data")
        return slice.toVec(withDType)
    }

    @Suppress("UNCHECKED_CAST")
    public inline fun <reified T : Any> dataVec(): List<T> {
        return when (T::class) {
            Float::class -> dataVec(WithDType.F32) as List<T>
            Int::class -> dataVec(WithDType.I32) as List<T>
            Long::class -> dataVec(WithDType.I64) as List<T>
            F16::class -> dataVec(WithDType.F16) as List<T>
            BF16::class -> dataVec(WithDType.BF16) as List<T>
            else -> throw UgError.Msg("unsupported dtype for dataVec")
        }
    }

    public fun elemCount(): Int = inner.shape.numElements()

    public fun dims(): List<Int> = inner.shape.dims

    public fun dims0(): Unit = inner.shape.dims0()

    public fun dims1(): Int = inner.shape.dims1()

    public fun dims2(): Pair<Int, Int> = inner.shape.dims2()

    public fun dims3(): Triple<Int, Int, Int> = inner.shape.dims3()

    public fun rank(): Int = inner.shape.rank()

    public fun unary(op: UnaryOp): LazyBuffer {
        val inner =
            LazyBufferInner(
                id = LazyBufferId.new(),
                data = null,
                op = Op.Unary(op, this),
                dtype = dtype,
                shape = shape,
                device = device,
            )
        return LazyBuffer(inner)
    }

    public fun cast(targetDType: DType): LazyBuffer {
        if (dtype == targetDType) return this
        val inner =
            LazyBufferInner(
                id = LazyBufferId.new(),
                data = null,
                op = Op.Unary(UnaryOp.Cast(targetDType), this),
                dtype = targetDType,
                shape = shape,
                device = device,
            )
        return LazyBuffer(inner)
    }

    public fun binary(op: BinaryOp, rhs: LazyBuffer): LazyBuffer {
        if (shape != rhs.shape) {
            ugBail("shape mismatch in $op, $shape vs ${rhs.shape}")
        }
        if (dtype != rhs.dtype) {
            ugBail("dtype mismatch in $op, $dtype vs ${rhs.dtype}")
        }
        val inner =
            LazyBufferInner(
                id = LazyBufferId.new(),
                data = null,
                op = Op.Binary(op, this, rhs),
                dtype = dtype,
                device = device,
                shape = shape,
            )
        return LazyBuffer(inner)
    }

    public fun custom(
        f: (List<Slice>) -> Unit,
        args: List<LazyBuffer>,
        s: Shape,
        dtype: DType,
        device: Device,
    ): LazyBuffer {
        val inner =
            LazyBufferInner(
                id = LazyBufferId.new(),
                data = null,
                op = Op.Custom(CustomF(f), args),
                dtype = dtype,
                device = device,
                shape = s,
            )
        return LazyBuffer(inner)
    }

    public fun customIp(
        f: (List<Slice>) -> Unit,
        args: List<LazyBuffer>,
    ): LazyBuffer {
        val inner =
            LazyBufferInner(
                id = LazyBufferId.new(),
                data = this.data,
                op = Op.CustomIp(CustomF(f), args, this),
                dtype = dtype,
                device = device,
                shape = shape,
            )
        return LazyBuffer(inner)
    }

    public fun set(values: LazyBuffer): LazyBuffer {
        if (shape != values.shape) {
            ugBail("shape mismatch in set, $shape vs ${values.shape}")
        }
        if (dtype != values.dtype) {
            ugBail("dtype mismatch in set, $dtype vs ${values.dtype}")
        }
        val dstLayout = io.github.kotlinmania.ug.Layout.fromShape(shape)
        val inner =
            LazyBufferInner(
                id = LazyBufferId.new(),
                data = this.data,
                op = Op.Set(values, this, dstLayout),
                dtype = dtype,
                device = device,
                shape = shape,
            )
        return LazyBuffer(inner)
    }

    public fun setL(values: LazyBuffer, dstLayout: io.github.kotlinmania.ug.Layout): LazyBuffer {
        var maxOffset = dstLayout.offset
        for ((d, s) in dstLayout.dims.zip(dstLayout.strides)) {
            maxOffset += maxOf(0, d - 1) * s
        }
        if (maxOffset >= shape.numElements()) {
            ugBail("set_l out of bounds, dst shape $shape, layout $dstLayout, max off $maxOffset")
        }
        if (dstLayout.numElements() != values.shape.numElements()) {
            ugBail("set_l mismatch between the values shape ${values.shape} and layout $dstLayout")
        }
        if (dtype != values.dtype) {
            ugBail("dtype mismatch in set, $dtype vs ${values.dtype}")
        }
        val inner =
            LazyBufferInner(
                id = LazyBufferId.new(),
                data = this.data,
                op = Op.Set(values, this, dstLayout),
                dtype = dtype,
                device = device,
                shape = shape,
            )
        return LazyBuffer(inner)
    }

    public fun matmul(rhs: LazyBuffer): LazyBuffer = matmul(rhs, false)

    public fun matmulT(rhs: LazyBuffer): LazyBuffer = matmul(rhs, true)

    public fun matmul(rhs: LazyBuffer, transpose: Boolean): LazyBuffer {
        val lhsDims = dims()
        val rhsDims = rhs.dims()
        val dim = lhsDims.size
        val rdim = rhsDims.size

        if (dim < 2 || rdim < 2) {
            ugBail("shape mismatch in matmul $lhsDims $rhsDims")
        }

        val m = lhsDims[dim - 2]
        val k = lhsDims[dim - 1]
        val (k2, n) =
            if (transpose) {
                Pair(rhsDims[rdim - 1], rhsDims[rdim - 2])
            } else {
                Pair(rhsDims[rdim - 2], rhsDims[rdim - 1])
            }

        val lhsBsz = lhsDims.take(dim - 2).fold(1) { acc, v -> acc * v }
        val rhsBsz = rhsDims.take(rdim - 2).fold(1) { acc, v -> acc * v }
        if (k != k2 || lhsBsz != rhsBsz) {
            ugBail("shape mismatch in matmul $lhsDims $rhsDims")
        }
        val bmnk = listOf(lhsBsz, m, n, k)
        val shapeList = lhsDims.take(dim - 2).toMutableList()
        shapeList.add(m)
        shapeList.add(n)
        val inner =
            LazyBufferInner(
                id = LazyBufferId.new(),
                data = null,
                op = Op.MatMul(this, rhs, bmnk, transpose),
                dtype = dtype,
                device = device,
                shape = Shape(shapeList),
            )
        return LazyBuffer(inner)
    }

    public fun reduce(op: ReduceOp, dim: Dim): LazyBuffer {
        val axis = dim.toIndex(shape, "reduce")
        val inner =
            LazyBufferInner(
                id = LazyBufferId.new(),
                data = null,
                op = Op.Reduce(op, this, axis),
                dtype = dtype,
                device = device,
                shape = shape,
            )
        return LazyBuffer(inner)
    }

    public fun splitDim(dim: Dim, size1: Int, size2: Int): LazyBuffer {
        val d = dim.toIndex(shape, "split_dim")
        val dimsList = dims().toMutableList()
        val size = dimsList.removeAt(d)
        if (size1 * size2 != size) {
            ugBail("unexpected target sizes for split_dim $d, ${size1}x$size2 != $size")
        }
        dimsList.add(d, size2)
        dimsList.add(d, size1)
        val inner =
            LazyBufferInner(
                id = LazyBufferId.new(),
                data = null,
                op = Op.Layout(LayoutOp.SplitDim(dim = d, lhs = d, rhs = d + 1), this),
                dtype = dtype,
                device = device,
                shape = Shape(dimsList),
            )
        return LazyBuffer(inner)
    }

    public fun mergeDims(dim: Dim): LazyBuffer {
        val d = dim.toIndex(shape, "merge_dims")
        if (d + 1 >= rank()) {
            ugBail("unexpected dim for merge_dims $d $shape")
        }
        val dimsList = dims().toMutableList()
        val sizeP = dimsList.removeAt(d + 1)
        dimsList[d] *= sizeP
        val inner =
            LazyBufferInner(
                id = LazyBufferId.new(),
                data = null,
                op = Op.Layout(LayoutOp.MergeDims(dim = d, lhs = d, rhs = d + 1), this),
                dtype = dtype,
                device = device,
                shape = Shape(dimsList),
            )
        return LazyBuffer(inner)
    }

    public fun squeeze(dim: Dim): LazyBuffer {
        val d = dim.toIndex(shape, "squeeze")
        val dimsList = dims().toMutableList()
        if (dimsList[d] != 1) {
            ugBail("cannot squeeze on dim $d for shape ${dims()}")
        }
        dimsList.removeAt(d)
        return reshape(Shape(dimsList))
    }

    public fun getOnDim(dim: Dim, index: Int): LazyBuffer {
        val d = dim.toIndex(shape, "get")
        val narrowed = narrow(d.D, index, 1)
        return narrowed.squeeze(d.D)
    }

    public fun get(index: Int): LazyBuffer = getOnDim(0.D, index)

    public fun unsqueeze(dim: Dim): LazyBuffer {
        val dimsList = dims().toMutableList()
        val d = dim.toIndexPlusOne(shape, "unsqueeze")
        dimsList.add(d, 1)
        return reshape(Shape(dimsList))
    }

    public fun reshape(s: Shape): LazyBuffer {
        val dstNel = s.numElements()
        val srcNel = shape.numElements()
        if (dstNel != srcNel) {
            ugBail("cannot reshape between $shape ($srcNel elts) and $s ($dstNel elts)")
        }
        val inner =
            LazyBufferInner(
                id = LazyBufferId.new(),
                data = null,
                op = Op.Reshape(this),
                dtype = dtype,
                device = device,
                shape = s,
            )
        return LazyBuffer(inner)
    }

    public fun narrow(dim: Dim, offset: Int, length: Int): LazyBuffer {
        val d = dim.toIndex(shape, "narrow")
        val size = dims()[d]
        if (offset + length > size) {
            ugBail("cannot narrow, offset $offset + length $length > size $d")
        }
        if (offset == 0 && length == size) {
            return this
        }
        val dimsList = dims().toMutableList()
        dimsList[d] = length
        val inner =
            LazyBufferInner(
                id = LazyBufferId.new(),
                data = null,
                op = Op.Layout(LayoutOp.Narrow(dim = d, offset = offset), this),
                dtype = dtype,
                device = device,
                shape = Shape(dimsList),
            )
        return LazyBuffer(inner)
    }

    public fun broadcast(s: Shape): LazyBuffer {
        if (rank() > s.rank()) {
            ugBail("target shape $s has less dimensions than original shape $shape")
        }
        val insertedDims = s.rank() - rank()
        val broadcastedDims = (0 until insertedDims).toMutableList()
        for ((dimIdx, dimLen) in dims().withIndex()) {
            val targetIdx = dimIdx + insertedDims
            if (s.dims[targetIdx] != dimLen) {
                if (dimLen == 1) {
                    broadcastedDims.add(targetIdx)
                } else {
                    ugBail("cannot broadcast from $shape to $s")
                }
            }
        }
        val inner =
            LazyBufferInner(
                id = LazyBufferId.new(),
                data = null,
                op =
                    Op.Layout(
                        LayoutOp.Broadcast(insertedDims, broadcastedDims),
                        this,
                    ),
                dtype = dtype,
                device = device,
                shape = s,
            )
        return LazyBuffer(inner)
    }

    public fun transpose(dim1: Dim, dim2: Dim): LazyBuffer {
        val d1 = dim1.toIndex(shape, "transpose")
        val d2 = dim2.toIndex(shape, "transpose")
        if (d1 == d2) return this
        val dimsList = dims().toMutableList()
        val tmp = dimsList[d1]
        dimsList[d1] = dimsList[d2]
        dimsList[d2] = tmp
        val inner =
            LazyBufferInner(
                id = LazyBufferId.new(),
                data = null,
                op = Op.Layout(LayoutOp.Transpose(dim1 = d1, dim2 = d2), this),
                dtype = dtype,
                device = device,
                shape = Shape(dimsList),
            )
        return LazyBuffer(inner)
    }

    public companion object {
        public fun allocUninit(dtype: DType, s: Shape, device: Device): LazyBuffer {
            val inner =
                LazyBufferInner(
                    id = LazyBufferId.new(),
                    data = null,
                    op = Op.Value,
                    dtype = dtype,
                    device = device,
                    shape = s,
                )
            return LazyBuffer(inner)
        }

        public fun cst(c: Const, s: Shape, device: Device): LazyBuffer {
            val inner =
                LazyBufferInner(
                    id = LazyBufferId.new(),
                    data = null,
                    op = Op.Const(c),
                    dtype = c.dtype,
                    device = device,
                    shape = s,
                )
            return LazyBuffer(inner)
        }

        public fun cst(c: Float, s: Shape, device: Device): LazyBuffer =
            cst(Const.F32(c), s, device)

        public fun cst(c: Double, s: Shape, device: Device): LazyBuffer =
            cst(Const.F32(c.toFloat()), s, device)

        public fun cst(c: Float, s: Pair<Int, Int>, device: Device): LazyBuffer =
            cst(c, Shape(s.first, s.second), device)

        public fun cst(c: Double, s: Pair<Int, Int>, device: Device): LazyBuffer =
            cst(c, Shape(s.first, s.second), device)

        public fun cst(c: Int, s: Shape, device: Device): LazyBuffer =
            cst(Const.I32(c), s, device)

        public fun cst(c: Int, s: Pair<Int, Int>, device: Device): LazyBuffer =
            cst(c, Shape(s.first, s.second), device)

        public fun fromSlice(data: Slice, s: Shape): LazyBuffer {
            val device = data.device
            val dtype = data.dtype
            if (s.numElements() != data.len) {
                ugBail("unexpected number of elements ${data.len} for shape $s")
            }
            val inner =
                LazyBufferInner(
                    id = LazyBufferId.new(),
                    data = data,
                    op = Op.Value,
                    dtype = dtype,
                    device = device,
                    shape = s,
                )
            return LazyBuffer(inner)
        }

        public fun copy(data: FloatArray, s: Shape, device: Device): LazyBuffer {
            if (s.numElements() != data.size) {
                ugBail("unexpected number of elements ${data.size} for shape $s")
            }
            val slice = device.allocateUninit(DType.F32, data.size)
            slice.copyHostToDevice(data)
            val inner =
                LazyBufferInner(
                    id = LazyBufferId.new(),
                    data = slice,
                    op = Op.Value,
                    dtype = DType.F32,
                    device = device,
                    shape = s,
                )
            return LazyBuffer(inner)
        }

        public fun copy(data: FloatArray, s: Pair<Int, Int>, device: Device): LazyBuffer =
            copy(data, Shape(s.first, s.second), device)

        public fun copy(data: IntArray, s: Shape, device: Device): LazyBuffer {
            if (s.numElements() != data.size) {
                ugBail("unexpected number of elements ${data.size} for shape $s")
            }
            val slice = device.allocateUninit(DType.I32, data.size)
            slice.copyHostToDevice(data)
            val inner =
                LazyBufferInner(
                    id = LazyBufferId.new(),
                    data = slice,
                    op = Op.Value,
                    dtype = DType.I32,
                    device = device,
                    shape = s,
                )
            return LazyBuffer(inner)
        }

        public fun copy(data: LongArray, s: Shape, device: Device): LazyBuffer {
            if (s.numElements() != data.size) {
                ugBail("unexpected number of elements ${data.size} for shape $s")
            }
            val slice = device.allocateUninit(DType.I64, data.size)
            slice.copyHostToDevice(data)
            val inner =
                LazyBufferInner(
                    id = LazyBufferId.new(),
                    data = slice,
                    op = Op.Value,
                    dtype = DType.I64,
                    device = device,
                    shape = s,
                )
            return LazyBuffer(inner)
        }

        public fun copy(data: Array<F16>, s: Shape, device: Device): LazyBuffer {
            if (s.numElements() != data.size) {
                ugBail("unexpected number of elements ${data.size} for shape $s")
            }
            val slice = device.allocateUninit(DType.F16, data.size)
            slice.copyHostToDevice(data)
            val inner =
                LazyBufferInner(
                    id = LazyBufferId.new(),
                    data = slice,
                    op = Op.Value,
                    dtype = DType.F16,
                    device = device,
                    shape = s,
                )
            return LazyBuffer(inner)
        }

        public fun copy(data: Array<BF16>, s: Shape, device: Device): LazyBuffer {
            if (s.numElements() != data.size) {
                ugBail("unexpected number of elements ${data.size} for shape $s")
            }
            val slice = device.allocateUninit(DType.BF16, data.size)
            slice.copyHostToDevice(data)
            val inner =
                LazyBufferInner(
                    id = LazyBufferId.new(),
                    data = slice,
                    op = Op.Value,
                    dtype = DType.BF16,
                    device = device,
                    shape = s,
                )
            return LazyBuffer(inner)
        }

        public fun custom(
            f: (List<Slice>) -> Unit,
            args: List<LazyBuffer>,
            s: Shape,
            dtype: DType,
            device: Device,
        ): LazyBuffer {
            val inner =
                LazyBufferInner(
                    id = LazyBufferId.new(),
                    data = null,
                    op = Op.Custom(CustomF(f), args),
                    dtype = dtype,
                    device = device,
                    shape = s,
                )
            return LazyBuffer(inner)
        }

        public fun ssa(
            ssa: io.github.kotlinmania.ug.lang.ssa.Kernel,
            args: List<LazyBuffer>,
            s: Shape,
            dtype: DType,
            device: Device,
        ): LazyBuffer {
            val inner =
                LazyBufferInner(
                    id = LazyBufferId.new(),
                    data = null,
                    op = Op.Ssa(ssa, args),
                    dtype = dtype,
                    device = device,
                    shape = s,
                )
            return LazyBuffer(inner)
        }

        public fun cat(args: List<LazyBuffer>, dim: Dim): LazyBuffer {
            if (args.isEmpty()) {
                ugBail("empty list in cat")
            }
            val arg0 = args[0]
            if (args.size == 1) {
                return arg0
            }
            val d = dim.toIndex(arg0.shape, "cat")
            val dimsList = arg0.dims().toMutableList()
            dimsList[d] = 0
            for ((argIdx, arg) in args.withIndex()) {
                if (arg.dtype != arg0.dtype) {
                    val dtypes = args.map { it.dtype }
                    ugBail("mismatch between dtypes in cat, shapes: $dtypes")
                }
                if (arg.rank() != arg0.rank()) {
                    val shapes = args.map { it.shape }
                    ugBail("mismatch between ranks in cat, shapes: $shapes")
                }
                dimsList[d] += arg.dims()[d]
                for (dimIdx in dimsList.indices) {
                    if (dimIdx != d && dimsList[dimIdx] != arg.dims()[dimIdx]) {
                        ugBail(
                            "mismatch between shapes in cat on dim $d, arg0: ${arg0.shape}, arg$argIdx: ${arg.shape}",
                        )
                    }
                }
            }
            var stride = 1
            for (dimLen in dimsList.subList(d + 1, dimsList.size)) {
                stride *= dimLen
            }
            var offset = 0
            var vs = allocUninit(arg0.dtype, Shape(dimsList), arg0.device)
            for (arg in args) {
                val dstLayout = io.github.kotlinmania.ug.Layout.fromShape(arg.shape).setOffset(offset * stride)
                vs = vs.setL(arg, dstLayout)
                offset += arg.dims()[d]
            }
            return vs
        }

        public fun cat(args: List<LazyBuffer>, dim: Int): LazyBuffer = cat(args, dim.D)
    }
}
