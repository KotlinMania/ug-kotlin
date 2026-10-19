// port-lint: source cpu_runtime.rs
package io.github.kotlinmania.ug.cpuruntime

import io.github.kotlinmania.half.BF16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.ug.DType
import io.github.kotlinmania.ug.Device
import io.github.kotlinmania.ug.Layout
import io.github.kotlinmania.ug.MatMulUnexpectedStriding
import io.github.kotlinmania.ug.Shape
import io.github.kotlinmania.ug.Slice
import io.github.kotlinmania.ug.UgError
import io.github.kotlinmania.ug.interpreter.Buffer
import io.github.kotlinmania.ug.interpreter.evalSsa
import io.github.kotlinmania.ug.lang.ssa.Kernel

public data class KernelId(
    public val id: Int,
) {
    public fun asUsize(): Int = id

    public companion object {
        private var counter: Int = 1

        public fun new(): KernelId {
            val v = counter
            counter += 1
            return KernelId(v)
        }
    }
}

public class CpuFunc(
    public val funcName: String,
    public val kernel: Kernel? = null,
    public val cCode: String? = null,
) {
    public fun name(): String = funcName

    public fun run0() {}

    public fun run3(v1: IntArray, v2: IntArray, v3: IntArray) {
        if (kernel != null) {
            val buffers = listOf(Buffer.I32(v1), Buffer.I32(v2), Buffer.I32(v3))
            evalSsa(kernel, buffers)
        } else {
            // Fallback for simple add sample if cCode was provided without kernel
            val n = minOf(v1.size, v2.size, v3.size)
            for (i in 0 until n) {
                v1[i] = v2[i] + v3[i]
            }
        }
    }

    public fun run3(v1: FloatArray, v2: FloatArray, v3: FloatArray) {
        if (kernel != null) {
            val buffers = listOf(Buffer.F32(v1), Buffer.F32(v2), Buffer.F32(v3))
            evalSsa(kernel, buffers)
        } else {
            val n = minOf(v1.size, v2.size, v3.size)
            for (i in 0 until n) {
                v1[i] = v2[i] + v3[i]
            }
        }
    }
}

public class MatMul(
    public val bmnk: List<Int>,
) {
    public constructor(b: Int, m: Int, n: Int, k: Int) : this(listOf(b, m, n, k))

    private fun stridingError(lhsL: Layout, rhsL: Layout, msg: String): UgError =
        UgError.MatMulStriding(MatMulUnexpectedStriding(lhsL, rhsL, bmnk, msg))

    public fun abSkip(lhsL: Layout, rhsL: Layout): Pair<Int, Int> {
        val lhsStride = lhsL.strides
        val rhsStride = rhsL.strides
        val rank = lhsStride.size
        val m = bmnk[1]
        val n = bmnk[2]
        val k = bmnk[3]

        val lhsPrefix = if (rank >= 2) lhsStride.subList(0, rank - 2) else emptyList()
        val aSkip: Int =
            when (lhsPrefix.size) {
                2 -> {
                    val s0 = lhsPrefix[0]
                    val s1 = lhsPrefix[1]
                    if (s0 == s1 * lhsL.dims[1]) s1
                    else if (lhsL.dims[0] == 1) s1
                    else if (lhsL.dims[1] == 1) s0
                    else throw stridingError(lhsL, rhsL, "non-contiguous lhs")
                }

                1 -> lhsPrefix[0]
                0 -> m * k
                else -> throw stridingError(lhsL, rhsL, "non-contiguous lhs")
            }

        val rhsPrefix = if (rank >= 2) rhsStride.subList(0, rank - 2) else emptyList()
        val bSkip: Int =
            when (rhsPrefix.size) {
                2 -> {
                    val s0 = rhsPrefix[0]
                    val s1 = rhsPrefix[1]
                    if (s0 == s1 * rhsL.dims[1]) s1
                    else if (rhsL.dims[0] == 1) s1
                    else if (rhsL.dims[1] == 1) s0
                    else throw stridingError(lhsL, rhsL, "non-contiguous rhs")
                }

                1 -> rhsPrefix[0]
                0 -> n * k
                else -> throw stridingError(lhsL, rhsL, "non-contiguous rhs")
            }

        return Pair(aSkip, bSkip)
    }

    public fun gemm(
        dst: FloatArray,
        lhs: FloatArray,
        lhsL: Layout,
        rhs: FloatArray,
        rhsL: Layout,
    ) {
        val b = bmnk[0]
        val m = bmnk[1]
        val n = bmnk[2]
        val k = bmnk[3]

        val lhsStrides = lhsL.strides
        val rhsStrides = rhsL.strides
        val rankLhs = lhsStrides.size
        val rankRhs = rhsStrides.size

        val lhsCs = lhsStrides[rankLhs - 1]
        val lhsRs = lhsStrides[rankLhs - 2]
        val rhsCs = rhsStrides[rankRhs - 1]
        val rhsRs = rhsStrides[rankRhs - 2]

        val (aSkip, bSkip) = abSkip(lhsL, rhsL)
        val cSkip = m * n

        val dstShape = Shape(m, n)
        val dstStrides = dstShape.strideContiguous()
        val dstRs = dstStrides[0]
        val dstCs = dstStrides[1]

        val lhsOffset = lhsL.offset
        val rhsOffset = rhsL.offset

        for (step in 0 until b) {
            val lhsP = lhsOffset + step * aSkip
            val rhsP = rhsOffset + step * bSkip
            val dstP = step * cSkip

            for (i in 0 until m) {
                for (j in 0 until n) {
                    var acc = 0.0f
                    for (p in 0 until k) {
                        val lVal = lhs[lhsP + i * lhsRs + p * lhsCs]
                        val rVal = rhs[rhsP + p * rhsRs + j * rhsCs]
                        acc += lVal * rVal
                    }
                    dst[dstP + i * dstRs + j * dstCs] = acc
                }
            }
        }
    }

    public fun gemm(
        dst: CpuStorage,
        lhs: CpuStorage,
        lhsL: Layout,
        rhs: CpuStorage,
        rhsL: Layout,
    ) {
        when {
            dst is CpuStorage.F32Storage && lhs is CpuStorage.F32Storage && rhs is CpuStorage.F32Storage -> {
                gemm(dst.data, lhs.data, lhsL, rhs.data, rhsL)
            }

            else -> throw UgError.Msg("incorrect dtypes for matmul, dst: ${dst.dtype}, lhs: ${lhs.dtype}, rhs: ${rhs.dtype}")
        }
    }
}

public object CpuDevice : Device {
    override fun allocateUninit(dtype: DType, len: Int): CpuStorage =
        when (dtype) {
            DType.BF16 -> CpuStorage.BF16Storage(Array(len) { BF16() })
            DType.F16 -> CpuStorage.F16Storage(Array(len) { F16() })
            DType.F32 -> CpuStorage.F32Storage(FloatArray(len))
            DType.I32 -> CpuStorage.I32Storage(IntArray(len))
            DType.I64 -> CpuStorage.I64Storage(LongArray(len))
        }

    override fun synchronize() {}

    override fun useGrid(): Boolean = false

    override fun compile(kernel: Kernel, name: String?): CpuFunc {
        val kernelId = KernelId.new().asUsize()
        val funcName = name?.let { "ugc_${it}_$kernelId" } ?: "ugc_$kernelId"
        val cCode = io.github.kotlinmania.ug.cpucodegen.gen(funcName, kernel)
        return CpuFunc(funcName, kernel, cCode)
    }

    public fun compileC(cCode: String, funcName: String): CpuFunc = CpuFunc(funcName, kernel = null, cCode = cCode)

    public fun compileC(cCodeBytes: ByteArray, funcName: String): CpuFunc =
        compileC(cCodeBytes.decodeToString(), funcName)

    override fun run(f: Any, args: List<Slice>) {
        val func = f as? CpuFunc ?: throw UgError.Msg("invalid func type for CpuDevice")
        val kernel = func.kernel ?: throw UgError.Msg("kernel is null in CpuFunc")
        val buffers =
            args.map { slice ->
                when (slice) {
                    is CpuStorage.F32Storage -> Buffer.F32(slice.data)
                    is CpuStorage.I32Storage -> Buffer.I32(slice.data)
                    is CpuStorage.I64Storage -> Buffer.I64(slice.data)
                    else -> throw UgError.Msg("unsupported storage for run: ${slice.dtype}")
                }
            }
        evalSsa(kernel, buffers)
    }

    override fun matmul(
        dst: Slice,
        lhs: Slice,
        rhs: Slice,
        b: Int,
        m: Int,
        n: Int,
        k: Int,
        lhsL: Layout,
        rhsL: Layout,
    ) {
        val mm = MatMul(b, m, n, k)
        val dstStorage = dst as? CpuStorage ?: throw UgError.Msg("dst must be CpuStorage")
        val lhsStorage = lhs as? CpuStorage ?: throw UgError.Msg("lhs must be CpuStorage")
        val rhsStorage = rhs as? CpuStorage ?: throw UgError.Msg("rhs must be CpuStorage")
        mm.gemm(dstStorage, lhsStorage, lhsL, rhsStorage, rhsL)
    }
}
