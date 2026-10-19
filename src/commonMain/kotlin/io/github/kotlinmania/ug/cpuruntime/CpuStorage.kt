// port-lint: source cpu_runtime.rs
package io.github.kotlinmania.ug.cpuruntime

import io.github.kotlinmania.half.BF16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.ug.DType
import io.github.kotlinmania.ug.Device
import io.github.kotlinmania.ug.Slice
import io.github.kotlinmania.ug.UgError

public sealed class CpuStorage : Slice {
    public class BF16Storage internal constructor(
        public val data: Array<BF16>,
    ) : CpuStorage() {
        override val len: Int get() = data.size
        override val dtype: DType get() = DType.BF16

        override fun equals(other: Any?): Boolean = other is BF16Storage && data.contentEquals(other.data)

        override fun hashCode(): Int = data.contentHashCode()
    }

    public class F16Storage internal constructor(
        public val data: Array<F16>,
    ) : CpuStorage() {
        override val len: Int get() = data.size
        override val dtype: DType get() = DType.F16

        override fun equals(other: Any?): Boolean = other is F16Storage && data.contentEquals(other.data)

        override fun hashCode(): Int = data.contentHashCode()
    }

    public data class F32Storage(
        public val data: FloatArray,
    ) : CpuStorage() {
        override val len: Int get() = data.size
        override val dtype: DType get() = DType.F32

        override fun equals(other: Any?): Boolean = other is F32Storage && data.contentEquals(other.data)

        override fun hashCode(): Int = data.contentHashCode()
    }

    public data class I32Storage(
        public val data: IntArray,
    ) : CpuStorage() {
        override val len: Int get() = data.size
        override val dtype: DType get() = DType.I32

        override fun equals(other: Any?): Boolean = other is I32Storage && data.contentEquals(other.data)

        override fun hashCode(): Int = data.contentHashCode()
    }

    public data class I64Storage(
        public val data: LongArray,
    ) : CpuStorage() {
        override val len: Int get() = data.size
        override val dtype: DType get() = DType.I64

        override fun equals(other: Any?): Boolean = other is I64Storage && data.contentEquals(other.data)

        override fun hashCode(): Int = data.contentHashCode()
    }

    override val device: Device get() = CpuDevice

    override fun copyHostToDevice(src: FloatArray) {
        when (this) {
            is F32Storage -> {
                if (src.size != len) throw UgError.Msg("htod len mismatch, dst $len, len ${src.size}")
                src.copyInto(data)
            }

            else -> {
                throw UgError.Msg("htod dtype mismatch, dst $dtype, src F32")
            }
        }
    }

    override fun copyHostToDevice(src: IntArray) {
        when (this) {
            is I32Storage -> {
                if (src.size != len) throw UgError.Msg("htod len mismatch, dst $len, len ${src.size}")
                src.copyInto(data)
            }

            else -> {
                throw UgError.Msg("htod dtype mismatch, dst $dtype, src I32")
            }
        }
    }

    override fun copyHostToDevice(src: LongArray) {
        when (this) {
            is I64Storage -> {
                if (src.size != len) throw UgError.Msg("htod len mismatch, dst $len, len ${src.size}")
                src.copyInto(data)
            }

            else -> {
                throw UgError.Msg("htod dtype mismatch, dst $dtype, src I64")
            }
        }
    }

    override fun copyHostToDevice(src: Array<F16>) {
        when (this) {
            is F16Storage -> {
                if (src.size != len) throw UgError.Msg("htod len mismatch, dst $len, len ${src.size}")
                src.copyInto(data)
            }

            else -> {
                throw UgError.Msg("htod dtype mismatch, dst $dtype, src F16")
            }
        }
    }

    override fun copyHostToDevice(src: Array<BF16>) {
        when (this) {
            is BF16Storage -> {
                if (src.size != len) throw UgError.Msg("htod len mismatch, dst $len, len ${src.size}")
                src.copyInto(data)
            }

            else -> {
                throw UgError.Msg("htod dtype mismatch, dst $dtype, src BF16")
            }
        }
    }

    override fun copyDeviceToHost(dst: FloatArray) {
        when (this) {
            is F32Storage -> {
                if (dst.size != len) throw UgError.Msg("dtoh len mismatch, dst ${dst.size}, len $len")
                data.copyInto(dst)
            }

            else -> {
                throw UgError.Msg("dtoh dtype mismatch, dst F32, src $dtype")
            }
        }
    }

    override fun copyDeviceToHost(dst: IntArray) {
        when (this) {
            is I32Storage -> {
                if (dst.size != len) throw UgError.Msg("dtoh len mismatch, dst ${dst.size}, len $len")
                data.copyInto(dst)
            }

            else -> {
                throw UgError.Msg("dtoh dtype mismatch, dst I32, src $dtype")
            }
        }
    }

    override fun copyDeviceToHost(dst: LongArray) {
        when (this) {
            is I64Storage -> {
                if (dst.size != len) throw UgError.Msg("dtoh len mismatch, dst ${dst.size}, len $len")
                data.copyInto(dst)
            }

            else -> {
                throw UgError.Msg("dtoh dtype mismatch, dst I64, src $dtype")
            }
        }
    }

    override fun copyDeviceToHost(dst: Array<F16>) {
        when (this) {
            is F16Storage -> {
                if (dst.size != len) throw UgError.Msg("dtoh len mismatch, dst ${dst.size}, len $len")
                data.copyInto(dst)
            }

            else -> {
                throw UgError.Msg("dtoh dtype mismatch, dst F16, src $dtype")
            }
        }
    }

    override fun copyDeviceToHost(dst: Array<BF16>) {
        when (this) {
            is BF16Storage -> {
                if (dst.size != len) throw UgError.Msg("dtoh len mismatch, dst ${dst.size}, len $len")
                data.copyInto(dst)
            }

            else -> {
                throw UgError.Msg("dtoh dtype mismatch, dst BF16, src $dtype")
            }
        }
    }

    override fun toFloatArray(): FloatArray =
        when (this) {
            is F32Storage -> data.copyOf()
            is I32Storage -> FloatArray(len) { data[it].toFloat() }
            is I64Storage -> FloatArray(len) { data[it].toFloat() }
            is F16Storage -> FloatArray(len) { data[it].toFloat() }
            is BF16Storage -> FloatArray(len) { data[it].toFloat() }
        }

    override fun toIntArray(): IntArray =
        when (this) {
            is I32Storage -> data.copyOf()
            is I64Storage -> IntArray(len) { data[it].toInt() }
            is F32Storage -> IntArray(len) { data[it].toInt() }
            is F16Storage -> IntArray(len) { data[it].toFloat().toInt() }
            is BF16Storage -> IntArray(len) { data[it].toFloat().toInt() }
        }

    override fun toLongArray(): LongArray =
        when (this) {
            is I64Storage -> data.copyOf()
            is I32Storage -> LongArray(len) { data[it].toLong() }
            is F32Storage -> LongArray(len) { data[it].toLong() }
            is F16Storage -> LongArray(len) { data[it].toFloat().toLong() }
            is BF16Storage -> LongArray(len) { data[it].toFloat().toLong() }
        }

    public companion object {
        public fun from(data: FloatArray): F32Storage = F32Storage(data.copyOf())

        public fun from(data: IntArray): I32Storage = I32Storage(data.copyOf())

        public fun from(data: LongArray): I64Storage = I64Storage(data.copyOf())

        internal fun from(data: Array<F16>): F16Storage = F16Storage(data.copyOf())

        internal fun from(data: Array<BF16>): BF16Storage = BF16Storage(data.copyOf())
    }
}
