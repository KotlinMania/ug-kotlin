package io.github.kotlinmania.ug

import io.github.kotlinmania.half.BF16
import io.github.kotlinmania.half.F16

public interface Slice {
    public val device: Device
    public val dtype: DType
    public val len: Int
    public val isEmpty: Boolean get() = len == 0

    public fun copyHostToDevice(src: FloatArray)

    public fun copyHostToDevice(src: IntArray)

    public fun copyHostToDevice(src: LongArray)

    public fun copyHostToDevice(src: Array<F16>)

    public fun copyHostToDevice(src: Array<BF16>)

    public fun copyDeviceToHost(dst: FloatArray)

    public fun copyDeviceToHost(dst: IntArray)

    public fun copyDeviceToHost(dst: LongArray)

    public fun copyDeviceToHost(dst: Array<F16>)

    public fun copyDeviceToHost(dst: Array<BF16>)

    public fun toFloatArray(): FloatArray

    public fun toIntArray(): IntArray

    public fun toLongArray(): LongArray
}

@Suppress("UNCHECKED_CAST")
public inline fun <reified T> Slice.toVec(): List<T> =
    when (T::class) {
        Float::class -> {
            toFloatArray().toList() as List<T>
        }

        Int::class -> {
            toIntArray().toList() as List<T>
        }

        Long::class -> {
            toLongArray().toList() as List<T>
        }

        F16::class -> {
            val storage =
                this as? io.github.kotlinmania.ug.cpuruntime.CpuStorage.F16Storage
                    ?: throw UgError.Msg("expected F16Storage")
            storage.data.toList() as List<T>
        }

        BF16::class -> {
            val storage =
                this as? io.github.kotlinmania.ug.cpuruntime.CpuStorage.BF16Storage
                    ?: throw UgError.Msg("expected BF16Storage")
            storage.data.toList() as List<T>
        }

        else -> {
            throw UgError.Msg("unsupported type for toVec: ${T::class}")
        }
    }

public fun <T : Any> Slice.toVec(withDType: WithDType<T>): List<T> {
    val storage = this as? io.github.kotlinmania.ug.cpuruntime.CpuStorage
    if (storage != null) {
        return withDType.fromCpuStorage(storage)
    }
    @Suppress("UNCHECKED_CAST")
    return when (withDType.dtype) {
        DType.F32 -> toFloatArray().toList() as List<T>
        DType.I32 -> toIntArray().toList() as List<T>
        DType.I64 -> toLongArray().toList() as List<T>
        DType.F16 -> {
            val dst = Array(len) { io.github.kotlinmania.half.F16() }
            copyDeviceToHost(dst)
            dst.toList() as List<T>
        }
        DType.BF16 -> {
            val dst = Array(len) { io.github.kotlinmania.half.BF16() }
            copyDeviceToHost(dst)
            dst.toList() as List<T>
        }
    }
}
