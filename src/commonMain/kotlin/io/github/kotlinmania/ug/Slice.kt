package io.github.kotlinmania.ug

import io.github.kotlinmania.half.BF16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.ug.cpuruntime.CpuStorage

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

public inline fun <reified T> Slice.toVec(): List<T> =
    when (T::class) {
        Float::class -> toVec(WithDType.F32).map { it as T }
        Int::class -> toVec(WithDType.I32).map { it as T }
        Long::class -> toVec(WithDType.I64).map { it as T }
        F16::class -> toVec(WithDType.F16).map { it as T }
        BF16::class -> toVec(WithDType.BF16).map { it as T }
        else -> UgError.Msg("unsupported type for toVec: ${T::class}").raise()
    }

public fun Slice.toVec(withDType: F32Type): List<Float> {
    val storage = this as? CpuStorage ?: CpuStorage.F32Storage(toFloatArray())
    return withDType.fromCpuStorage(storage)
}

public fun Slice.toVec(withDType: I32Type): List<Int> {
    val storage = this as? CpuStorage ?: CpuStorage.I32Storage(toIntArray())
    return withDType.fromCpuStorage(storage)
}

public fun Slice.toVec(withDType: I64Type): List<Long> {
    val storage = this as? CpuStorage ?: CpuStorage.I64Storage(toLongArray())
    return withDType.fromCpuStorage(storage)
}

public fun Slice.toVec(withDType: F16Type): List<F16> {
    val storage =
        this as? CpuStorage ?: run {
            val data = Array(len) { F16() }
            copyDeviceToHost(data)
            CpuStorage.F16Storage(data)
        }
    return withDType.fromCpuStorage(storage)
}

public fun Slice.toVec(withDType: BF16Type): List<BF16> {
    val storage =
        this as? CpuStorage ?: run {
            val data = Array(len) { BF16() }
            copyDeviceToHost(data)
            CpuStorage.BF16Storage(data)
        }
    return withDType.fromCpuStorage(storage)
}
