// port-lint: source dtype.rs
package io.github.kotlinmania.ug

import io.github.kotlinmania.ug.cpuruntime.CpuStorage
import io.github.kotlinmania.half.BF16 as HalfBF16
import io.github.kotlinmania.half.F16 as HalfF16

public sealed class WithDType {
    public abstract val dtype: DType

    public companion object {
        public val F32: F32Type = F32Type
        public val I32: I32Type = I32Type
        public val I64: I64Type = I64Type
        public val F16: F16Type = F16Type
        public val BF16: BF16Type = BF16Type
    }
}

public object F32Type : WithDType() {
    override val dtype: DType get() = DType.F32

    public fun zero(): Float = 0.0f

    public fun one(): Float = 1.0f

    public fun createArray(len: Int): List<Float> = FloatArray(len).toList()

    public fun toCpuStorage(data: List<Float>): CpuStorage = CpuStorage.F32Storage(data.toFloatArray())

    public fun fromCpuStorage(storage: CpuStorage): List<Float> =
        (storage as? CpuStorage.F32Storage)?.data?.toList() ?: UgError.Msg("expected F32Storage").raise()
}

public object I32Type : WithDType() {
    override val dtype: DType get() = DType.I32

    public fun zero(): Int = 0

    public fun one(): Int = 1

    public fun createArray(len: Int): List<Int> = IntArray(len).toList()

    public fun toCpuStorage(data: List<Int>): CpuStorage = CpuStorage.I32Storage(data.toIntArray())

    public fun fromCpuStorage(storage: CpuStorage): List<Int> =
        (storage as? CpuStorage.I32Storage)?.data?.toList() ?: UgError.Msg("expected I32Storage").raise()
}

public object I64Type : WithDType() {
    override val dtype: DType get() = DType.I64

    public fun zero(): Long = 0L

    public fun one(): Long = 1L

    public fun createArray(len: Int): List<Long> = LongArray(len).toList()

    public fun toCpuStorage(data: List<Long>): CpuStorage = CpuStorage.I64Storage(data.toLongArray())

    public fun fromCpuStorage(storage: CpuStorage): List<Long> =
        (storage as? CpuStorage.I64Storage)?.data?.toList() ?: UgError.Msg("expected I64Storage").raise()
}

public object F16Type : WithDType() {
    override val dtype: DType get() = DType.F16

    public fun zero(): HalfF16 = HalfF16()

    public fun one(): HalfF16 = HalfF16.fromF32(1.0f)

    public fun createArray(len: Int): List<HalfF16> = List(len) { HalfF16() }

    public fun toCpuStorage(data: List<HalfF16>): CpuStorage = CpuStorage.F16Storage(data.toTypedArray())

    public fun fromCpuStorage(storage: CpuStorage): List<HalfF16> =
        (storage as? CpuStorage.F16Storage)?.data?.toList() ?: UgError.Msg("expected F16Storage").raise()
}

public object BF16Type : WithDType() {
    override val dtype: DType get() = DType.BF16

    public fun zero(): HalfBF16 = HalfBF16()

    public fun one(): HalfBF16 = HalfBF16.fromF32(1.0f)

    public fun createArray(len: Int): List<HalfBF16> = List(len) { HalfBF16() }

    public fun toCpuStorage(data: List<HalfBF16>): CpuStorage = CpuStorage.BF16Storage(data.toTypedArray())

    public fun fromCpuStorage(storage: CpuStorage): List<HalfBF16> =
        (storage as? CpuStorage.BF16Storage)?.data?.toList() ?: UgError.Msg("expected BF16Storage").raise()
}
