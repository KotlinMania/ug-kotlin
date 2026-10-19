// port-lint: source dtype.rs
package io.github.kotlinmania.ug

import io.github.kotlinmania.half.BF16 as HalfBF16
import io.github.kotlinmania.half.F16 as HalfF16
import io.github.kotlinmania.ug.cpu_runtime.CpuStorage

public interface WithDType<T> {
    public val dtype: DType

    public fun zero(): T

    public fun one(): T

    public fun createArray(len: Int): List<T>

    public fun toCpuStorage(data: List<T>): CpuStorage

    public fun fromCpuStorage(storage: CpuStorage): List<T>

    public companion object {
        public val F32: WithDType<Float> =
            object : WithDType<Float> {
                override val dtype: DType get() = DType.F32

                override fun zero(): Float = 0.0f

                override fun one(): Float = 1.0f

                override fun createArray(len: Int): List<Float> = FloatArray(len).toList()

                override fun toCpuStorage(data: List<Float>): CpuStorage = CpuStorage.F32Storage(data.toFloatArray())

                override fun fromCpuStorage(storage: CpuStorage): List<Float> =
                    (storage as? CpuStorage.F32Storage)?.data?.toList() ?: throw UgError.Msg("expected F32Storage")
            }

        public val I32: WithDType<Int> =
            object : WithDType<Int> {
                override val dtype: DType get() = DType.I32

                override fun zero(): Int = 0

                override fun one(): Int = 1

                override fun createArray(len: Int): List<Int> = IntArray(len).toList()

                override fun toCpuStorage(data: List<Int>): CpuStorage = CpuStorage.I32Storage(data.toIntArray())

                override fun fromCpuStorage(storage: CpuStorage): List<Int> =
                    (storage as? CpuStorage.I32Storage)?.data?.toList() ?: throw UgError.Msg("expected I32Storage")
            }

        public val I64: WithDType<Long> =
            object : WithDType<Long> {
                override val dtype: DType get() = DType.I64

                override fun zero(): Long = 0L

                override fun one(): Long = 1L

                override fun createArray(len: Int): List<Long> = LongArray(len).toList()

                override fun toCpuStorage(data: List<Long>): CpuStorage = CpuStorage.I64Storage(data.toLongArray())

                override fun fromCpuStorage(storage: CpuStorage): List<Long> =
                    (storage as? CpuStorage.I64Storage)?.data?.toList() ?: throw UgError.Msg("expected I64Storage")
            }

        public val F16: WithDType<HalfF16> =
            object : WithDType<HalfF16> {
                override val dtype: DType get() = DType.F16

                override fun zero(): HalfF16 = HalfF16()

                override fun one(): HalfF16 = HalfF16.fromF32(1.0f)

                override fun createArray(len: Int): List<HalfF16> = Array(len) { HalfF16() }.toList()

                override fun toCpuStorage(data: List<HalfF16>): CpuStorage = CpuStorage.F16Storage(data.toTypedArray())

                override fun fromCpuStorage(storage: CpuStorage): List<HalfF16> =
                    (storage as? CpuStorage.F16Storage)?.data?.toList() ?: throw UgError.Msg("expected F16Storage")
            }

        public val BF16: WithDType<HalfBF16> =
            object : WithDType<HalfBF16> {
                override val dtype: DType get() = DType.BF16

                override fun zero(): HalfBF16 = HalfBF16()

                override fun one(): HalfBF16 = HalfBF16.fromF32(1.0f)

                override fun createArray(len: Int): List<HalfBF16> = Array(len) { HalfBF16() }.toList()

                override fun toCpuStorage(data: List<HalfBF16>): CpuStorage = CpuStorage.BF16Storage(data.toTypedArray())

                override fun fromCpuStorage(storage: CpuStorage): List<HalfBF16> =
                    (storage as? CpuStorage.BF16Storage)?.data?.toList() ?: throw UgError.Msg("expected BF16Storage")
            }
    }
}
