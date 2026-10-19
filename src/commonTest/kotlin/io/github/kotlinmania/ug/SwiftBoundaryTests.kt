package io.github.kotlinmania.ug

import io.github.kotlinmania.half.BF16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.ug.cpuruntime.CpuStorage
import io.github.kotlinmania.ug.cpuruntime.MatMul
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertSame

class SwiftBoundaryTests {
    @Test
    fun concreteDTypesRoundTripStorage() {
        assertEquals(listOf(1.5f, -2.0f), WithDType.F32.fromCpuStorage(WithDType.F32.toCpuStorage(listOf(1.5f, -2.0f))))
        assertEquals(listOf(1, -2), WithDType.I32.fromCpuStorage(WithDType.I32.toCpuStorage(listOf(1, -2))))
        assertEquals(listOf(1L, -2L), WithDType.I64.fromCpuStorage(WithDType.I64.toCpuStorage(listOf(1L, -2L))))
        val f16 = listOf(F16.fromF32(1.5f), F16.fromF32(-2.0f))
        val bf16 = listOf(BF16.fromF32(1.5f), BF16.fromF32(-2.0f))
        assertEquals(f16.map { it.toF32() }, WithDType.F16.fromCpuStorage(WithDType.F16.toCpuStorage(f16)).map { it.toF32() })
        assertEquals(bf16.map { it.toF32() }, WithDType.BF16.fromCpuStorage(WithDType.BF16.toCpuStorage(bf16)).map { it.toF32() })
        val failure = assertFailsWith<UgException> { WithDType.F32.fromCpuStorage(WithDType.I32.toCpuStorage(listOf(1))) }
        assertIs<UgError.Msg>(failure.error)
        assertEquals("expected F32Storage", failure.message)
    }

    @Test
    fun halfStorageViewsPreserveMutationsAndCopies() {
        val f16 = assertIs<CpuStorage.F16Storage>(WithDType.F16.toCpuStorage(listOf(F16.fromF32(1.0f))))
        val f16View = f16.data
        f16[0] = F16.fromF32(2.0f)
        assertEquals(2.0f, f16View[0].toF32())
        val f16Copy = arrayOf(F16())
        f16.copyDeviceToHost(f16Copy)
        f16.copyHostToDevice(arrayOf(F16.fromF32(3.0f)))
        assertEquals(2.0f, f16Copy[0].toF32())
        assertEquals(3.0f, f16[0].toF32())
        val bf16 = assertIs<CpuStorage.BF16Storage>(WithDType.BF16.toCpuStorage(listOf(BF16.fromF32(1.0f))))
        val bf16View = bf16.data
        bf16[0] = BF16.fromF32(2.0f)
        assertEquals(2.0f, bf16View[0].toF32())
        val bf16Copy = arrayOf(BF16())
        bf16.copyDeviceToHost(bf16Copy)
        bf16.copyHostToDevice(arrayOf(BF16.fromF32(3.0f)))
        assertEquals(2.0f, bf16Copy[0].toF32())
        assertEquals(3.0f, bf16[0].toF32())
    }

    @Test
    fun raisedErrorsRetainDomainValueAndCause() {
        val cause = UgError.Msg("invalid tensor")
        val error = cause.withPath("weights")
        val failure = assertFailsWith<UgException> { error.raise() }
        assertSame(error, UgError.fromException(failure))
        assertNull(UgError.fromException(IllegalArgumentException("unrelated")))
        assertEquals("path: weights invalid tensor", failure.message)
        assertSame(cause, assertIs<UgException>(failure.cause).error)
    }

    @Test
    fun stridingFailureRetainsItsVariant() {
        val lhs = Layout(Shape(2, 3, 4, 6), listOf(111, 19, 6, 1))
        val rhs = Layout(Shape(2, 3, 6, 5))
        val failure = assertFailsWith<UgException> { MatMul(1, 4, 5, 6).abSkip(lhs, rhs) }
        assertEquals("non-contiguous lhs", assertIs<UgError.MatMulStriding>(failure.error).info.msg)
    }

    @Test
    fun namedCallbackRunsOnceWithItsArguments() {
        var calls = 0
        val slices = emptyList<Slice>()
        val callback =
            CustomFCallback { args ->
                assertSame(slices, args)
                calls += 1
            }
        CustomF(callback)(slices)
        assertEquals(1, calls)
    }
}
