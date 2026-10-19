// port-lint: tests tests/display_tests.rs
package io.github.kotlinmania.ug

import io.github.kotlinmania.ug.cpuruntime.CpuDevice
import kotlin.test.Test
import kotlin.test.assertEquals

class DisplayTests {
    private fun mulCst(lb: LazyBuffer, v: Float): LazyBuffer {
        val t = LazyBuffer.cst(v, lb.shape, lb.device)
        return lb.binary(io.github.kotlinmania.ug.lang.BinaryOp.Mul, t)
    }

    @Test
    fun displayScalar() {
        val dev = CpuDevice
        val t = LazyBuffer.copy(intArrayOf(1234), Shape(), dev)
        val s = t.toString()
        assertEquals("[1234]\nTensor[[], i32]", s)

        val tF32 = t.cast(DType.F32).unary(io.github.kotlinmania.ug.lang.UnaryOp.Neg)
        val s1 = mulCst(tF32, 0.1f).toString()
        assertEquals("[-123.4000]\nTensor[[], f32]", s1)

        val s2 = mulCst(tF32, 1e-8f).toString()
        assertEquals("[-1.2340e-5]\nTensor[[], f32]", s2)

        val s3 = mulCst(tF32, 1e8f).toString()
        assertEquals("[-1.2340e11]\nTensor[[], f32]", s3)

        val s4 = mulCst(tF32, 0.0f).toString()
        assertEquals("[0.]\nTensor[[], f32]", s4)
    }

    @Test
    fun displayVector() {
        val dev = CpuDevice
        val t = LazyBuffer.copy(intArrayOf(), Shape(0), dev)
        val s = t.toString()
        assertEquals("[]\nTensor[[0], i32]", s)

        val t2 = LazyBuffer.copy(floatArrayOf(0.1234567f, 1.0f, -1.2f, 4.1f, Float.NaN), Shape(5), dev)
        val s2 = t2.toString()
        assertEquals("[ 0.1235,  1.0000, -1.2000,  4.1000,     NaN]\nTensor[[5], f32]", s2)

        val t3 = LazyBuffer.cst(42.0f, Shape(50), dev)
        val s3 = "\n" + t3.toString()
        val expected =
            """
[42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.,
 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.,
 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.,
 42., 42.]
Tensor[[50], f32]
            """.trimIndent()
        assertEquals("\n" + expected, s3)

        val t4 = LazyBuffer.cst(42.0f, Shape(11000), dev)
        val s4 = t4.toString()
        assertEquals("[42., 42., 42., ..., 42., 42., 42.]\nTensor[[11000], f32]", s4)
    }

    @Test
    fun displayMultiDim() {
        val t = LazyBuffer.cst(42.0f, Shape(200, 100), CpuDevice)
        val s = "\n" + t.toString()
        val expected =
            """
[[42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.],
 ...
 [42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.]]
Tensor[[200, 100], f32]
            """.trimIndent()
        assertEquals("\n" + expected, s)

        val tReshaped = t.reshape(Shape(2, 1, 1, 100, 100))
        val sReshaped = "\n" + tReshaped.toString()
        val expectedReshaped =
            """
[[[[[42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    ...
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.]]]],
 [[[[42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    ...
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.]]]]]
Tensor[[2, 1, 1, 100, 100], f32]
            """.trimIndent()
        assertEquals("\n" + expectedReshaped, sReshaped)
    }
}
