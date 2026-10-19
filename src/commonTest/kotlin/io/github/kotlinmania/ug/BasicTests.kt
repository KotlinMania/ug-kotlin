// port-lint: tests tests/basic_tests.rs
package io.github.kotlinmania.ug

import io.github.kotlinmania.ug.interpreter.Buffer
import io.github.kotlinmania.ug.interpreter.evalSsa
import io.github.kotlinmania.ug.lower.lower
import io.github.kotlinmania.ug.lowerop.lower
import kotlin.test.Test
import kotlin.test.assertEquals

class BasicTests {
    @Test
    fun evalAdd() {
        val kernel = Samples.ssa.simpleAdd(2)
        val a = Buffer.I32(intArrayOf(0, 0))
        val b = Buffer.I32(intArrayOf(3, 4))
        val c = Buffer.I32(intArrayOf(1, 2))
        evalSsa(kernel, listOf(a, b, c), emptyList(), 0)
        val aResult = a.asI32().toList()
        assertEquals(listOf(4, 6), aResult)
    }

    @Test
    fun evalDotprod() {
        val kernel = Samples.ssa.simpleDotprod(2)
        val a = Buffer.F32(floatArrayOf(0.0f))
        val b = Buffer.F32(floatArrayOf(3.0f, 4.5f))
        val c = Buffer.F32(floatArrayOf(1.0f, 2.0f))
        evalSsa(kernel, listOf(a, b, c), emptyList(), 0)
        val aResult = a.asF32().toList()
        assertEquals(listOf(12.0f), aResult)
    }

    @Test
    fun lowerAdd() {
        val kernel = Samples.simpleAdd(2)
        val ssaKernel = kernel.lower()
        val c = Buffer.F32(floatArrayOf(0.0f, 0.0f))
        val b = Buffer.F32(floatArrayOf(3.0f, 4.0f))
        val a = Buffer.F32(floatArrayOf(1.0f, 2.0f))
        evalSsa(ssaKernel, listOf(a, b, c), emptyList(), 0)
        val cResult = c.asF32().toList()
        assertEquals(listOf(4.0f, 6.0f), cResult)
    }

    @Test
    fun softmax() {
        val kernel = Samples.op.softmax(2, 4)
        val ssaKernel = kernel.lower()
        val a = Buffer.F32(floatArrayOf(0f, 1f, 2f, 3f, 2f, 1f, 2f, 1f))
        val b = Buffer.F32(FloatArray(8))
        evalSsa(ssaKernel, listOf(a, b), emptyList(), 0)
        val bResult = b.asF32().toList()
        val expected =
            listOf(
                0.032058604f,
                0.08714432f,
                0.23688284f,
                0.6439143f,
                0.3655293f,
                0.13447072f,
                0.3655293f,
                0.13447072f,
            )
        for (i in expected.indices) {
            assertEquals(expected[i], bResult[i], 1e-5f)
        }
    }

    @Test
    fun ssaSoftmax() {
        val nCols = 4
        val nRows = 2
        val ssaKernel = Samples.ssa.softmax(nRows, nCols)
        val a = Buffer.F32(floatArrayOf(0f, 1f, 2f, 3f, 2f, 1f, 2f, 1f))
        val b = Buffer.F32(FloatArray(nRows * nCols))
        for (groupIdx in 0 until nRows) {
            evalSsa(ssaKernel, listOf(a, b), emptyList(), groupIdx, warpSize = nCols)
        }
        val bResult = b.asF32().toList()
        val expected =
            listOf(
                0.032058604f,
                0.08714432f,
                0.23688284f,
                0.6439143f,
                0.3655293f,
                0.13447072f,
                0.3655293f,
                0.13447072f,
            )
        for (i in expected.indices) {
            assertEquals(expected[i], bResult[i], 1e-5f)
        }
    }

    @Test
    fun ssaSoftmaxReduce() {
        val nCols = 4
        val nRows = 2
        val ssaKernel = Samples.ssa.softmaxReduce(nRows, nCols)
        val a = Buffer.F32(floatArrayOf(0f, 1f, 2f, 3f, 2f, 1f, 2f, 1f))
        val b = Buffer.F32(FloatArray(nRows * nCols))
        for (groupIdx in 0 until nRows) {
            evalSsa(ssaKernel, listOf(a, b), emptyList(), groupIdx, warpSize = nCols)
        }
        val bResult = b.asF32().toList()
        val expected =
            listOf(
                0.032058604f,
                0.08714432f,
                0.23688284f,
                0.6439143f,
                0.3655293f,
                0.13447072f,
                0.3655293f,
                0.13447072f,
            )
        for (i in expected.indices) {
            assertEquals(expected[i], bResult[i], 1e-5f)
        }
    }

    @Test
    fun affine() {
        val layout = Layout.fromShape(4)
        val ptr = io.github.kotlinmania.ug.lang.Arg.ptr(DType.F32)
        val src = io.github.kotlinmania.ug.lang.op.load(ptr.id, layout, DType.F32)
        val sub = io.github.kotlinmania.ug.lang.op.broadcastBinary(io.github.kotlinmania.ug.lang.BinaryOp.Sub, src, io.github.kotlinmania.ug.lang.op.cst(0.5f))
        val mul = io.github.kotlinmania.ug.lang.op.broadcastBinary(io.github.kotlinmania.ug.lang.BinaryOp.Mul, sub, io.github.kotlinmania.ug.lang.op.cst(3.0f))
        val st = io.github.kotlinmania.ug.lang.op.store(ptr.id, layout, mul)
        val kernel = io.github.kotlinmania.ug.lang.op.Kernel("affine", listOf(ptr), listOf(st))
        val ssaKernel = kernel.lower()
        val a = Buffer.F32(floatArrayOf(1f, 2f, 3f, 4f))
        evalSsa(ssaKernel, listOf(a), emptyList(), 0, warpSize = 4)
        val aResult = a.asF32().toList()
        assertEquals(listOf(1.5f, 4.5f, 7.5f, 10.5f), aResult)
    }
}
