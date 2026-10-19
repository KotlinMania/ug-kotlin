// port-lint: source common_tests.rs
package io.github.kotlinmania.ug

public object CommonTests {
    public object lb {
        public fun add(dev: Device) {
            val lhs = LazyBuffer.cst(40.0f, Pair(5, 2), dev)
            val rhs = LazyBuffer.cst(2.0f, Pair(5, 2), dev)
            val lb = lhs.binary(io.github.kotlinmania.ug.lang.BinaryOp.Add, rhs)
            val data = lb.dataVec<Float>()
            val expected = List(10) { 42.0f }
            if (data != expected) {
                ugBail("add mismatch: expected $expected, got $data")
            }
        }

        public fun mm(dev: Device) {
            val lhs = LazyBuffer.cst(1.0f, Pair(4, 2), dev)
            val rhs = LazyBuffer.cst(2.0f, Pair(2, 3), dev)
            val lb = lhs.matmul(rhs)
            val data = lb.dataVec<Float>()
            val expected = List(12) { 4.0f }
            if (data != expected) {
                ugBail("mm mismatch: expected $expected, got $data")
            }
        }

        public fun cat(dev: Device) {
            val arg0 = LazyBuffer.copy(floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f), Pair(2, 2), dev)
            val arg1 = LazyBuffer.copy(floatArrayOf(1.1f, 2.1f, 3.1f, 4.1f), Pair(2, 2), dev)
            val arg2 = LazyBuffer.copy(floatArrayOf(1.2f, 2.2f, 3.2f, 4.2f), Pair(2, 2), dev)
            val lb0 = LazyBuffer.cat(listOf(arg0, arg1, arg2), 0)
            val data0 = lb0.dataVec<Float>()
            val expected0 = listOf(1.0f, 2.0f, 3.0f, 4.0f, 1.1f, 2.1f, 3.1f, 4.1f, 1.2f, 2.2f, 3.2f, 4.2f)
            if (data0.size != expected0.size) {
                ugBail("cat 0 len mismatch: expected ${expected0.size}, got ${data0.size}")
            }
            for (i in expected0.indices) {
                if (kotlin.math.abs(data0[i] - expected0[i]) > 1e-5f) {
                    ugBail("cat 0 mismatch at $i: expected ${expected0[i]}, got ${data0[i]}")
                }
            }

            val lb1 = LazyBuffer.cat(listOf(arg0, arg1, arg2), 1)
            val data1 = lb1.dataVec<Float>()
            val expected1 = listOf(1.0f, 2.0f, 1.1f, 2.1f, 1.2f, 2.2f, 3.2f, 4.2f, 0.0f, 0.0f, 0.0f, 0.0f)
            if (data1.size != expected1.size) {
                ugBail("cat 1 len mismatch: expected ${expected1.size}, got ${data1.size}")
            }
            for (i in expected1.indices) {
                if (kotlin.math.abs(data1[i] - expected1[i]) > 1e-5f) {
                    ugBail("cat 1 mismatch at $i: expected ${expected1[i]}, got ${data1[i]}")
                }
            }
        }
    }
}
