// port-lint: tests tests/cpu_tests.rs
package io.github.kotlinmania.ug

import io.github.kotlinmania.ug.cpu_code_gen.gen
import io.github.kotlinmania.ug.cpu_runtime.CpuDevice
import kotlin.test.Test
import kotlin.test.assertEquals

class CpuTests {
    @Test
    fun runAdd() {
        val kernel = Samples.ssa.simpleAdd(6)
        val cCode = gen("foo", kernel)
        val func = CpuDevice.compileC(cCode, "foo")
        val v1 = IntArray(6)
        val v2 = intArrayOf(3, 1, 4, 1, 5, 9)
        val v3 = intArrayOf(1, 2, 3, 4, 5, 6)
        func.run3(v1, v2, v3)
        assertEquals(listOf(4, 3, 7, 5, 10, 15), v1.toList())
    }
}
