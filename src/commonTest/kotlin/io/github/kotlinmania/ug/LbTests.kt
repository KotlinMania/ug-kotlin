// port-lint: tests tests/lb_tests.rs
package io.github.kotlinmania.ug

import io.github.kotlinmania.ug.cpu_runtime.CpuDevice
import io.github.kotlinmania.ug.cpu_runtime.CpuStorage
import io.github.kotlinmania.ug.interpreter.Buffer
import io.github.kotlinmania.ug.interpreter.evalSsa
import io.github.kotlinmania.ug.lower_op.Opts
import kotlin.test.Test
import kotlin.test.assertEquals

class LbTests {
    @Test
    fun scheduleInterpret() {
        val cpu = CpuDevice
        val lhs = LazyBuffer.cst(40.0f, Pair(5, 2), cpu)
        val rhs = LazyBuffer.cst(2.0f, Pair(5, 2), cpu)
        val lb = lhs.binary(io.github.kotlinmania.ug.lang.BinaryOp.Add, rhs)
        val schedule = Schedule.createOne(lb)
        val items = schedule.items()
        assertEquals(1, items.size)
        for (item in items) {
            val kernel =
                when (item) {
                    is ScheduleItem.Kernel -> item.item.kernel()
                    else -> ugBail("unexpected item")
                }
            val ssa = kernel.lower(Opts.default())
            val buffer = Buffer.F32(FloatArray(10))
            evalSsa(ssa, listOf(buffer), emptyList(), 0)
            val res = buffer.asF32().toList()
            val expected = List(10) { 42.0f }
            assertEquals(expected, res)
        }
    }

    @Test
    fun scheduleCpu() {
        val cpu = CpuDevice
        val lhs = LazyBuffer.cst(40.0f, Pair(5, 2), cpu)
        val rhs = LazyBuffer.cst(2.0f, Pair(5, 2), cpu)
        val lb = lhs.binary(io.github.kotlinmania.ug.lang.BinaryOp.Add, rhs)
        val schedule = Schedule.createOne(lb)
        val items = schedule.items()
        assertEquals(1, items.size)
        val kernel =
            when (val item = items[0]) {
                is ScheduleItem.Kernel -> item.item.kernel()
                else -> ugBail("unexpected item")
            }
        val ssa = kernel.lower(Opts.default())
        val func = cpu.compile(ssa, null)
        val buffer = cpu.allocateUninit(DType.F32, 10)
        cpu.run(func, listOf(buffer))
        val res = buffer.toVec<Float>()
        val expected = List(10) { 42.0f }
        assertEquals(expected, res)
    }

    @Test
    fun scheduleCpuCompile() {
        CommonTests.lb.add(CpuDevice)
    }

    @Test
    fun scheduleMm() {
        CommonTests.lb.mm(CpuDevice)
    }

    @Test
    fun scheduleCat() {
        CommonTests.lb.cat(CpuDevice)
    }

    @Test
    fun lbCopy() {
        val cpu = CpuDevice
        val shape = Pair(2, 3)
        val buf = floatArrayOf(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
        val lhs = LazyBuffer.copy(buf, shape, cpu)
        val two = LazyBuffer.cst(2.0f, shape, cpu)
        val half = LazyBuffer.cst(0.5f, shape, cpu)
        val lb1 = lhs.binary(io.github.kotlinmania.ug.lang.BinaryOp.Add, two)
        val lb = lb1.binary(io.github.kotlinmania.ug.lang.BinaryOp.Mul, half)
        val schedule = Schedule.createOne(lb)
        val compiled = schedule.compile()
        compiled.run()
        val data = lb.dataVec<Float>()
        assertEquals(listOf(1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f), data)

        val storage = lhs.data as CpuStorage.F32Storage
        storage.data[1] = 0.0f
        storage.data[2] = -8.0f
        storage.data[5] = -2.0f

        compiled.run()
        val data2 = lb.dataVec<Float>()
        assertEquals(listOf(1.0f, 1.0f, -3.0f, 2.5f, 3.0f, 0.0f), data2)
    }

    @Test
    fun scheduleBroadcast() {
        val cpu = CpuDevice
        val lhs = LazyBuffer.cst(40.0f, Pair(5, 2), cpu)
        val rhs = LazyBuffer.cst(2.0f, Shape(), cpu)
        val rhsBc = rhs.broadcast(Shape(5, 2))
        val lb = lhs.binary(io.github.kotlinmania.ug.lang.BinaryOp.Add, rhsBc)
        val data = lb.dataVec<Float>()
        assertEquals(List(10) { 42.0f }, data)
    }

    @Test
    fun lbCustom() {
        val cpu = CpuDevice
        val addOne: (List<Slice>) -> Unit = { vs ->
            val src = vs[0] as CpuStorage.F32Storage
            val dst = vs[1] as CpuStorage.F32Storage
            for (i in dst.data.indices) {
                dst.data[i] = src.data[i] + 1.0f
            }
        }
        val shape = Shape(2, 3)
        val buf = floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f)
        val bufLb = LazyBuffer.copy(buf, shape, cpu)
        val twoLb = LazyBuffer.cst(2.0f, shape, cpu)
        val lbCustom = LazyBuffer.custom(addOne, listOf(bufLb), shape, DType.F32, cpu)
        val lb = lbCustom.binary(io.github.kotlinmania.ug.lang.BinaryOp.Mul, twoLb)

        val schedule = Schedule.createOne(lb)
        val compiled = schedule.compile()
        compiled.run()

        val data = lb.dataVec<Float>()
        assertEquals(listOf(2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f), data)

        val storage = bufLb.data as CpuStorage.F32Storage
        storage.data[1] = 0.0f
        storage.data[2] = -8.0f
        storage.data[5] = -2.0f

        compiled.run()
        val data2 = lb.dataVec<Float>()
        assertEquals(listOf(2.0f, 2.0f, -14.0f, 8.0f, 10.0f, -2.0f), data2)
    }

    @Test
    fun lbLayout() {
        val cpu = CpuDevice
        val shape = Shape(2, 3)
        val buf = floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f)
        val lb0 = LazyBuffer.copy(buf, shape, cpu)

        val lb1 = lb0.transpose(0.D, 1.D)
        val data1 = lb1.dataVec<Float>()
        assertEquals(listOf(0.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f), data1)

        val lb2 = lb1.transpose(0.D, 1.D)
        val lb3 = lb2.mergeDims(0.D)
        assertEquals(listOf(6), lb3.dims().toList())

        val data3 = lb3.dataVec<Float>()
        assertEquals(listOf(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f), data3)
    }

    @Test
    fun lbCustomInPlace() {
        val cpu = CpuDevice
        val addOne: (List<Slice>) -> Unit = { vs ->
            val dst = vs[0] as CpuStorage.F32Storage
            for (i in dst.data.indices) {
                dst.data[i] += 1.0f
            }
        }
        val shape = Shape(2, 3)
        val buf = floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f)
        var lb = LazyBuffer.copy(buf, shape, cpu)
        lb = lb.customIp(addOne, emptyList())
        lb = lb.customIp(addOne, emptyList())
        lb = lb.customIp(addOne, emptyList())
        val data = lb.dataVec<Float>()
        assertEquals(listOf(3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f), data)

        val storage = lb.data as CpuStorage.F32Storage
        storage.data[1] = 0.0f
        storage.data[2] = -8.0f
        storage.data[5] = -2.0f

        val data2 = lb.dataVec<Float>()
        assertEquals(listOf(6.0f, 3.0f, -5.0f, 9.0f, 10.0f, 1.0f), data2)
    }

    @Test
    fun lbSet() {
        val cpu = CpuDevice
        val shape = Shape(2, 3)
        val buf = floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f)
        var dstLb = LazyBuffer.copy(buf, shape, cpu)
        val srcLb = dstLb.binary(io.github.kotlinmania.ug.lang.BinaryOp.Mul, dstLb)
        val dataSrc = srcLb.dataVec<Float>()
        assertEquals(listOf(0.0f, 1.0f, 4.0f, 9.0f, 16.0f, 25.0f), dataSrc)
        val dataDst = dstLb.dataVec<Float>()
        assertEquals(listOf(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f), dataDst)

        dstLb = dstLb.set(srcLb)
        val dataSet = dstLb.dataVec<Float>()
        assertEquals(listOf(0.0f, 1.0f, 4.0f, 9.0f, 16.0f, 25.0f), dataSet)

        val buf3 = floatArrayOf(3.0f, 1.0f, 2.0f)
        val vs = LazyBuffer.copy(buf3, Shape(3), cpu)
        dstLb = dstLb.setL(vs, Layout.fromShape(3))
        val dataSetL = dstLb.dataVec<Float>()
        assertEquals(listOf(3.0f, 1.0f, 2.0f, 9.0f, 16.0f, 25.0f), dataSetL)
    }
}
