// port-lint: source schedule.rs
package io.github.kotlinmania.ug

import io.github.kotlinmania.ug.cache.CompilationCache
import io.github.kotlinmania.ug.cache.NormalizedKernel
import io.github.kotlinmania.ug.lang.ArgId
import io.github.kotlinmania.ug.lang.Type
import io.github.kotlinmania.ug.lang.op.Ast
import io.github.kotlinmania.ug.lang.op.AstInner
import io.github.kotlinmania.ug.lower_op.Opts

public class KernelItem(
    public val ast: Ast,
    public val dst: Pair<ArgId, LazyBuffer>,
    public val dstLayout: Layout?,
    public val args: List<Pair<ArgId, LazyBuffer>>,
) {
    public fun ast(): Ast = ast

    public fun kernel(): io.github.kotlinmania.ug.lang.op.Kernel {
        val kernelArgs =
            args.map { (id, lb) ->
                io.github.kotlinmania.ug.lang.op.Arg(id, Type.Ptr(lb.dtype))
            }
        val layout = dstLayout ?: Layout.fromShape(dst.second.shape)
        val sto = io.github.kotlinmania.ug.lang.op.store(dst.first, layout, ast)
        return io.github.kotlinmania.ug.lang.op.Kernel(
            name = "realize_${dst.second.id.id}",
            args = kernelArgs,
            ops = listOf(sto),
        )
    }
}

public sealed class ScheduleItem {
    public data class Kernel(
        public val item: KernelItem,
    ) : ScheduleItem()

    public data class MatMul(
        public val dst: LazyBuffer,
        public val lhs: LazyBuffer,
        public val rhs: LazyBuffer,
        public val bmnk: List<Int>,
        public val transpose: Boolean,
    ) : ScheduleItem()

    public data class Custom(
        public val f: CustomF,
        public val args: List<Pair<ArgId, LazyBuffer>>,
    ) : ScheduleItem()

    public data class Ssa(
        public val ssa: io.github.kotlinmania.ug.lang.ssa.Kernel,
        public val args: List<Pair<ArgId, LazyBuffer>>,
    ) : ScheduleItem()
}

public sealed class Func {
    public data class Kernel(
        public val func: Any,
        public val args: List<Pair<ArgId, LazyBuffer>>,
    ) : Func()

    public data class MatMul(
        public val dst: LazyBuffer,
        public val lhs: LazyBuffer,
        public val rhs: LazyBuffer,
        public val bmnk: List<Int>,
        public val transpose: Boolean,
    ) : Func()

    public data class Custom(
        public val f: CustomF,
        public val args: List<Pair<ArgId, LazyBuffer>>,
    ) : Func()
}

public class CompiledSchedule(
    public val funcs: List<Func>,
    public val device: Device,
) {
    public fun run() {
        for (func in funcs) {
            when (func) {
                is Func.Kernel -> {
                    val slices =
                        func.args.map { (_, lb) ->
                            lb.maybeAllocateUninit()
                            lb.data ?: throw UgError.Msg("no data for arg in kernel run")
                        }
                    device.run(func.func, slices)
                }

                is Func.MatMul -> {
                    val lhs = func.lhs
                    val rhs = func.rhs
                    val dst = func.dst
                    val lhsDims = lhs.dims()
                    val lhsRank = lhs.rank()
                    val rhsDims = rhs.dims()
                    val rhsRank = rhs.rank()

                    val lhsL =
                        if (lhsRank < rhsRank) {
                            val prefix = IntArray(rhsRank - lhsRank) { 1 }
                            Layout.fromShape(Shape(prefix + lhsDims))
                        } else {
                            Layout.fromShape(lhs.shape)
                        }

                    val rhsL0 =
                        if (rhsRank < lhsRank) {
                            val prefix = IntArray(lhsRank - rhsRank) { 1 }
                            Layout.fromShape(Shape(prefix + rhsDims))
                        } else {
                            Layout.fromShape(rhs.shape)
                        }
                    val rhsL = if (func.transpose) rhsL0.transpose() else rhsL0

                    dst.maybeAllocateUninit()
                    lhs.maybeAllocateUninit()
                    rhs.maybeAllocateUninit()

                    val dstSlice = dst.data ?: throw UgError.Msg("no dst data")
                    val lhsSlice = lhs.data ?: throw UgError.Msg("no lhs data")
                    val rhsSlice = rhs.data ?: throw UgError.Msg("no rhs data")
                    val bmnk = func.bmnk
                    device.matmul(
                        dst = dstSlice,
                        lhs = lhsSlice,
                        rhs = rhsSlice,
                        b = bmnk[0],
                        m = bmnk[1],
                        n = bmnk[2],
                        k = bmnk[3],
                        lhsL = lhsL,
                        rhsL = rhsL,
                    )
                }

                is Func.Custom -> {
                    val slices =
                        func.args.map { (_, lb) ->
                            lb.maybeAllocateUninit()
                            lb.data ?: throw UgError.Msg("no data for custom run")
                        }
                    func.f(slices)
                }
            }
        }
    }
}

public class Schedule(
    public val items: List<ScheduleItem>,
    public val perArgId: Map<ArgId, LazyBuffer>,
    public val device: Device,
) {
    public fun getArgId(argId: ArgId): LazyBuffer =
        perArgId[argId] ?: throw UgError.Msg("no arg for id $argId")

    public fun items(): List<ScheduleItem> = items

    public fun compile(compilationCache: CompilationCache = CompilationCache()): CompiledSchedule {
        val funcs = mutableListOf<Func>()
        for (item in items) {
            val call =
                when (item) {
                    is ScheduleItem.MatMul ->
                        Func.MatMul(
                            dst = item.dst,
                            lhs = item.lhs,
                            rhs = item.rhs,
                            bmnk = item.bmnk,
                            transpose = item.transpose,
                        )

                    is ScheduleItem.Custom ->
                        Func.Custom(
                            f = item.f,
                            args = item.args,
                        )

                    is ScheduleItem.Ssa -> {
                        val ssa = item.ssa
                        val cached = compilationCache.getSsa(ssa.instrs)
                        if (cached != null) {
                            Func.Kernel(func = cached, args = item.args)
                        } else {
                            val func = device.compile(ssa, null)
                            compilationCache.insertSsa(ssa.instrs, func)
                            Func.Kernel(func = func, args = item.args)
                        }
                    }

                    is ScheduleItem.Kernel -> {
                        val kernelItem = item.item
                        val kernel = kernelItem.kernel()
                        val normKernel = NormalizedKernel.new(kernel)
                        val cached = compilationCache.get(normKernel)
                        if (cached != null) {
                            val args =
                                kernel.args.map { arg ->
                                    val argId = arg.id
                                    val lb = getArgId(argId)
                                    Pair(argId, lb)
                                }
                            Func.Kernel(func = cached, args = args)
                        } else {
                            val kernelName = if (kernel.ops.isEmpty()) null else kernel.ops[0].name()
                            val optKernel = kernel.optimize()
                            val opts =
                                if (device.useGrid() &&
                                    optKernel.ops.size == 1 &&
                                    optKernel.ops[0].layout.rank >= 1 &&
                                    optKernel.ops[0].layout.numElements >= 1
                                ) {
                                    val dims =
                                        optKernel.ops[0]
                                            .layout
                                            .dims
                                            .withIndex()
                                            .map { Pair(it.index, it.value) }
                                            .sortedByDescending { it.second }
                                    if (dims.size >= 2 && dims[1].second > 1) {
                                        Opts.default()
                                            .withBlockAxis(dims[0].first)
                                            .withThreadBlock(dims[1].first, 32)
                                    } else if (dims.size > 1 && dims[0].second > 1) {
                                        Opts.default().withGlobalAxis(dims[0].first, 32)
                                    } else {
                                        Opts.default()
                                    }
                                } else {
                                    Opts.default()
                                }
                            val ssa = optKernel.lower(opts)
                            val args =
                                ssa.args.map { (arg, _) ->
                                    val argId = arg.id
                                    val lb = getArgId(argId)
                                    Pair(argId, lb)
                                }
                            val func = device.compile(ssa, kernelName)
                            compilationCache.insert(normKernel, func)
                            Func.Kernel(func = func, args = args)
                        }
                    }
                }
            funcs.add(call)
        }
        return CompiledSchedule(funcs, device)
    }

    public companion object {
        public fun create(buffers: List<LazyBuffer>): Schedule {
            if (buffers.isEmpty()) {
                ugBail("no buffers provided")
            }
            val device = buffers[0].device
            val cnts = mutableMapOf<LazyBufferId, Int>()
            for (buffer in buffers) {
                idCnts(buffer, cnts)
            }
            val context = Context(cnts)
            for (buffer in buffers) {
                context.pushScheduleItem(buffer)
            }
            return Schedule(
                items = context.items,
                perArgId = context.perArgId,
                device = device,
            )
        }

        public fun createOne(buffer: LazyBuffer): Schedule = create(listOf(buffer))
    }
}

private class Context(
    val idCnts: Map<LazyBufferId, Int>,
) {
    val items = mutableListOf<ScheduleItem>()
    val perArgId = mutableMapOf<ArgId, LazyBuffer>()
    val astCache = mutableMapOf<LazyBufferId, Ast>()

    fun getArgId(argId: ArgId): LazyBuffer =
        perArgId[argId] ?: throw UgError.Msg("no arg for id $argId")

    fun walk(b: LazyBuffer): Ast {
        val id = b.id
        astCache[id]?.let { return it }

        val dtype = b.dtype
        val shape = b.shape
        val ast =
            if (b.realized()) {
                val argId = ArgId.new()
                perArgId[argId] = b
                io.github.kotlinmania.ug.lang.op.load(argId, Layout.fromShape(shape), dtype)
            } else {
                when (val op = b.op) {
                    is Op.Unary -> {
                        val innerAst = walk(op.arg)
                        io.github.kotlinmania.ug.lang.op.unary(op.op, innerAst)
                    }

                    is Op.Binary -> {
                        val lhs = walk(op.lhs)
                        val rhs = walk(op.rhs)
                        io.github.kotlinmania.ug.lang.op.binary(op.op, lhs, rhs)
                    }

                    is Op.MatMul -> {
                        pushScheduleItem(op.lhs)
                        pushScheduleItem(op.rhs)
                        val dstId = ArgId.new()
                        perArgId[dstId] = b
                        items.add(
                            ScheduleItem.MatMul(
                                dst = b,
                                lhs = op.lhs,
                                rhs = op.rhs,
                                bmnk = op.bmnk,
                                transpose = op.transpose,
                            ),
                        )
                        io.github.kotlinmania.ug.lang.op.load(dstId, Layout.fromShape(shape), dtype)
                    }

                    is Op.Reduce -> {
                        val innerAst = walk(op.arg)
                        io.github.kotlinmania.ug.lang.op.reduce(op.op, innerAst, op.axis)
                    }

                    is Op.Const -> io.github.kotlinmania.ug.lang.op.cst(op.c)

                    is Op.Value -> {
                        val argId = ArgId.new()
                        perArgId[argId] = b
                        io.github.kotlinmania.ug.lang.op.load(argId, Layout.fromShape(shape), dtype)
                    }

                    is Op.Reshape -> {
                        val dstId = pushScheduleItem(op.arg)
                        io.github.kotlinmania.ug.lang.op.load(dstId, Layout.fromShape(shape), dtype)
                    }

                    is Op.Layout -> {
                        val innerAst = walk(op.arg)
                        io.github.kotlinmania.ug.lang.op.layout(op.op, innerAst, shape)
                    }

                    is Op.Ssa -> {
                        val args = mutableListOf<Pair<ArgId, LazyBuffer>>()
                        for (arg in op.args) {
                            val argId = pushScheduleItem(arg)
                            args.add(Pair(argId, arg))
                        }
                        val dstId = ArgId.new()
                        perArgId[dstId] = b
                        args.add(Pair(dstId, b))
                        items.add(ScheduleItem.Ssa(ssa = op.ssa, args = args))
                        io.github.kotlinmania.ug.lang.op.load(dstId, Layout.fromShape(shape), dtype)
                    }

                    is Op.Set -> {
                        val argId = pushScheduleItem(op.src)
                        val values = walk(op.values)
                        pushKernel(op.src, values, op.dstLayout)
                        io.github.kotlinmania.ug.lang.op.load(argId, Layout.fromShape(shape), dtype)
                    }

                    is Op.CustomIp -> {
                        val args = mutableListOf<Pair<ArgId, LazyBuffer>>()
                        for (arg in op.args) {
                            val argId = pushScheduleItem(arg)
                            args.add(Pair(argId, arg))
                        }
                        val argId = pushScheduleItem(op.src)
                        args.add(Pair(argId, op.src))
                        items.add(ScheduleItem.Custom(f = op.f, args = args))
                        io.github.kotlinmania.ug.lang.op.load(argId, Layout.fromShape(shape), dtype)
                    }

                    is Op.Custom -> {
                        val args = mutableListOf<Pair<ArgId, LazyBuffer>>()
                        for (arg in op.args) {
                            val argId = pushScheduleItem(arg)
                            args.add(Pair(argId, arg))
                        }
                        val dstId = ArgId.new()
                        perArgId[dstId] = b
                        args.add(Pair(dstId, b))
                        items.add(ScheduleItem.Custom(f = op.f, args = args))
                        io.github.kotlinmania.ug.lang.op.load(dstId, Layout.fromShape(shape), dtype)
                    }
                }
            }

        val finalAst =
            if ((idCnts[id] ?: 0) > 1) {
                val dstId = pushKernel(b, ast, null)
                io.github.kotlinmania.ug.lang.op.load(dstId, Layout.fromShape(shape), dtype)
            } else {
                ast
            }
        astCache[id] = finalAst
        return finalAst
    }

    fun pushKernel(
        buffer: LazyBuffer,
        ast: Ast,
        dstLayout: Layout?,
    ): ArgId {
        val inner = ast.inner
        if (inner is AstInner.Load) {
            val src = getArgId(inner.src)
            if (src.data === buffer.data && src.data != null) {
                return inner.src
            }
        }

        val dstId = ArgId.new()
        perArgId[dstId] = buffer
        val argIds = ast.argIds().toMutableSet()
        argIds.add(dstId)
        val args =
            argIds.map { argId ->
                val arg = getArgId(argId)
                Pair(argId, arg)
            }
        val si = KernelItem(ast = ast, dst = Pair(dstId, buffer), dstLayout = dstLayout, args = args)
        items.add(ScheduleItem.Kernel(si))
        return dstId
    }

    fun pushScheduleItem(buffer: LazyBuffer): ArgId {
        val ast = walk(buffer)
        return pushKernel(buffer, ast, null)
    }
}

private fun idCnts(
    b: LazyBuffer,
    cnts: MutableMap<LazyBufferId, Int>,
) {
    if (b.realized()) return

    val id = b.id
    val cnt = cnts.getOrPut(id) { 0 } + 1
    cnts[id] = cnt
    if (cnt > 1) return

    when (val op = b.op) {
        is Op.Value, is Op.Const -> {}
        is Op.Reshape -> idCnts(op.arg, cnts)
        is Op.Layout -> idCnts(op.arg, cnts)
        is Op.Reduce -> idCnts(op.arg, cnts)
        is Op.Unary -> idCnts(op.arg, cnts)
        is Op.Set -> {
            idCnts(op.src, cnts)
            idCnts(op.values, cnts)
        }

        is Op.MatMul -> {
            idCnts(op.lhs, cnts)
            idCnts(op.rhs, cnts)
        }

        is Op.Binary -> {
            idCnts(op.lhs, cnts)
            idCnts(op.rhs, cnts)
        }

        is Op.CustomIp -> {
            for (arg in op.args) {
                idCnts(arg, cnts)
            }
            idCnts(op.src, cnts)
        }

        is Op.Ssa -> {
            for (arg in op.args) {
                idCnts(arg, cnts)
            }
        }

        is Op.Custom -> {
            for (arg in op.args) {
                idCnts(arg, cnts)
            }
        }
    }
}
