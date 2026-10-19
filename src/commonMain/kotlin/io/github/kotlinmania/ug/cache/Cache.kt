// port-lint: source cache.rs
package io.github.kotlinmania.ug.cache

import io.github.kotlinmania.ug.UgError
import io.github.kotlinmania.ug.lang.Arg
import io.github.kotlinmania.ug.lang.ArgId
import io.github.kotlinmania.ug.lang.op.Ast
import io.github.kotlinmania.ug.lang.op.AstInner
import io.github.kotlinmania.ug.lang.op.Kernel
import io.github.kotlinmania.ug.lang.op.Store
import io.github.kotlinmania.ug.lang.op.binary
import io.github.kotlinmania.ug.lang.op.cst
import io.github.kotlinmania.ug.lang.op.load
import io.github.kotlinmania.ug.lang.op.reduce
import io.github.kotlinmania.ug.lang.op.store
import io.github.kotlinmania.ug.lang.op.unary
import io.github.kotlinmania.ug.lang.ssa.Instr

public data class NormalizedKernel(
    public val args: List<Arg>,
    public val ops: List<Store>,
) {
    public companion object {
        public fun new(k: Kernel): NormalizedKernel {
            fun walk(ast: Ast, argMap: Map<ArgId, ArgId>): Ast =
                when (val in_ = ast.inner) {
                    is AstInner.Id -> {
                        UgError.Msg("unexpected id node").raise()
                    }

                    is AstInner.Load -> {
                        val src = argMap[in_.src] ?: UgError.Msg("BUG: missing arg id ${in_.src}").raise()
                        load(src, in_.layout, ast.dtype)
                    }

                    is AstInner.Unary -> {
                        val arg = walk(in_.arg, argMap)
                        unary(in_.op, arg)
                    }

                    is AstInner.Binary -> {
                        val lhs = walk(in_.lhs, argMap)
                        val rhs = walk(in_.rhs, argMap)
                        binary(in_.op, lhs, rhs)
                    }

                    is AstInner.Constant -> {
                        cst(in_.value)
                    }

                    is AstInner.Reduce -> {
                        val arg = walk(in_.arg, argMap)
                        reduce(in_.op, arg, in_.dim)
                    }

                    is AstInner.LayoutOperation -> {
                        val arg = walk(in_.arg, argMap)
                        Ast(AstInner.LayoutOperation(in_.op, arg), ast.dtype, ast.shape)
                    }
                }

            val argMap = mutableMapOf<ArgId, ArgId>()
            val args = ArrayList<Arg>(k.args.size)
            val ops = ArrayList<Store>(k.ops.size)
            for ((id, arg) in k.args.withIndex()) {
                val newId = ArgId.fromInt(id)
                argMap[arg.id] = newId
                args.add(Arg.new(newId, arg.type))
            }
            for (op in k.ops) {
                val dst = argMap[op.dst] ?: UgError.Msg("BUG: missing arg id ${op.dst}").raise()
                val value = walk(op.value, argMap)
                ops.add(store(dst, op.layout, value))
            }
            return NormalizedKernel(args, ops)
        }
    }
}

public class CompilationCache {
    private val opCache: MutableMap<NormalizedKernel, Any> = mutableMapOf()
    private val ssaCache: MutableMap<List<Instr>, Any> = mutableMapOf()

    public fun get(kernel: NormalizedKernel): Any? = opCache[kernel]

    public fun insert(kernel: NormalizedKernel, func: Any) {
        opCache[kernel] = func
    }

    public fun getSsa(kernel: List<Instr>): Any? = ssaCache[kernel]

    public fun insertSsa(kernel: List<Instr>, func: Any) {
        ssaCache[kernel] = func
    }
}
