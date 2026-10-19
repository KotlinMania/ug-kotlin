package io.github.kotlinmania.ug

public interface Device {
    public fun allocateUninit(dtype: DType, len: Int): Slice

    public fun synchronize() {}

    public fun compile(kernel: io.github.kotlinmania.ug.lang.ssa.Kernel, name: String? = null): Any

    public fun run(f: Any, args: List<Slice>)

    public fun matmul(
        dst: Slice,
        lhs: Slice,
        rhs: Slice,
        b: Int,
        m: Int,
        n: Int,
        k: Int,
        lhsL: Layout,
        rhsL: Layout,
    )

    public fun matmul(
        dst: Slice,
        lhs: Slice,
        rhs: Slice,
        bmnk: List<Int>,
        lhsL: Layout,
        rhsL: Layout,
    ) {
        matmul(dst, lhs, rhs, bmnk[0], bmnk[1], bmnk[2], bmnk[3], lhsL, rhsL)
    }

    public fun useGrid(): Boolean = false
}
