// port-lint: source error.rs
package io.github.kotlinmania.ug

import io.github.kotlinmania.thiserror.StdError

public data class MatMulUnexpectedStriding(
    public val lhsL: Layout,
    public val rhsL: Layout,
    public val bmnk: List<Int>,
    public val msg: String,
) {
    public constructor(lhsL: Layout, rhsL: Layout, b: Int, m: Int, n: Int, k: Int, msg: String) : this(
        lhsL,
        rhsL,
        listOf(b, m, n, k),
        msg,
    )
}

public sealed class UgError(
    message: String,
    cause: Throwable? = null,
) : Exception(message, cause),
    StdError {
    public class Msg(
        message: String,
    ) : UgError(message)

    public class ParseInt(
        message: String,
        cause: Throwable? = null,
    ) : UgError(message, cause)

    public class Io(
        message: String,
        cause: Throwable? = null,
    ) : UgError(message, cause)

    public class FromUtf8(
        message: String,
        cause: Throwable? = null,
    ) : UgError(message, cause)

    public class BorrowMut(
        message: String = "borrow mut error",
    ) : UgError(message)

    public class Borrow(
        message: String = "borrow error",
    ) : UgError(message)

    public class LibLoading(
        message: String,
        cause: Throwable? = null,
    ) : UgError(message, cause)

    public class Wrapped(
        message: String,
        cause: Throwable? = null,
    ) : UgError(message, cause)

    public class WithPath(
        public val inner: UgError,
        public val path: String,
    ) : UgError("path: $path ${inner.message}", inner)

    public class WithBacktrace(
        public val inner: UgError,
        public val backtrace: String = "",
    ) : UgError(inner.message.orEmpty(), inner)

    public class MatMulStriding(
        public val info: MatMulUnexpectedStriding,
    ) : UgError(info.msg)

    public class CannotFindTensor(
        public val path: String,
    ) : UgError("cannot find tensor $path")

    public class SafeTensor(
        message: String,
        cause: Throwable? = null,
    ) : UgError(message, cause)

    public class UnsupportedSafeTensorDtype(
        public val dtypeName: String,
    ) : UgError("unsupported safetensor dtype $dtypeName")

    public fun withPath(path: String): UgError = WithPath(this, path)

    public fun bt(): UgError = this

    public companion object {
        public fun wrap(err: Throwable): UgError = Wrapped(err.message ?: err.toString(), err)

        public fun msg(message: String): UgError = Msg(message)

        public fun debug(obj: Any?): UgError = Msg(obj.toString())
    }
}

public fun ugBail(message: String): Nothing = throw UgError.Msg(message)

public fun <T, U> zip(r1: Result<T>, r2: Result<U>): Result<Pair<T, U>> =
    if (r1.isSuccess && r2.isSuccess) {
        Result.success(Pair(r1.getOrThrow(), r2.getOrThrow()))
    } else if (r1.isFailure) {
        Result.failure(r1.exceptionOrNull()!!)
    } else {
        Result.failure(r2.exceptionOrNull()!!)
    }
