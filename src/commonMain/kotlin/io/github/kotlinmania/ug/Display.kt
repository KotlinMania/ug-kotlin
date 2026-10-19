// port-lint: source display.rs
package io.github.kotlinmania.ug

import io.github.kotlinmania.half.BF16
import io.github.kotlinmania.half.F16
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.roundToLong

public data class PrinterOptions(
    public var precision: Int = 4,
    public var threshold: Int = 1000,
    public var edgeItems: Int = 3,
    public var lineWidth: Int = 80,
    public var sciMode: Boolean? = null,
) {
    public companion object {
        public fun constDefault(): PrinterOptions =
            PrinterOptions(
                precision = 4,
                threshold = 1000,
                edgeItems = 3,
                lineWidth = 80,
                sciMode = null,
            )
    }
}

private var globalPrintOptions: PrinterOptions = PrinterOptions.constDefault()

public fun printOptions(): PrinterOptions = globalPrintOptions

public fun setPrintOptions(options: PrinterOptions) {
    globalPrintOptions = options
}

public fun setPrintOptionsDefault() {
    globalPrintOptions = PrinterOptions.constDefault()
}

public fun setPrintOptionsShort() {
    globalPrintOptions =
        PrinterOptions(
            precision = 2,
            threshold = 1000,
            edgeItems = 2,
            lineWidth = 80,
            sciMode = null,
        )
}

public fun setPrintOptionsFull() {
    globalPrintOptions =
        PrinterOptions(
            precision = 4,
            threshold = Int.MAX_VALUE,
            edgeItems = 3,
            lineWidth = 80,
            sciMode = null,
        )
}

public fun setLineWidth(lineWidth: Int) {
    globalPrintOptions.lineWidth = lineWidth
}

public fun setPrecision(precision: Int) {
    globalPrintOptions.precision = precision
}

public fun setEdgeItems(edgeItems: Int) {
    globalPrintOptions.edgeItems = edgeItems
}

public fun setThreshold(threshold: Int) {
    globalPrintOptions.threshold = threshold
}

public fun setSciMode(sciMode: Boolean?) {
    globalPrintOptions.sciMode = sciMode
}

internal class FloatFormatter(
    val intMode: Boolean,
    val sciMode: Boolean,
    val precision: Int,
) {
    fun formatElem(v: Float): String {
        if (v.isNaN()) return "NaN"
        if (v == Float.POSITIVE_INFINITY) return "inf"
        if (v == Float.NEGATIVE_INFINITY) return "-inf"

        if (sciMode) {
            if (v == 0.0f) {
                val frac = "0".repeat(precision)
                return if (precision > 0) "0.${frac}e0" else "0e0"
            }
            val sign = if (v < 0.0f) "-" else ""
            val absV = abs(v.toDouble())
            var exp = floor(log10(absV)).toInt()
            var mantissa = absV / 10.0.pow(exp.toDouble())
            val factor = 10.0.pow(precision.toDouble())
            var roundedMantissa = (mantissa * factor).roundToLong()
            if (roundedMantissa >= (10.0 * factor).roundToLong()) {
                exp += 1
                roundedMantissa = (1.0 * factor).roundToLong()
            }
            val intPart = roundedMantissa / factor.toLong()
            val fracPart = roundedMantissa % factor.toLong()
            val fracStr = fracPart.toString().padStart(precision, '0')
            val mantissaStr = if (precision > 0) "$intPart.$fracStr" else "$intPart"
            return "$sign$mantissaStr" + "e$exp"
        }

        if (intMode) {
            val longVal = v.toLong()
            val sign = if (v < 0.0f && longVal == 0L) "-" else ""
            return "$sign$longVal."
        }

        val sign = if (v < 0.0f) "-" else ""
        val absV = abs(v.toDouble())
        val factor = 10.0.pow(precision.toDouble())
        val rounded = (absV * factor).roundToLong()
        val intPart = rounded / factor.toLong()
        val fracPart = rounded % factor.toLong()
        val fracStr = fracPart.toString().padStart(precision, '0')
        return if (precision > 0) "$sign$intPart.$fracStr" else "$sign$intPart"
    }

    fun fmt(v: Float, maxW: Int, sb: StringBuilder) {
        val s = formatElem(v)
        sb.append(s.padStart(maxW))
    }

    companion object {
        fun new(t: LazyBuffer, po: PrinterOptions): FloatFormatter {
            var intMode = true
            var sciMode = false
            val values =
                try {
                    t.dataVec<Float>().filter { !it.isNaN() && !it.isInfinite() && it != 0.0f }
                } catch (_: Throwable) {
                    emptyList()
                }

            if (values.isNotEmpty()) {
                var nonzeroFiniteMin = Float.MAX_VALUE
                var nonzeroFiniteMax = 0.0f
                for (v in values) {
                    val a = abs(v)
                    if (a < nonzeroFiniteMin) nonzeroFiniteMin = a
                    if (a > nonzeroFiniteMax) nonzeroFiniteMax = a
                }

                for (value in values) {
                    if (kotlin.math.ceil(value) != value) {
                        intMode = false
                        break
                    }
                }

                if (nonzeroFiniteMax / nonzeroFiniteMin > 1000.0f ||
                    nonzeroFiniteMax > 1e8f ||
                    nonzeroFiniteMin < 1e-4f
                ) {
                    sciMode = true
                }
            }

            po.sciMode?.let { sciMode = it }
            return FloatFormatter(intMode = intMode, sciMode = sciMode, precision = po.precision)
        }
    }
}

internal class IntFormatter {
    fun fmt(v: Long, maxW: Int, sb: StringBuilder) {
        sb.append(v.toString().padStart(maxW))
    }
}

private fun getSummarizedData(t: LazyBuffer, edgeItems: Int): LazyBuffer {
    val dims = t.dims()
    if (dims.isEmpty()) {
        return t
    } else if (dims.size == 1) {
        return if (dims[0] > 2 * edgeItems) {
            val n1 = t.narrow(0.D, 0, edgeItems)
            val n2 = t.narrow(0.D, dims[0] - edgeItems, edgeItems)
            LazyBuffer.cat(listOf(n1, n2), 0)
        } else {
            t
        }
    } else if (dims[0] > 2 * edgeItems) {
        val vs = mutableListOf<LazyBuffer>()
        for (i in 0 until edgeItems) {
            vs.add(getSummarizedData(t.get(i), edgeItems))
        }
        for (i in (dims[0] - edgeItems) until dims[0]) {
            vs.add(getSummarizedData(t.get(i), edgeItems))
        }
        return LazyBuffer.cat(vs, 0)
    } else {
        val vs = mutableListOf<LazyBuffer>()
        for (i in 0 until dims[0]) {
            vs.add(getSummarizedData(t.get(i), edgeItems))
        }
        return LazyBuffer.cat(vs, 0)
    }
}

private fun writeNewlineIndent(i: Int, sb: StringBuilder) {
    sb.append('\n')
    for (k in 0 until i) {
        sb.append(' ')
    }
}

private fun fmtTensorFloat(
    t: LazyBuffer,
    indent: Int,
    maxW: Int,
    summarize: Boolean,
    po: PrinterOptions,
    tf: FloatFormatter,
    sb: StringBuilder,
) {
    val dims = t.dims()
    val edgeItems = po.edgeItems
    sb.append('[')
    when {
        dims.isEmpty() -> {
            val v = t.dataVec<Float>()
            if (v.isNotEmpty()) {
                tf.fmt(v[0], maxW, sb)
            }
        }

        dims.size == 1 && summarize && dims[0] > 2 * edgeItems -> {
            val vs1 = t.narrow(0.D, 0, edgeItems).dataVec<Float>()
            for (v in vs1) {
                tf.fmt(v, maxW, sb)
                sb.append(", ")
            }
            sb.append("...")
            val vs2 = t.narrow(0.D, dims[0] - edgeItems, edgeItems).dataVec<Float>()
            for (v in vs2) {
                sb.append(", ")
                tf.fmt(v, maxW, sb)
            }
        }

        dims.size == 1 -> {
            val elementsPerLine = maxOf(1, po.lineWidth / (maxW + 2))
            val vs = t.dataVec<Float>()
            for ((i, v) in vs.withIndex()) {
                if (i > 0) {
                    if (i % elementsPerLine == 0) {
                        sb.append(',')
                        writeNewlineIndent(indent, sb)
                    } else {
                        sb.append(", ")
                    }
                }
                tf.fmt(v, maxW, sb)
            }
        }

        else -> {
            if (summarize && dims[0] > 2 * edgeItems) {
                for (i in 0 until edgeItems) {
                    fmtTensorFloat(t.get(i), indent + 1, maxW, summarize, po, tf, sb)
                    sb.append(',')
                    writeNewlineIndent(indent, sb)
                }
                sb.append("...")
                writeNewlineIndent(indent, sb)
                for (i in (dims[0] - edgeItems) until dims[0]) {
                    fmtTensorFloat(t.get(i), indent + 1, maxW, summarize, po, tf, sb)
                    if (i + 1 != dims[0]) {
                        sb.append(',')
                        writeNewlineIndent(indent, sb)
                    }
                }
            } else {
                for (i in 0 until dims[0]) {
                    fmtTensorFloat(t.get(i), indent + 1, maxW, summarize, po, tf, sb)
                    if (i + 1 != dims[0]) {
                        sb.append(',')
                        writeNewlineIndent(indent, sb)
                    }
                }
            }
        }
    }
    sb.append(']')
}

private fun LazyBuffer.dataVecIntLong(): List<Long> =
    if (dtype == DType.I32) dataVec<Int>().map { it.toLong() } else dataVec<Long>()

private fun fmtTensorInt(
    t: LazyBuffer,
    indent: Int,
    maxW: Int,
    summarize: Boolean,
    po: PrinterOptions,
    tf: IntFormatter,
    sb: StringBuilder,
) {
    val dims = t.dims()
    val edgeItems = po.edgeItems
    sb.append('[')
    when {
        dims.isEmpty() -> {
            val v = t.dataVecIntLong()
            if (v.isNotEmpty()) {
                tf.fmt(v[0], maxW, sb)
            }
        }

        dims.size == 1 && summarize && dims[0] > 2 * edgeItems -> {
            val vs1 = t.narrow(0.D, 0, edgeItems).dataVecIntLong()
            for (v in vs1) {
                tf.fmt(v, maxW, sb)
                sb.append(", ")
            }
            sb.append("...")
            val vs2 = t.narrow(0.D, dims[0] - edgeItems, edgeItems).dataVecIntLong()
            for (v in vs2) {
                sb.append(", ")
                tf.fmt(v, maxW, sb)
            }
        }

        dims.size == 1 -> {
            val elementsPerLine = maxOf(1, po.lineWidth / (maxW + 2))
            val vs = t.dataVecIntLong()
            for ((i, v) in vs.withIndex()) {
                if (i > 0) {
                    if (i % elementsPerLine == 0) {
                        sb.append(',')
                        writeNewlineIndent(indent, sb)
                    } else {
                        sb.append(", ")
                    }
                }
                tf.fmt(v, maxW, sb)
            }
        }

        else -> {
            if (summarize && dims[0] > 2 * edgeItems) {
                for (i in 0 until edgeItems) {
                    fmtTensorInt(t.get(i), indent + 1, maxW, summarize, po, tf, sb)
                    sb.append(',')
                    writeNewlineIndent(indent, sb)
                }
                sb.append("...")
                writeNewlineIndent(indent, sb)
                for (i in (dims[0] - edgeItems) until dims[0]) {
                    fmtTensorInt(t.get(i), indent + 1, maxW, summarize, po, tf, sb)
                    if (i + 1 != dims[0]) {
                        sb.append(',')
                        writeNewlineIndent(indent, sb)
                    }
                }
            } else {
                for (i in 0 until dims[0]) {
                    fmtTensorInt(t.get(i), indent + 1, maxW, summarize, po, tf, sb)
                    if (i + 1 != dims[0]) {
                        sb.append(',')
                        writeNewlineIndent(indent, sb)
                    }
                }
            }
        }
    }
    sb.append(']')
}

public fun LazyBuffer.formatTensor(options: PrinterOptions = printOptions()): String {
    realize()
    val po = options
    val summarize = elemCount() > po.threshold
    val toDisplay =
        if (summarize) {
            getSummarizedData(this, po.edgeItems)
        } else {
            this
        }
    val sb = StringBuilder()
    when (dtype) {
        DType.I32, DType.I64 -> {
            val tf = IntFormatter()
            val vs =
                try {
                    toDisplay.dataVecIntLong()
                } catch (_: Throwable) {
                    emptyList()
                }
            val maxW = vs.maxOfOrNull { it.toString().length } ?: 1
            fmtTensorInt(this, 1, maxW, summarize, po, tf, sb)
            sb.append('\n')
        }

        DType.F32, DType.F16, DType.BF16 -> {
            val tf = FloatFormatter.new(toDisplay, po)
            val vs =
                try {
                    toDisplay.dataVec<Float>()
                } catch (_: Throwable) {
                    emptyList()
                }
            val maxW = vs.maxOfOrNull { tf.formatElem(it).length } ?: 1
            fmtTensorFloat(this, 1, maxW, summarize, po, tf, sb)
            sb.append('\n')
        }
    }
    val dimsStr = shape.dims.toString()
    sb.append("Tensor[$dimsStr, ${dtype.asStr}]")
    return sb.toString()
}
