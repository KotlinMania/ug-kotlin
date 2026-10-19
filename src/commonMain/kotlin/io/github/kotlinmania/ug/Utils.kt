// port-lint: source utils.rs
package io.github.kotlinmania.ug

public object Utils {
    public var keepTmp: Boolean = false

    public fun getNumThreads(): Int = 4
}

public fun getNumThreads(): Int = Utils.getNumThreads()
