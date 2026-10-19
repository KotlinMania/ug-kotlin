# ug-kotlin in Kotlin

[![GitHub link](https://img.shields.io/badge/GitHub-KotlinMania%2Fug-kotlin-blue.svg)](https://github.com/KotlinMania/ug-kotlin)
[![Maven Central](https://img.shields.io/maven-central/v/io.github.kotlinmania/ug-kotlin)](https://central.sonatype.com/artifact/io.github.kotlinmania/ug-kotlin)
[![Build status](https://img.shields.io/github/actions/workflow/status/KotlinMania/ug-kotlin/ci.yml?branch=main)](https://github.com/KotlinMania/ug-kotlin/actions)

This is a Kotlin Multiplatform line-by-line transliteration port of [`LaurentMazare/ug`](https://github.com/LaurentMazare/ug) (crate `ug`).

**Original Project:** This port is based on [`LaurentMazare/ug`](https://github.com/LaurentMazare/ug). All design credit and project intent belong to the upstream authors; this repository is a faithful port to Kotlin Multiplatform with no behavioural changes intended.

### Porting status

This is an **in-progress port**. The goal is feature parity with the upstream Rust crate while providing a native Kotlin Multiplatform API. Every Kotlin file carries a `// port-lint: source <path>` header naming its upstream Rust counterpart so the AST-distance tool can track provenance.

---

## Upstream README — `LaurentMazare/ug`

> The text below is reproduced from [`https://github.com/LaurentMazare/ug`](https://github.com/LaurentMazare/ug). It is the upstream project's own description and remains under the upstream authors' authorship; links have been rewritten to absolute upstream URLs where appropriate so they continue to resolve from this repository.

# ug

![rust ci badge](https://github.com/LaurentMazare/ug/actions/workflows/rust-ci.yml/badge.svg)

Experimental compiler for deep-learning models inspired by
[triton](https://github.com/triton-lang/triton),
[tinygrad](https://github.com/tinygrad/tinygrad/), and
[micrograd](https://github.com/karpathy/micrograd).

---

## About this Kotlin port

### Installation

```kotlin
dependencies {
    implementation("io.github.kotlinmania:ug-kotlin:0.1.1")
}
```

### Building

```bash
./gradlew build
./gradlew test
```

### Targets

- macOS arm64
- Linux x64
- Windows mingw-x64
- iOS arm64 / simulator-arm64 (Swift export + XCFramework)
- JS (browser + Node.js)
- Wasm-JS (browser + Node.js)
- Android (API 24+)

### Porting guidelines

See [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md) for translator discipline, port-lint header convention, and Rust → Kotlin idiom mapping.

### License

This Kotlin port is distributed under the same license terms as the upstream [`LaurentMazare/ug`](https://github.com/LaurentMazare/ug). See [LICENSE](LICENSE) (and any sibling `LICENSE-*` / `NOTICE` files mirrored from upstream) for the full text.

Original work copyrighted by the ug authors.  
Kotlin port: Copyright (c) 2026 Sydney Renee and The Solace Project.

### Acknowledgments

Thanks to the [`LaurentMazare/ug`](https://github.com/LaurentMazare/ug) maintainers and contributors for the original Rust implementation. This port reproduces their work in Kotlin Multiplatform; bug reports about upstream design or behavior should go to the upstream repository.
