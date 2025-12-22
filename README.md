# Bolt

**Lightning-fast Rust compiler for development.**

Bolt is an experimental Rust compiler that prioritizes iteration speed over production optimization. It compiles and runs Rust code **100-200x faster** than `rustc`/`cargo` by using Cranelift JIT instead of LLVM, with a vision for **async safety checking** and **expression-level incremental compilation**.

## Benchmarks

| Compiler | Time | Notes |
|----------|------|-------|
| **Bolt** | **~1.2ms** | Parse + Type Check + JIT + Execute |
| rustc | ~200ms | Compile only |
| cargo (cold) | ~400ms | Full build |
| cargo (incremental) | ~110ms | No changes |

```
Bolt:  ████ 1.2ms
rustc: ████████████████████████████████████████████████████████████████████████████████ 200ms
```

## Installation

```bash
git clone https://github.com/varun29ankuS/bolt.git
cd bolt
cargo build --release
./target/release/bolt run your_file.rs
```

## Usage

```bash
bolt run main.rs              # Compile + execute
bolt check main.rs            # Check for errors (no codegen)
bolt build main.rs            # Build without running
bolt check main.rs -f json    # LLM-friendly JSON output
bolt cache stats              # Cache statistics
```

## Vision & Architecture

Bolt isn't just "rustc but faster" - it rethinks the compilation model for development iteration.

### Core Innovations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BOLT ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        FAST PATH (~1ms)                             │   │
│   │                                                                     │   │
│   │   Source ──▶ Parse ──▶ HIR ──▶ TypeCheck ──▶ Codegen ──▶ Execute   │   │
│   │                         │                       │                   │   │
│   └─────────────────────────┼───────────────────────┼───────────────────┘   │
│                             │                       │                       │
│   ┌─────────────────────────▼───────────────────────▼───────────────────┐   │
│   │                    BACKGROUND (async)                               │   │
│   │                                                                     │   │
│   │   ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐   │   │
│   │   │  BorrowCheck │    │  Speculative    │    │   Cache/mmap     │   │   │
│   │   │  (eventual)  │    │  Monomorph      │    │   IR fragments   │   │   │
│   │   └──────────────┘    └─────────────────┘    └──────────────────┘   │   │
│   │         │                                                           │   │
│   │         ▼                                                           │   │
│   │   Warnings surface async (next compile blocks if unsafe)            │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    PERSISTENT CACHE                                 │   │
│   │                                                                     │   │
│   │   ┌────────────────┐  ┌────────────────┐  ┌─────────────────────┐   │   │
│   │   │ Expression IR  │  │ Mono instances │  │ mmap'd HIR (skip    │   │   │
│   │   │ fragments      │  │ Vec<i32>, etc  │  │ parsing entirely)   │   │   │
│   │   └────────────────┘  └────────────────┘  └─────────────────────┘   │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. Compile First, Verify Later

Traditional: `Parse → Type → Borrow → Codegen → Run` (all blocking)

Bolt vision:
```
Parse → Type → Codegen → Run     [~1ms, you see results]
         └──▶ BorrowCheck        [async, warns if unsafe]
```

Code runs **immediately**. If borrow checker finds issues, they surface as warnings. Next compilation blocks only if previous check failed.

### 2. Expression-Level Incremental Compilation

rustc's incremental is function-level. Bolt aims for **expression-level**:

```rust
fn complex() -> i64 {
    let a = expensive_op1();  // cached ✓
    let b = expensive_op2();  // cached ✓
    let c = a + b;            // CHANGED → recompile only this
    let d = more_work(c);     // depends on c → recompile
    d
}
```

Each expression hashed by AST + dependencies. Only recompile what changed.

### 3. Speculative Monomorphization

Pre-compile common generic instantiations before they're needed:

```
Background thread profiles usage:
  Vec<i32>     → pre-compiled ✓
  Option<String> → pre-compiled ✓
  HashMap<K,V> → ready when you need it
```

When your code calls `Vec::new()`, the compiled version is already waiting.

### 4. Memory-Mapped IR

Skip parsing entirely for unchanged files:

```
Traditional: Load → Parse → Typecheck → Codegen  [2ms]
With mmap:   mmap cached HIR → Codegen           [0.05ms]
```

Persist compiled IR to disk. Memory-map it back. Zero deserialization.

## Current Status

### Implemented ✓

- **Core Language**: Functions, structs, enums, generics, pattern matching, closures
- **Type System**: Full inference, unified TypeRegistry, monomorphization
- **LLM Diagnostics**: JSON output with error codes, suggestions, confidence levels
- **Syntax**: `use`, `Result<T,E>`, `?` operator, `#[derive(...)]`

### In Progress

- Trait method resolution
- Expression type tracking for codegen

### Roadmap

| Stage | Feature | Impact |
|-------|---------|--------|
| **Now** | Trait methods, lifetimes | Core language completeness |
| **Next** | Async borrow checker | Instant feedback, safety async |
| **Then** | Expression-level incremental | Sub-millisecond warm compiles |
| **Later** | Memory-mapped IR | 0.05ms startup |
| **Goal** | Self-hosting (BOL-44) | Bolt compiles Bolt |

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Cold compile | ~1.2ms | <5ms |
| Warm compile | ~1.2ms | <0.5ms (expression-level) |
| Startup (cached) | ~1ms | <0.1ms (mmap) |
| Rust coverage | ~40% | 80% (self-hosting) |

## Project Structure

```
bolt/
├── src/
│   ├── cli/mod.rs        # CLI with JSON output support
│   ├── parser/           # syn-based parser + HIR lowering
│   ├── hir.rs            # High-level IR
│   ├── ty/mod.rs         # Unified type system (shared typeck↔codegen)
│   ├── typeck/mod.rs     # Type inference and checking
│   ├── borrowck/mod.rs   # Borrow checking (to become async)
│   ├── codegen/mod.rs    # Cranelift JIT
│   ├── cache/mod.rs      # Compilation caching
│   └── error.rs          # LLM-friendly diagnostics
└── test/                 # Test programs
```

## Why Bolt?

| | rustc | Bolt |
|---|---|---|
| **Philosophy** | Correctness first | Speed first, correctness async |
| **Backend** | LLVM (optimal, slow) | Cranelift (good, fast) |
| **Incremental** | Function-level | Expression-level (planned) |
| **Safety check** | Blocking | Async (planned) |
| **Target** | Production | Development |

Bolt is your development compiler. rustc is your release compiler.

## Links

- [Linear Project](https://linear.app/shodh-memory/project/bolt) - Issues and roadmap
- [BOL-14](https://linear.app/shodh-memory/issue/BOL-14) - Async borrow checking vision
- [BOL-44](https://linear.app/shodh-memory/issue/BOL-44) - Self-hosting milestone

## License

MIT
