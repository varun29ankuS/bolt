# Bolt ⚡

**A fast Rust compiler for rapid development iteration.**

Bolt compiles Rust code 10-15x faster than rustc by using Cranelift JIT instead of LLVM. It's designed for the development inner loop—not as a rustc replacement, but as a complement to it.

```
Development:  bolt check src/lib.rs     → 172ms
Production:   cargo build --release     → use rustc
```

## Why Bolt?

**The problem**: `cargo check` takes 2+ minutes on medium projects. That's too slow for rapid iteration.

**The solution**: A lightweight compiler optimized for development speed. Bolt handles ~70% of Rust syntax—enough for most iteration cycles. When you hit an edge case, it suggests falling back to rustc.

```
Error: Use of moved value: `x`

  💡 Bolt couldn't handle this. Try: rustc --edition 2021 src/lib.rs
```

## Benchmarks

| File | Bolt | rustc | Speedup |
|------|------|-------|---------|
| 20 lines | 18ms | 186ms | **10x** |
| 80 lines | 26ms | 285ms | **11x** |
| 2300 lines | 172ms | - | - |

## Installation

```bash
git clone https://github.com/varun29ankuS/bolt.git
cd bolt
cargo build --release
```

## Usage

```bash
bolt run main.rs              # Compile and execute
bolt check main.rs            # Type check only
bolt check main.rs -f json    # JSON output (for tooling)
```

### JSON Output

For editor integrations and automated tooling:

```bash
$ bolt check file.rs -f json
```
```json
{
  "diagnostics": [
    {"level": "error", "code": "type_mismatch", "message": "Expected i64, found String"}
  ],
  "summary": {"errors": 1, "warnings": 0}
}
```

## What's Supported

**Working:**
- Functions, structs, enums, generics
- Pattern matching, closures, impl blocks
- Type inference, monomorphization
- Basic borrow checking
- `use`, `Result<T,E>`, `?` operator

**Limitations:**
- No trait bounds (impl blocks work, trait constraints don't)
- No async/await
- No procedural macros
- Some complex type inference patterns

## Architecture

```
Source → Lexer → Parser → HIR → TypeCheck → BorrowCheck → Codegen (Cranelift) → Execute
                   │
                   ├── syn-based (stable)
                   └── Chumsky-based (experimental, faster)
```

~20,000 lines of Rust. Clean pipeline, no magic.

## Roadmap

### Vision

Bolt aims to be the **development compiler** for Rust—optimizing for iteration speed over production performance. The goal is sub-100ms feedback for most changes.

### Milestones

| Phase | Goal | Status |
|-------|------|--------|
| **v0.1** | Core language subset, 10x speedup | ✅ Done |
| **v0.2** | Self-hosting (parse own source) | ✅ Done (16/16 files) |
| **v0.3** | Self-hosting (type-check own source) | ✅ Done (19/19 files) |
| **v0.4** | Trait bounds, better type inference | Planned |
| **v0.5** | Async borrow checking | Planned |
| **v1.0** | Stable API, editor integrations | Planned |

### Future Ideas

- **Expression-level incremental compilation**: Only recompile changed expressions, not whole functions
- **Memory-mapped IR**: Skip parsing for unchanged files entirely
- **Speculative monomorphization**: Pre-compile common generic instantiations (`Vec<i32>`, etc.)

## Project Structure

```
bolt/
├── src/
│   ├── lexer/       # Tokenization
│   ├── parser/      # syn-based parser
│   ├── parser2/     # Chumsky parser (experimental)
│   ├── hir.rs       # High-level IR
│   ├── ty/          # Type system
│   ├── typeck/      # Type inference
│   ├── borrowck/    # Borrow checking
│   ├── codegen/     # Cranelift backend
│   └── cli/         # Command-line interface
└── examples/        # Example programs
```

## Contributing

Issues and PRs welcome. The codebase is intentionally kept simple and readable.

## License

MIT
