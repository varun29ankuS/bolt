# Bolt

**Lightning-fast Rust compiler for development.**

Bolt is an experimental Rust compiler that prioritizes iteration speed over production optimization. It compiles and runs Rust code **100-200x faster** than `rustc`/`cargo` by using Cranelift JIT instead of LLVM.

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
# Clone and build
git clone https://github.com/user/bolt.git
cd bolt
cargo build --release

# Run
./target/release/bolt run your_file.rs
```

## Usage

```bash
# Run a Rust file (compile + execute)
bolt run main.rs

# Check for errors (no codegen)
bolt check main.rs

# Build without running
bolt build main.rs

# LLM-friendly JSON output
bolt check main.rs --format json
bolt check main.rs --format json-pretty

# Cache management
bolt cache stats
bolt cache clear
bolt cache clean
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Bolt Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Source.rs                                                     │
│       │                                                         │
│       ▼                                                         │
│   ┌───────────┐                                                 │
│   │  Parser   │  syn crate → AST                                │
│   └─────┬─────┘                                                 │
│         │                                                       │
│         ▼                                                       │
│   ┌───────────┐                                                 │
│   │  Lowering │  AST → HIR (High-level IR)                      │
│   └─────┬─────┘                                                 │
│         │                                                       │
│         ▼                                                       │
│   ┌───────────┐     ┌──────────────┐                            │
│   │  TypeCk   │────▶│ TypeRegistry │  Unified type system       │
│   └─────┬─────┘     └──────────────┘                            │
│         │                                                       │
│         ▼                                                       │
│   ┌───────────┐                                                 │
│   │ BorrowCk  │  Move/borrow analysis                           │
│   └─────┬─────┘                                                 │
│         │                                                       │
│         ▼                                                       │
│   ┌───────────┐                                                 │
│   │  Codegen  │  Cranelift JIT                                  │
│   └─────┬─────┘                                                 │
│         │                                                       │
│         ▼                                                       │
│      Execute                                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### Implemented

- **Core Language**
  - Functions, structs, enums, tuples
  - Generics with monomorphization
  - Pattern matching
  - References and borrowing
  - Heap allocation (malloc/free)
  - Closures

- **Type System**
  - Full type inference
  - Unified TypeRegistry shared across passes
  - Generic type resolution
  - Struct/enum field access

- **Developer Experience**
  - LLM-friendly JSON diagnostics (BOL-15)
  - Error codes matching rustc (E0382, E0499, etc.)
  - Colored terminal output
  - Compilation caching

- **Syntax Support**
  - `use` statements with all patterns
  - `Result<T, E>` and `?` operator
  - `#[derive(...)]` macros (Default, Clone, Copy)
  - String literals and print functions

### Roadmap

| Priority | Feature | Status | Linear |
|----------|---------|--------|--------|
| P0 | Self-hosting (Bolt compiles Bolt) | Backlog | BOL-44 |
| P1 | Async borrow checking | Backlog | BOL-14 |
| P1 | Trait method resolution | In Progress | - |
| P2 | Expression-level incremental compilation | Backlog | BOL-26 |
| P2 | Memory-mapped IR for instant startup | Backlog | BOL-27, BOL-41 |
| P2 | FFI and extern support | Backlog | BOL-33 |
| P3 | Built-in test framework | Backlog | BOL-36 |
| P3 | Speculative monomorphization | Backlog | BOL-28 |

## Design Philosophy

### 1. Speed Over Optimization
Bolt generates "good enough" code instantly rather than optimal code slowly. For development iteration, 1ms compile time beats 1% faster runtime.

### 2. Compile First, Verify Later (Vision)
The goal is to run code immediately while safety checks happen in the background:
```
Current:  Parse → Type → Borrow → Codegen → Run (serial)
Goal:     Parse → Type → Codegen → Run  [Borrow checking async]
```

### 3. LLM-Native Tooling
Error output is structured for AI consumption:
```json
{
  "level": "error",
  "code": "borrow_of_moved_value",
  "message": "borrow of moved value: `x`",
  "location": {"file": "src/main.rs", "line": 10, "column": 5},
  "suggestions": [{"message": "consider cloning", "replacement": "x.clone()"}]
}
```

### 4. Simplicity
~8,000 lines of Rust. No complex build system, no proc-macro server, no incremental compilation database (yet).

## Project Structure

```
bolt/
├── src/
│   ├── main.rs           # Entry point
│   ├── lib.rs            # Library exports
│   ├── cli/mod.rs        # Command-line interface
│   ├── parser/
│   │   ├── mod.rs        # Parser using syn
│   │   └── lower.rs      # AST → HIR lowering
│   ├── hir.rs            # High-level IR definitions
│   ├── ty/mod.rs         # Unified type system
│   ├── typeck/mod.rs     # Type checking
│   ├── borrowck/mod.rs   # Borrow checking
│   ├── codegen/mod.rs    # Cranelift code generation
│   ├── cache/mod.rs      # Compilation caching
│   └── error.rs          # Diagnostics and error types
└── test/                 # Test programs
```

## Why Not Just Use rustc?

| | rustc | Bolt |
|---|---|---|
| **Goal** | Production binaries | Development speed |
| **Backend** | LLVM (slow, optimal) | Cranelift (fast, good) |
| **Startup** | ~100ms minimum | ~1ms |
| **Use case** | Ship to users | Iterate locally |

Bolt is not a replacement for rustc. Use Bolt for `cargo run` equivalents during development, use rustc/cargo for releases.

## Contributing

See [Linear project](https://linear.app/shodh-memory/project/bolt) for tracked issues and roadmap.

## License

MIT
