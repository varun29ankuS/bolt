# Bolt - Fast Rust Type Checker

## Project Vision
Replace `cargo check` wait times (minutes) with millisecond feedback. Enable rapid Rust development iteration.

## Architecture Overview

```
Source -> Lexer -> Parser -> HIR -> TypeCheck -> BorrowCheck -> Codegen -> Execute
              |         |          |            |             |
           rustc_lexer  syn     type inference  NLL-lite    Cranelift JIT
```

### Key Design Decisions
1. **Stub-based deps** - Don't compile external crates, use pre-defined type stubs
2. **Async borrow check** - Code runs immediately, borrow check in background
3. **Cranelift JIT** - Fast codegen, no LLVM overhead
4. **Lenient mode** - Accept more programs, report all errors without cascading

## Module Map (~26k lines)

| Module | Purpose | Lines |
|--------|---------|-------|
| `codegen/` | Cranelift JIT compilation | 4.8k |
| `parser/` | Syn-based Rust parser | 2.5k |
| `parser2/` | Chumsky parser (experimental) | 2.4k |
| `typeck/` | Type inference & checking | 2k |
| `borrowck/` | Ownership & borrow checking | 2.3k |
| `ty/` | Type system & registry | 1.6k |
| `api.rs` | Public library interface | 1k |
| `cli/` | Command-line interface | 1k |
| `extern_crates/` | Stub definitions | 600 |
| `fixes.rs` | Fix suggestion generation | 650 |
| `hir.rs` | High-level IR definitions | 900 |
| `error.rs` | Diagnostics & errors | 1k |

## Current Capabilities

### Working
- Full type inference with generics
- Pattern matching with exhaustiveness
- Closures with capture analysis
- Trait bounds (basic)
- NLL borrow checking (99% on simple code)
- Cargo.toml integration
- Watch mode with auto-recompile
- JSON output for tooling

### Not Working Yet
- `#[derive(...)]` proc macros
- `async/await`
- Complex trait bounds (GATs, HRTBs)
- Full incremental compilation

## Current Metrics
- Self-check: 26 false positive errors (down from 55)
- Self-check time: ~0.44s
- Simple file: <1ms type check

## Key Files for Common Tasks

| Task | Files |
|------|-------|
| Fix type errors | `src/typeck/mod.rs` |
| Fix borrow errors | `src/borrowck/mod.rs`, `src/borrowck/nll.rs` |
| Add external crate | `src/extern_crates/stubs.rs` |
| Parser issues | `src/parser/mod.rs`, `src/parser/lower.rs` |
| Codegen bugs | `src/codegen/mod.rs` |
| CLI changes | `src/cli/mod.rs` |
| Add fix suggestions | `src/fixes.rs` |

## Testing

```bash
# Quick self-check
./target/release/bolt.exe check .

# Run example
./target/release/bolt.exe run examples/very_simple.rs

# Count errors
./target/release/bolt.exe check . 2>&1 | grep -c "^Error:"

# Rebuild
cargo build --release
```

## Commit Convention
```
feat: new feature
fix: bug fix
perf: performance improvement
refactor: code restructure
docs: documentation
test: add tests
```
