# Bolt ⚡

[![Crates.io](https://img.shields.io/crates/v/bolt-rs.svg)](https://crates.io/crates/bolt-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Lightning-fast Rust type checker for rapid development.**

Get instant feedback on your Rust code. Bolt checks your source files in milliseconds by using stubs for external crates instead of compiling them.

```bash
bolt check .        # ~50ms  (your code only, stubs for deps)
cargo check         # 3+ min cold, ~2s cached
```

## Install

```bash
cargo install bolt-rs   # Installs the 'bolt' binary
```

Or build from source:
```bash
git clone https://github.com/varun29ankuS/bolt.git
cd bolt && cargo build --release
```

## Usage

```bash
# Check a file
bolt check src/main.rs

# Check entire project (auto-detects Cargo.toml)
bolt check .

# JSON output for tooling/IDE integration
bolt check . --format json
```

## Why Bolt?

**The problem**: `cargo check` takes seconds to minutes. That breaks your flow.

**The solution**: A fast type checker optimized for the edit-check loop. Check your code in milliseconds, not seconds.

```
You write code
    ↓
bolt check .     ← ~50ms feedback
    ↓
Fix errors
    ↓
bolt check .     ← ~50ms feedback
    ↓
Ready? → cargo build --release
```

## What Works

| Feature | Status |
|---------|--------|
| Type checking | ✅ Full |
| Borrow checking | ✅ Async (non-blocking) |
| Watch mode | ✅ `bolt watch --run` with auto-cancel |
| Cargo.toml detection | ✅ Auto-finds entry point |
| External crates | ✅ Stub resolution (serde, std, etc.) |
| Generics | ✅ Full |
| Pattern matching | ✅ Full |
| Closures | ✅ Full |
| impl blocks | ✅ Full |
| Error recovery | ✅ Graceful fallback suggestions |

### Async Borrow Checking

Bolt's borrow checker runs **in the background** - your code executes immediately while safety checks happen concurrently. If issues are found, they're reported after execution (and block the next run).

```bash
bolt run main.rs    # Runs instantly, borrow check in background
                    # If errors found → shown after execution
                    # Next run blocks until you fix them
```

## What Doesn't Work (Yet)

| Feature | Workaround |
|---------|------------|
| Proc macros (#[derive]) | Manual impl or use rustc |
| async/await | Use rustc for async code |
| Complex trait bounds | Bolt suggests rustc fallback |

When Bolt can't handle something, it tells you:
```
Error: Bolt couldn't fully check this file
  💡 Try: rustc --edition 2021 src/lib.rs
```

## Benchmarks

| Scenario | Bolt | cargo check | Notes |
|----------|------|-------------|-------|
| Single file | ~25-50ms | ~150ms (cached) | Bolt is 3-6x faster |
| Project (cold) | <1s | 3+ minutes | Bolt skips dependency compilation |
| Project (cached) | <1s | ~1-3s | Similar for cached builds |

*Bolt checks your code only - it uses stubs for external crates instead of compiling them. This is why it's fast but may miss some cross-crate type errors.*

## Architecture

```
Source → Parse → HIR → TypeCheck → Run Code
                           ↓           ↑
                    External Stubs     │ (async)
                                       ↓
                                  BorrowCheck
```

**Key innovations:**
- **Async Borrow Checking**: Safety checks run in background - code executes immediately
- **Ownership Ledger**: Blockchain-inspired borrow tracking with full audit trail
- **Stub Resolution**: Type-check against external crates without compiling them
- **Watch Mode**: Concurrent compilation with automatic cancellation on new changes

## Roadmap

| Version | Focus | Status |
|---------|-------|--------|
| **0.1** | **Core type checking, async borrow checker, watch mode** | ✅ Current |
| 0.2 | Proc macro expansion (#[derive] support) | 🔄 Next |
| 0.3 | Incremental checking (only re-check changed functions) | Planned |
| 0.4 | Hybrid build (bolt compile + cargo link) | Planned |
| 1.0 | IDE integrations, LSP server | Planned |

### Vision

**Gradual compilation**: Eventually, Bolt aims to compile your code incrementally at the function level - only recompiling what changed. Combined with async borrow checking and Cranelift's fast codegen, this could enable sub-10ms feedback loops.

**Hybrid builds**: Use Bolt for rapid iteration, then seamlessly hand off to cargo for production builds with full optimization.

## Contributing

PRs welcome! The codebase is ~25k lines of clean, readable Rust.

```
src/
├── parser/      # syn-based parsing
├── typeck/      # Type inference
├── borrowck/    # Borrow checking + ownership ledger
├── extern_crates/ # External crate stubs
└── cli/         # Command line interface
```

## License

MIT

---

**Bolt is not a rustc replacement.** It's a fast feedback tool for development. Use rustc/cargo for production builds.
