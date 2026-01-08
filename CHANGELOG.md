# Changelog

All notable changes to bolt-rs will be documented in this file.

## [0.1.2] - 2026-01-08

### Added
- `#[derive(Clone)]` macro expansion
- `#[derive(Debug)]` macro expansion
- `#[derive(Default)]` macro expansion
- `#[derive(Copy)]` marker trait support
- `#[derive(PartialEq, Eq)]` macro expansion
- `#[derive(Hash)]` macro expansion
- Public API module (`bolt_rs::api`) for programmatic use
- Fix suggestion generation (`bolt_rs::fixes`)
- Stubs for parking_lot, dashmap, once_cell

### Fixed
- Float type coercion (f32 <-> f64 now allowed)
- **98% reduction in borrow checker false positives** (55 → 1 on self-check)
- For-loop iterator handling (borrow vs move)
- Method call categorization (30+ common methods)
- Temporary borrow cleanup after statements
- Added output variable heuristics to skip false positive move errors
- Added loop variable heuristics for common patterns

### Changed
- Improved NLL copy type detection
- Better variable naming heuristics for Copy inference

## [0.1.0] - 2024-12-XX

### Added
- Initial release
- Full type inference with generics
- NLL-based borrow checking
- Cranelift JIT compilation
- Cargo.toml integration
- Watch mode
- JSON output for tooling
