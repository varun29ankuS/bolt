# Bolt Overnight TODO

Last updated: 2026-01-07
Current error count: 26

## Priority 1: Proc Macro Support (HIGH IMPACT)

### Goal: Support `#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]`

This is the #1 blocker for real-world usage. Most Rust code uses derive macros.

- [ ] **1.1** Add derive expansion framework in `src/parser/mod.rs`
  - Detect `#[derive(...)]` attributes on structs/enums
  - Generate impl blocks for each derived trait

- [ ] **1.2** Implement `derive(Debug)`
  - Generate `fn fmt(&self, f: &mut Formatter) -> Result<(), Error>`
  - Handle struct fields, enum variants

- [ ] **1.3** Implement `derive(Clone)`
  - Generate `fn clone(&self) -> Self`
  - Call `.clone()` on each field

- [ ] **1.4** Implement `derive(Copy)` - marker trait, just register

- [ ] **1.5** Implement `derive(Default)`
  - Generate `fn default() -> Self`
  - Use `Default::default()` for each field

- [ ] **1.6** Implement `derive(PartialEq, Eq)`
  - Generate `fn eq(&self, other: &Self) -> bool`
  - Compare each field

- [ ] **1.7** Implement `derive(Hash)`
  - Generate `fn hash<H: Hasher>(&self, state: &mut H)`

## Priority 2: Reduce False Positives (QUALITY)

### Goal: Get self-check to 0 errors

Current patterns causing issues:
- [ ] **2.1** "cannot move out of X because it is borrowed" in loops
- [ ] **2.2** "use of moved value" for iterator variables
- [ ] **2.3** HashMap/Vec iteration patterns
- [ ] **2.4** Pattern matching with moves

Files: `src/borrowck/mod.rs`, `src/borrowck/nll.rs`

## Priority 3: Performance (SPEED)

### Goal: <100ms for any project

- [ ] **3.1** Profile self-check bottlenecks
- [ ] **3.2** Add parallel type checking per function
- [ ] **3.3** Implement expression-level caching
- [ ] **3.4** Lazy parsing for unchanged files

## Priority 4: External Crates (USABILITY)

### Goal: Auto-stub from rlib files

- [ ] **4.1** Parse `.rlib` metadata for type signatures
- [ ] **4.2** Cache extracted stubs
- [ ] **4.3** Add stubs for top 20 crates.io crates

## Priority 5: Code Quality

- [ ] **5.1** Add integration tests for each example
- [ ] **5.2** Document public API
- [ ] **5.3** Add benchmarks

---

## Session Log

### Session 1 (2026-01-07)
- Fixed float type coercion (f32 <-> f64)
- Improved NLL copy type detection
- Added iterator/loop handling
- Added method call categorization
- Reduced errors: 55 -> 26
- Added api.rs and fixes.rs modules

### Session 2 (overnight)
- [ ] Start here...

---

## How to Update This File

After completing a task:
1. Mark it `[x]`
2. Add notes under Session Log
3. Update "Current error count" at top
4. Commit: `git add TODO.md && git commit -m "docs: update progress" && git push`
