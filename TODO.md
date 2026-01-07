# Bolt 0.1.2 Release TODO

**Target:** Morning of 2026-01-08
**Current errors:** 26

---

## PHASE 1: Derive Macros [CRITICAL]

### 1.1 Create derive expansion module
- [ ] Create `src/derive.rs` with expansion framework
- [ ] Hook into parser to detect `#[derive(...)]` attributes
- [ ] Generate impl blocks and add to HIR

### 1.2 Implement derive(Clone)
- [ ] Generate `fn clone(&self) -> Self`
- [ ] Handle struct fields (call `.clone()` on each)
- [ ] Handle tuple structs
- [ ] Handle unit structs
- [ ] Test: `examples/derive_clone.rs`

### 1.3 Implement derive(Debug)
- [ ] Generate `fn fmt(&self, f: &mut Formatter) -> Result`
- [ ] Handle struct fields with `{:?}` formatting
- [ ] Handle enums with variant names
- [ ] Test: create `examples/derive_debug.rs`

### 1.4 Implement derive(Default)
- [ ] Generate `fn default() -> Self`
- [ ] Use `Default::default()` for each field
- [ ] Test: create `examples/derive_default.rs`

### 1.5 Implement derive(Copy)
- [ ] Register as marker trait (no methods)
- [ ] Mark struct as Copy in type registry
- [ ] Test: create `examples/derive_copy.rs`

### 1.6 Implement derive(PartialEq, Eq)
- [ ] Generate `fn eq(&self, other: &Self) -> bool`
- [ ] Compare each field with `==`
- [ ] Test: create `examples/derive_eq.rs`

### 1.7 Implement derive(Hash)
- [ ] Generate `fn hash<H: Hasher>(&self, state: &mut H)`
- [ ] Call `.hash(state)` on each field
- [ ] Test: create `examples/derive_hash.rs`

---

## PHASE 2: False Positive Fixes [HIGH]

### 2.1 Loop variable patterns
- [ ] Fix "use of moved value" in for loops
- [ ] Handle iterator rebinding
- [ ] File: `src/borrowck/mod.rs`

### 2.2 HashMap iteration
- [ ] Fix borrow conflicts during iteration
- [ ] Handle `.iter()`, `.iter_mut()`, `.into_iter()`
- [ ] File: `src/borrowck/nll.rs`

### 2.3 Pattern matching moves
- [ ] Fix moves in match arms
- [ ] Handle `if let` patterns
- [ ] File: `src/borrowck/mod.rs`

**Target:** Reduce errors from 26 to <10

---

## PHASE 3: Testing [MEDIUM]

### 3.1 Integration tests
- [ ] Add `tests/` directory
- [ ] Test all examples pass
- [ ] Test error detection works

### 3.2 Derive macro tests
- [ ] Test each derive variant
- [ ] Test combined derives `#[derive(Clone, Debug, Default)]`

---

## PHASE 4: Release [FINAL]

### 4.1 Documentation
- [ ] Update README.md with new features
- [ ] Verify CHANGELOG.md is complete
- [ ] Check all doc comments

### 4.2 Publish
- [ ] `cargo test`
- [ ] `cargo publish --dry-run`
- [ ] `cargo publish`
- [ ] Tag release: `git tag v0.1.2 && git push --tags`

---

## Session Log

### Session 1: 2026-01-07 (human + AI)
- Fixed float coercion
- Improved borrow checker
- Added api.rs, fixes.rs
- Errors: 55 → 26
- Set up overnight automation

### Session 2: 2026-01-07 overnight (autonomous)
_Start here..._

---

## Quick Reference

```bash
# Build
cargo build --release

# Test
./target/release/bolt.exe check .
./target/release/bolt.exe check examples/derive_clone.rs

# Error count
./target/release/bolt.exe check . 2>&1 | grep -c "^Error:"

# Commit
git add -A && git commit -m "feat: [description]" && git push
```
