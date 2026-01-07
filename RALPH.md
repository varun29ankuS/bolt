# Bolt Autonomous Development Session

You are autonomously improving `bolt-rs`, a fast Rust type checker.

## First: Read Context

1. Read `CLAUDE.md` for project architecture
2. Read `TODO.md` for current priorities and progress
3. Check current state: `./target/release/bolt.exe check . 2>&1 | grep -c "^Error:"`

## Your Session Goal

Pick the HIGHEST PRIORITY uncompleted task from `TODO.md` and work on it.

**Priority order:**
1. Proc macro support (derive) - HIGHEST IMPACT
2. Reduce false positives
3. Performance
4. External crates

## Work Process

### 1. Understand the Task
- Read relevant source files
- Understand existing patterns
- Plan your approach

### 2. Implement Incrementally
- Make small, testable changes
- Build frequently: `cargo build --release 2>&1 | tail -10`
- Test after each change

### 3. Verify
```bash
# Must pass
cargo build --release

# Check for regressions
./target/release/bolt.exe check . 2>&1 | grep -c "^Error:"

# Test examples still work
./target/release/bolt.exe check examples/very_simple.rs
```

### 4. Commit Progress
```bash
git add -A
git commit -m "feat: [what you did]"
git push
```

### 5. Update TODO.md
- Mark completed items `[x]`
- Add session notes
- Update error count

## Proc Macro Implementation Guide

If working on `#[derive(...)]`:

### Location
`src/parser/mod.rs` or new file `src/derive.rs`

### Approach
```rust
// In parser, when you see #[derive(Trait)] on a struct:
// 1. Parse the struct normally
// 2. For each derived trait, generate an impl block
// 3. Add the impl to krate.items

fn expand_derive(struct_item: &Item, traits: &[String]) -> Vec<Item> {
    let mut impls = vec![];
    for trait_name in traits {
        match trait_name.as_str() {
            "Clone" => impls.push(generate_clone_impl(struct_item)),
            "Debug" => impls.push(generate_debug_impl(struct_item)),
            // etc
            _ => {} // Unknown derive, skip
        }
    }
    impls
}
```

### Testing Derive
```bash
# Create test file
echo 'struct Foo { x: i32 }
impl Clone for Foo { fn clone(&self) -> Self { Foo { x: self.x } } }
fn main() { let a = Foo { x: 1 }; let b = a.clone(); }' > /tmp/test.rs

./target/release/bolt.exe check /tmp/test.rs
```

## Session Rules

1. **One task at a time** - Complete or make significant progress before switching
2. **Always commit working code** - Don't leave broken state
3. **Update TODO.md** - Track what you did
4. **Test before commit** - Verify no regressions
5. **If stuck > 30min** - Move to next task, note blocker in TODO.md

## End of Session

Before stopping:
1. Commit all working changes
2. Update TODO.md with progress
3. Note any blockers or ideas for next session
