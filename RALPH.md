# Bolt 0.1.2 Overnight Development

You are autonomously working toward the **bolt-rs 0.1.2 release**.

## STEP 1: Load Context

Read these files in order:
1. `CLAUDE.md` - Architecture overview
2. `TODO.md` - Task list and progress
3. `CHANGELOG.md` - What's planned for 0.1.2

Check current state:
```bash
./target/release/bolt.exe check . 2>&1 | grep -c "^Error:"
```

## STEP 2: Pick Next Task

Work through `TODO.md` in order:
1. **PHASE 1** - Derive macros (most important!)
2. **PHASE 2** - False positive fixes
3. **PHASE 3** - Testing
4. **PHASE 4** - Release prep

Pick the **first uncompleted task** and focus on it.

## STEP 3: Implement Derive Macros

This is the #1 priority. Here's how:

### Location
Create `src/derive.rs` for the expansion logic.

### Integration Point
In `src/parser/mod.rs`, after parsing a struct/enum with `#[derive(...)]`:
1. Parse normally to get the Item
2. Call derive expansion
3. Add generated impls to krate.items

### Template for derive.rs

```rust
//! Derive macro expansion for common traits

use crate::hir::*;

/// Expand derive attributes on an item
pub fn expand_derives(item: &Item, attrs: &[Attribute]) -> Vec<Item> {
    let mut impls = vec![];

    for attr in attrs {
        if attr.path == "derive" {
            for trait_name in &attr.tokens {
                if let Some(impl_item) = expand_single_derive(item, trait_name) {
                    impls.push(impl_item);
                }
            }
        }
    }

    impls
}

fn expand_single_derive(item: &Item, trait_name: &str) -> Option<Item> {
    match trait_name {
        "Clone" => Some(expand_clone(item)),
        "Debug" => Some(expand_debug(item)),
        "Default" => Some(expand_default(item)),
        "Copy" => Some(expand_copy(item)),
        "PartialEq" => Some(expand_partial_eq(item)),
        "Eq" => Some(expand_eq(item)),
        "Hash" => Some(expand_hash(item)),
        _ => None, // Unknown derive, skip
    }
}

fn expand_clone(item: &Item) -> Item {
    // Generate: impl Clone for StructName {
    //     fn clone(&self) -> Self {
    //         Self { field1: self.field1.clone(), ... }
    //     }
    // }
    todo!("implement clone expansion")
}

// ... implement other derives
```

### Testing Each Derive

After implementing each derive:
```bash
# Create test
echo '#[derive(Clone)]
struct Foo { x: i32, y: String }
fn main() {
    let a = Foo { x: 1, y: String::from("hi") };
    let b = a.clone();
}' > /tmp/test_clone.rs

# Test
./target/release/bolt.exe check /tmp/test_clone.rs
```

## STEP 4: Build & Test

After every change:
```bash
cargo build --release 2>&1 | tail -5

# Must not increase errors
./target/release/bolt.exe check . 2>&1 | grep -c "^Error:"

# Examples must still work
./target/release/bolt.exe check examples/very_simple.rs
```

## STEP 5: Commit Progress

After completing each task:
```bash
git add -A
git commit -m "feat(derive): implement Clone expansion"
git push
```

## STEP 6: Update TODO.md

Mark completed tasks with `[x]` and add session notes.

## Session Rules

1. **Focus on derive macros first** - This is the release blocker
2. **Small incremental commits** - Don't batch up huge changes
3. **Test after each change** - No regressions allowed
4. **Update TODO.md** - Track progress for next iteration
5. **If stuck >20 min** - Note blocker, move to Phase 2 tasks

## End of Session Checklist

Before stopping:
- [ ] All changes committed and pushed
- [ ] TODO.md updated with progress
- [ ] CHANGELOG.md updated if needed
- [ ] No build errors
- [ ] Error count same or lower

## Target for Morning

- `#[derive(Clone, Debug, Default)]` working
- Self-check errors < 15
- All tests passing
- Ready for `cargo publish`
