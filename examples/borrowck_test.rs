// Test borrow checker improvements:
// 1. Type alias resolution for Copy types

type DefId = u32;
type TypeId = u32;

fn main() {
    let id: DefId = 42;
    let id2 = id;  // Should NOT error - DefId = u32 is Copy
    let id3 = id;  // Should NOT error - DefId = u32 is Copy
    print(id2);
    print(id3);
    print(999);  // Success marker
}
