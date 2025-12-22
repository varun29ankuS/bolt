// Test heap allocation with array-like indexing

fn main() -> i64 {
    // Allocate space for 8 i64s (64 bytes)
    let mut ptr = bolt_malloc(64);

    // Store values using index operator
    ptr[0] = 10;
    ptr[1] = 20;
    ptr[2] = 30;
    ptr[3] = 40;

    // Read them back
    let a = ptr[0];  // 10
    let b = ptr[1];  // 20
    let c = ptr[2];  // 30
    let d = ptr[3];  // 40

    // Free the memory
    bolt_free(ptr, 64);

    a + b + c + d  // 100
}
