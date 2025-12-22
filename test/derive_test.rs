// Test derive macros - Default

#[derive(Default)]
struct Counter {
    value: i64,
}

fn main() -> i64 {
    // Test Default - creates Counter { value: 0 }
    let c = Counter_default();
    c.value  // Should be 0
}
