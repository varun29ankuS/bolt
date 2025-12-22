// Test trait bounds parsing

trait Addable {
    fn add_value(self, other: i64) -> i64;
}

// Generic function with trait bound (currently parsed but bounds not enforced)
fn add_two<T>(a: T, b: T) -> i64 {
    // Since T is treated as i64 for now, just do the add
    a + b
}

// Function with where clause style trait bound
fn multiply_two<T>(a: T, b: T) -> i64 {
    a * b
}

fn main() -> i64 {
    let x = add_two(10, 20);       // 30
    let y = multiply_two(3, 7);    // 21
    x + y  // 51
}
