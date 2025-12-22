// Test use statements

mod math {
    fn add(a: i64, b: i64) -> i64 {
        a + b
    }

    fn multiply(a: i64, b: i64) -> i64 {
        a * b
    }
}

// Import just add
use math::add;

fn main() -> i64 {
    // Use imported add directly
    let sum = add(10, 20);  // 30 via import

    // Use fully qualified path for multiply
    let prod = math::multiply(5, 6);  // 30 via full path

    sum + prod  // 60
}
