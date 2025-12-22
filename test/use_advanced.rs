// Test advanced use statements

mod math {
    fn add(a: i64, b: i64) -> i64 {
        a + b
    }

    fn subtract(a: i64, b: i64) -> i64 {
        a - b
    }

    fn multiply(a: i64, b: i64) -> i64 {
        a * b
    }
}

mod geometry {
    fn square(x: i64) -> i64 {
        x * x
    }
}

// Import multiple items with curly braces
use math::{add, subtract};

// Import with alias
use math::multiply as mult;

// Import from another module
use geometry::square;

fn main() -> i64 {
    let a = add(5, 3);        // 8
    let b = subtract(10, 4);  // 6
    let c = mult(3, 4);       // 12
    let d = square(5);        // 25

    a + b + c + d  // 8 + 6 + 12 + 25 = 51
}
