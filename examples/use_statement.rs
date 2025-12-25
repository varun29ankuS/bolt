// Test use statements

mod math {
    pub fn add(a: i64, b: i64) -> i64 {
        a + b
    }

    pub fn multiply(a: i64, b: i64) -> i64 {
        a * b
    }
}

use math::add;
use math::multiply;

fn main() {
    // Use imported functions directly
    let x = add(5, 10);
    print(x);  // should be 15

    let y = multiply(3, 4);
    print(y);  // should be 12
}
