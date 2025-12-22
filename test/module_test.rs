// Test inline modules

mod math {
    fn add(a: i64, b: i64) -> i64 {
        a + b
    }

    fn multiply(a: i64, b: i64) -> i64 {
        a * b
    }
}

fn main() -> i64 {
    let sum = math::add(10, 20);      // 30
    let prod = math::multiply(5, 6);  // 30
    sum + prod  // 60
}
