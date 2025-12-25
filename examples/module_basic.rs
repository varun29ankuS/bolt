// Basic module test

mod math {
    pub fn add(a: i64, b: i64) -> i64 {
        a + b
    }

    pub fn mul(a: i64, b: i64) -> i64 {
        a * b
    }
}

fn main() {
    print(math::add(10, 20));  // 30
    print(math::mul(5, 6));    // 30
}
