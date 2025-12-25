// Module with standalone function returning i64 (no structs)

mod math {
    pub fn add(x: i64, y: i64) -> i64 {
        x + y
    }
}

fn main() {
    print(math::add(10, 20));  // should be 30
}
