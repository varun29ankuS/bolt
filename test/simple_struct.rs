// Simple struct test

struct Point {
    x: i64,
    y: i64,
}

impl Point {
    fn sum(&self) -> i64 {
        self.x + self.y
    }
}

fn main() -> i64 {
    let p = Point { x: 10, y: 20 };
    p.sum()  // should return 30
}
