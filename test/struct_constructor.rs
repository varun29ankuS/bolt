// Test struct constructor

struct Point {
    x: i64,
    y: i64,
}

impl Point {
    fn new(x: i64, y: i64) -> Point {
        Point { x, y }
    }

    fn sum(&self) -> i64 {
        self.x + self.y
    }
}

fn main() -> i64 {
    let p = Point::new(10, 20);
    p.sum()  // should return 30
}
