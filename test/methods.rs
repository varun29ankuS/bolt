struct Point {
    x: i64,
    y: i64,
}

impl Point {
    fn sum(&self) -> i64 {
        self.x + self.y
    }

    fn new(x: i64, y: i64) -> Point {
        Point { x: x, y: y }
    }
}

fn main() -> i64 {
    let p = Point::new(15, 25);
    p.sum()
}
