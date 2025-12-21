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
    let p = Point { x: 15, y: 25 };
    p.sum()
}
