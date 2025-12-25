// Test derive Clone macro

#[derive(Clone)]
struct Point {
    x: i64,
    y: i64,
}

impl Point {
    fn new(x: i64, y: i64) -> Point {
        Point { x, y }
    }
}

fn main() {
    let p1 = Point::new(10, 20);
    let p2 = p1.clone();

    print(p1.x);  // should be 10
    print(p1.y);  // should be 20
    print(p2.x);  // should be 10
    print(p2.y);  // should be 20
}
