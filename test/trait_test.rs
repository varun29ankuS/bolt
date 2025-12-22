// Test trait implementation and method resolution

struct Point {
    x: i64,
    y: i64,
}

impl Point {
    fn new(x: i64, y: i64) -> Point {
        Point { x, y }
    }

    fn x(&self) -> i64 {
        self.x
    }

    fn y(&self) -> i64 {
        self.y
    }

    fn sum(&self) -> i64 {
        self.x + self.y
    }

    fn add(&self, other: Point) -> i64 {
        self.x + self.y + other.x + other.y
    }
}

struct Counter {
    value: i64,
}

impl Counter {
    fn new() -> Counter {
        Counter { value: 0 }
    }

    fn get(&self) -> i64 {
        self.value
    }

    fn add(&self, n: i64) -> i64 {
        self.value + n
    }
}

fn main() -> i64 {
    // Test Point methods
    let p1 = Point::new(10, 20);
    let p2 = Point { x: 5, y: 3 };

    let sum1 = p1.sum();  // 30
    let x1 = p1.x();      // 10
    let y1 = p1.y();      // 20

    // Test Counter methods
    let c = Counter::new();
    let v = c.get();      // 0
    let v2 = c.add(100);  // 100

    // Test method chaining resolution
    let p3 = Point { x: 1, y: 2 };
    let sum3 = p3.sum();  // 3

    // Return combined result
    sum1 + x1 + y1 + v + v2 + sum3  // 30 + 10 + 20 + 0 + 100 + 3 = 163
}
