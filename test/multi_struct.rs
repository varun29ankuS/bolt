// Test multiple struct instances without constructor functions

struct Point {
    x: i64,
    y: i64,
}

impl Point {
    fn sum(&self) -> i64 {
        self.x + self.y
    }

    fn x(&self) -> i64 {
        self.x
    }
}

struct Counter {
    value: i64,
}

impl Counter {
    fn get(&self) -> i64 {
        self.value
    }

    fn add(&self, n: i64) -> i64 {
        self.value + n
    }
}

fn main() -> i64 {
    // Create structs inline
    let p1 = Point { x: 10, y: 20 };
    let p2 = Point { x: 5, y: 3 };

    let sum1 = p1.sum();  // 30
    let x1 = p1.x();      // 10
    let sum2 = p2.sum();  // 8

    let c = Counter { value: 50 };
    let v = c.get();      // 50
    let v2 = c.add(25);   // 75

    // Result: 30 + 10 + 8 + 50 + 75 = 173
    sum1 + x1 + sum2 + v + v2
}
