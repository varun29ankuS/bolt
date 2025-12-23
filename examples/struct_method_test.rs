// Test user-defined struct methods with the new MethodTable resolution

struct Point {
    x: i64,
    y: i64,
}

impl Point {
    fn distance_squared(&self) -> i64 {
        self.x * self.x + self.y * self.y
    }
}

fn main() {
    let p1 = Point { x: 3, y: 4 };

    // Test method call - should resolve via MethodTable
    let dist = p1.distance_squared();
    print(dist);  // Should print 25 (3*3 + 4*4)
}
