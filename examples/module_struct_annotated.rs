// Struct in module with explicit type annotation

mod mymod {
    pub struct Point {
        pub x: i64,
        pub y: i64,
    }

    impl Point {
        pub fn sum(&self) -> i64 {
            self.x + self.y
        }
    }
}

fn main() {
    // Use struct literal directly with variable (no method call first)
    let p: mymod::Point = mymod::Point { x: 10, y: 20 };
    print(p.sum());  // should be 30
}
