// Direct method call on struct literal in module

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
    // Call method directly on struct literal (no variable)
    print((mymod::Point { x: 10, y: 20 }).sum());  // should be 30
}
