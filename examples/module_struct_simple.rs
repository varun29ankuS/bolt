// Simple struct in module test - no method, just field access

mod mymod {
    pub struct Point {
        pub x: i64,
        pub y: i64,
    }
}

fn main() {
    let p = mymod::Point { x: 10, y: 20 };
    print(p.x);  // should be 10
    print(p.y);  // should be 20
}
