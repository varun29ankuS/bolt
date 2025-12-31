// Test external crate resolution
use serde::Serialize;
use std::clone::Clone;

#[derive(Clone)]
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let p = Point { x: 10, y: 20 };
    let p2 = p.clone();
    println!("Point: {}, {}", p2.x, p2.y);
}
