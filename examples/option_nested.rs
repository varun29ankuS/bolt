// Test Level 2: Some(Struct{field}) extraction

struct Point {
    x: i32,
    y: i32,
}

enum Option {
    Some(Point),
    None,
}

fn main() {
    let opt = Some(Point { x: 10, y: 20 });

    match opt {
        Some(Point { x, y }) => print(x + y),  // 30
        None => print(0),
    }
}
