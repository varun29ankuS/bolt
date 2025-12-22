// Test nested modules

mod geometry {
    fn distance(x: i64, y: i64) -> i64 {
        x + y  // simplified
    }

    mod shapes {
        fn triangle_area(base: i64, height: i64) -> i64 {
            base * height / 2
        }
    }
}

fn main() -> i64 {
    let dist = geometry::distance(5, 10);  // 15
    let area = geometry::shapes::triangle_area(6, 4);  // 12

    dist + area  // 15 + 12 = 27
}
