// Test match on enums (unit variants)

enum Color {
    Red,
    Green,
    Blue,
}

fn color_value(c: Color) -> i32 {
    match c {
        Color::Red => 1,
        Color::Green => 2,
        Color::Blue => 3,
    }
}

enum Status {
    Ok,
    Error,
    Pending,
}

fn main() {
    let r = Color::Red;
    let g = Color::Green;
    let b = Color::Blue;

    print(color_value(r));
    print(color_value(g));
    print(color_value(b));

    // Direct match in main
    let s = Status::Error;
    let code = match s {
        Status::Ok => 0,
        Status::Error => 1,
        Status::Pending => 2,
    };
    print(code);
}
