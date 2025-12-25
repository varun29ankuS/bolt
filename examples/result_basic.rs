// Result enum test (user-defined, like Option)

enum Result {
    Ok(i32),
    Err(i32),
}

fn divide(a: i32, b: i32) -> Result {
    if b == 0 {
        Result::Err(-1)
    } else {
        Result::Ok(a / b)
    }
}

fn main() {
    let r1 = divide(10, 2);
    match r1 {
        Result::Ok(v) => print(v),    // 5
        Result::Err(e) => print(e),
    }

    let r2 = divide(10, 0);
    match r2 {
        Result::Ok(v) => print(v),
        Result::Err(e) => print(e),   // -1
    }
}
