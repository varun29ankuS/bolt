enum Result {
    Ok(i32),
    Err(i32),
}

fn main() {
    let r1: Result = Result::Ok(42);
    match r1 {
        Result::Ok(v) => print(v),    // 42
        Result::Err(e) => print(e),
    }

    let r2: Result = Result::Err(99);
    match r2 {
        Result::Ok(v) => print(v),
        Result::Err(e) => print(e),   // 99
    }
}
