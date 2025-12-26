enum Result {
    Ok(i32),
    Err(i32),
}

fn make_ok() -> Result {
    Result::Ok(42)
}

fn make_err() -> Result {
    Result::Err(99)
}

fn main() {
    let r1 = make_ok();
    match r1 {
        Result::Ok(v) => print(v),    // 42
        Result::Err(e) => print(e),
    }

    let r2 = make_err();
    match r2 {
        Result::Ok(v) => print(v),
        Result::Err(e) => print(e),   // 99
    }
}
