enum Result {
    Ok(i32),
    Err(i32),
}

fn make_result(ok: bool) -> Result {
    if ok {
        Result::Ok(42)
    } else {
        Result::Err(99)
    }
}

fn main() {
    let r1 = make_result(true);
    match r1 {
        Result::Ok(v) => print(v),    // 42
        Result::Err(e) => print(e),
    }

    let r2 = make_result(false);
    match r2 {
        Result::Ok(v) => print(v),
        Result::Err(e) => print(e),   // 99
    }
}
