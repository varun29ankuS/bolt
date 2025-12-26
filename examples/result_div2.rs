enum Result {
    Ok(i32),
    Err(i32),
}

fn divide(a: i32, b: i32) -> Result {
    if b == 0 {
        Result::Err(-1)
    } else {
        let q = a / b;
        Result::Ok(q)
    }
}

fn main() {
    let r1 = divide(10, 2);
    match r1 {
        Result::Ok(v) => print(v),
        Result::Err(e) => print(e),
    }
}
