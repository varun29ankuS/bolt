enum Result {
    Ok(i64),
    Err(i64),
}

fn divide(a: i64, b: i64) -> Result {
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
